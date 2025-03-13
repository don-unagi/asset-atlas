import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Import our agent modules
from agents import AgentState, AgentState2
from agents.graph import setup_graph_with_tracking, setup_new_investments_graph

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Capital Compass",
    page_icon="💰",
    layout="wide"
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, api_key=openai_api_key)

# # Initialize embedding model
# embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l")

# # Initialize Vector DB
# vector_db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Streamlit UI components
st.title("Capital Compass - AI Financial Advisor")
st.subheader("Portfolio Management and Recommendation System")

with st.expander("About this app", expanded=False):
    st.write("""
    This app provides personalized financial advice based on your portfolio, risk tolerance, and investment goals.
    It analyzes technical indicators, relevant news, and market trends to offer tailored recommendations.
    """)

# Sidebar for user inputs
with st.sidebar:
    st.header("Your Profile")
    
    # Risk tolerance selection
    risk_level = st.select_slider(
        "Risk Tolerance",
        options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
        value="Moderate"
    )
    
    # Investment goals
    investment_goals = st.text_area("Investment Goals", "Retirement in 20 years, building wealth, passive income")
    
    st.divider()
    
    # Portfolio input
    st.header("Your Portfolio")
    
    # Default portfolio for demonstration
    default_portfolio = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B'],
        'Shares': [10, 5, 3, 2, 4],
        'Purchase Price': [250.00, 250.00, 250.00, 250.00, 250.00]
    })
    
    # Let user edit the portfolio
    portfolio_df = st.data_editor(
        default_portfolio,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker Symbol"),
            "Shares": st.column_config.NumberColumn("Number of Shares", min_value=0),
            "Purchase Price": st.column_config.NumberColumn("Purchase Price ($)", min_value=0.01, format="$%.2f")
        },
        num_rows="dynamic",
        use_container_width=True
    )
    
    st.divider()
    generate_button = st.button("Generate Recommendations", type="primary", use_container_width=True)

# Initialize session state to store our analysis results
if 'portfolio_analyzed' not in st.session_state:
    st.session_state.portfolio_analyzed = False
    
if 'final_state' not in st.session_state:
    st.session_state.final_state = None
    
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None

# Main content area
if generate_button or st.session_state.portfolio_analyzed:
    # If we're here because of the session state, we don't need to re-run the analysis
    if generate_button:
        # Convert portfolio dataframe to required format
        portfolio_data = {}
        
        # Create a placeholder for the progress indicator
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Show a spinner while processing
        with st.spinner("Analyzing your portfolio and generating recommendations..."):
            progress_placeholder.progress(0, "Starting analysis...")
            status_placeholder.info("Fetching current market data...")
            
            for _, row in portfolio_df.iterrows():
                ticker = row['Ticker']
                shares = row['Shares']
                purchase_price = row['Purchase Price']
                
                try:
                    # Get current price
                    stock = yf.Ticker(ticker)
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                    
                    # Calculate values
                    current_value = current_price * shares
                    purchase_value = purchase_price * shares
                    gain_loss = current_value - purchase_value
                    gain_loss_pct = (gain_loss / purchase_value) * 100
                    
                    # Store in portfolio data
                    portfolio_data[ticker] = {
                        "shares": shares,
                        "purchase_price": purchase_price,
                        "current_price": current_price,
                        "value": current_value,
                        "gain_loss": gain_loss,
                        "gain_loss_pct": gain_loss_pct
                    }
                except Exception as e:
                    st.error(f"Error processing {ticker}: {e}")
            
            # Initialize state
            initial_state = AgentState(
                portfolio_data=portfolio_data,
                risk_level=risk_level,
                investment_goals=investment_goals,
                technical_analysis={},
                news_analysis=[],
                fundamental_analysis={},
                rag_context=None,
                messages=[{
                    "role": "human",
                    "content": f"Please analyze my portfolio with risk level '{risk_level}' and investment goals: '{investment_goals}'."
                }],
                next="",
                recommendations=[],
                portfolio_strengths=[],
                portfolio_weaknesses=[],
                new_investments=[],
                allocation_advice="",
                risk_assessment="",
                final_report=""
            )
            
            # Create a custom callback to track progress
            def track_progress(state):
                # Simple progress tracking based on what's in the state
                # Each node adds its output to the state, so we can use that to determine progress
                
                # Check which stage we're at based on what's in the state
                if "final_report" in state and state["final_report"]:
                    progress_placeholder.progress(100, "Complete")
                    status_placeholder.success("Analysis complete!")
                elif "recommendations" in state and state["recommendations"]:
                    progress_placeholder.progress(90, "Generating Recommendations")
                    status_placeholder.info("Recommendations generated. Finalizing report...")
                elif "rag_context" in state and state["rag_context"]:
                    progress_placeholder.progress(75, "RAG Analysis")
                    status_placeholder.info("Value investing analysis complete. Generating recommendations...")
                elif "news_analysis" in state and state["news_analysis"]:
                    progress_placeholder.progress(60, "News Analysis")
                    status_placeholder.info("News analysis complete. Applying value investing principles...")
                elif "portfolio_analysis" in state and state["portfolio_analysis"]:
                    progress_placeholder.progress(40, "Portfolio Analysis")
                    status_placeholder.info("Portfolio analysis complete. Gathering financial news...")
                elif "technical_analysis" in state and state["technical_analysis"]:
                    progress_placeholder.progress(20, "Technical Analysis")
                    status_placeholder.info("Technical analysis complete. Analyzing portfolio...")
                else:
                    progress_placeholder.progress(0, "Starting analysis...")
                    status_placeholder.info("Initializing technical analysis...")
                
                return state
            
            # Run the graph with progress tracking
            graph = setup_graph_with_tracking(track_progress)
            
            # Run the graph
            final_state = graph.invoke(initial_state)
            
            # Store the final state in session state
            st.session_state.final_state = final_state
            st.session_state.portfolio_analyzed = True
            st.session_state.portfolio_data = portfolio_data
    else:
        # Use the stored final state
        final_state = st.session_state.final_state
        portfolio_data = st.session_state.portfolio_data
    
    # Display the results
    # Display the executive summary for the end user
    st.subheader("Your Investment Portfolio Analysis")
    
    # Check if we have a final report
    if "final_report" in final_state and final_state["final_report"]:
        # Format and display the final report in a clean, professional way
        report = final_state["final_report"]
        st.markdown(report)
    else:
        st.error("Unable to generate portfolio analysis. Please try again.")
        
    # Display portfolio valuations and allocations with gain/loss
    if portfolio_data:
        st.subheader("Portfolio Valuation & Allocation")
        
        # Create a dataframe for portfolio valuation
        valuation_data = []
        total_value = 0
        total_cost = 0
        
        for ticker, data in portfolio_data.items():
            current_value = data['value']
            purchase_value = data['purchase_price'] * data['shares']
            total_value += current_value
            total_cost += purchase_value
            
            valuation_data.append({
                'Ticker': ticker,
                'Shares': data['shares'],
                'Purchase Price': f"${data['purchase_price']:.2f}",
                'Current Price': f"${data['current_price']:.2f}",
                'Cost Basis': f"${purchase_value:.2f}",
                'Current Value': f"${current_value:.2f}",
                'Gain/Loss ($)': f"${data['gain_loss']:.2f}",
                'Gain/Loss (%)': f"{data['gain_loss_pct']:.2f}%"
            })
        
        # Calculate total gain/loss
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
        
        # Display the valuation table
        valuation_df = pd.DataFrame(valuation_data)
        st.dataframe(valuation_df, use_container_width=True)
        
        # Display total portfolio value and gain/loss
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Portfolio Value", f"${total_value:.2f}", f"${total_gain_loss:.2f}")
        with col2:
            st.metric("Total Cost Basis", f"${total_cost:.2f}")
        with col3:
            st.metric("Total Return", f"{total_gain_loss_pct:.2f}%")
        
        # Create and display allocation pie chart
        st.subheader("Portfolio Allocation")
        
        # Prepare data for pie chart
        allocation_data = {}
        for ticker, data in portfolio_data.items():
            allocation_data[ticker] = data['value']
        
        # Create a figure for the pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create the pie chart
        wedges, texts, autotexts = ax.pie(
            allocation_data.values(), 
            labels=allocation_data.keys(),
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white'}
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Set font size for the labels and percentages
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=12)
        
        # Add a title
        plt.title('Portfolio Allocation by Value', fontsize=16)
        
        # Display the pie chart
        st.pyplot(fig)
        
    # Display key recommendations in an easy-to-read format
    if "recommendations" in final_state and final_state["recommendations"]:
        st.subheader("Key Recommendations")
        
        # Create columns for recommendation cards
        cols = st.columns(min(3, len(final_state["recommendations"])))
        
        # Sort recommendations by priority (if available)
        sorted_recommendations = sorted(
            final_state["recommendations"], 
            key=lambda x: x.get("priority", 5)
        )
        
        # Display top recommendations in cards
        for i, rec in enumerate(sorted_recommendations[:3]):  # Show top 3 recommendations
            with cols[i % 3]:
                action = rec.get("action", "")
                ticker = rec.get("ticker", "")
                reasoning = rec.get("reasoning", "")
                
                # Style based on action type
                if action == "BUY":
                    st.success(f"**{action}: {ticker}**")
                elif action == "SELL":
                    st.error(f"**{action}: {ticker}**")
                else:  # HOLD
                    st.info(f"**{action}: {ticker}**")
                    
                # Display reasoning with markdown formatting preserved
                st.markdown(reasoning)
        
        # If there are more than 3 recommendations, add a expander for the rest
        if len(sorted_recommendations) > 3:
            with st.expander("View All Recommendations"):
                for rec in sorted_recommendations[3:]:
                    action = rec.get("action", "")
                    ticker = rec.get("ticker", "")
                    reasoning = rec.get("reasoning", "")
                    
                    # Style based on action type
                    if action == "BUY":
                        st.success(f"**{action}: {ticker}**")
                    elif action == "SELL":
                        st.error(f"**{action}: {ticker}**")
                    else:  # HOLD
                        st.info(f"**{action}: {ticker}**")
                    
                    # Display reasoning with markdown formatting preserved
                    st.markdown(reasoning)
                    st.divider()
    
    # Display relevant news with links if available
    if "news_analysis" in final_state and final_state["news_analysis"]:
        st.subheader("Relevant Market News")
        
        with st.expander("View Market News"):
            for news_item in final_state["news_analysis"]:
                if isinstance(news_item, dict):
                    title = news_item.get("title", "")
                    summary = news_item.get("summary", "")
                    url = news_item.get("url", "")
                    sentiment = news_item.get("sentiment", "neutral")
                    
                    # Style based on sentiment
                    if sentiment.lower() == "positive":
                        st.success(f"**{title}**")
                    elif sentiment.lower() == "negative":
                        st.error(f"**{title}**")
                    else:
                        st.info(f"**{title}**")
                        
                    st.write(summary)
                    if url:
                        st.write(f"[Read more]({url})")
                    st.divider()
    
    # Display portfolio strengths and weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Strengths")
        strengths = final_state.get("portfolio_strengths", [])
        if strengths:
            for strength in strengths:
                st.write(f"✅ {strength}")
        else:
            st.write("No specific strengths identified.")
            
    with col2:
        st.subheader("Areas for Improvement")
        weaknesses = final_state.get("portfolio_weaknesses", [])
        if weaknesses:
            for weakness in weaknesses:
                st.write(f"⚠️ {weakness}")
        else:
            st.write("No specific areas for improvement identified.")
    
    # Display allocation advice and risk assessment
    st.subheader("Investment Strategy")
    
    allocation_advice = final_state.get("allocation_advice", "")
    risk_assessment = final_state.get("risk_assessment", "")
    
    if allocation_advice:
        st.write(f"**Allocation Advice:** {allocation_advice}")
        
    if risk_assessment:
        st.write(f"**Risk Assessment:** {risk_assessment}")
        
    # Add a second generate button for new investment recommendations
    st.divider()
    st.subheader("Discover New Investment Opportunities")
    st.write("Click the button below to discover new investment opportunities that would complement your portfolio.")
    
    if st.button("Generate New Investment Recommendations", key="new_inv_button"):
        # Create a placeholder for the progress bar
        new_inv_progress_placeholder = st.empty()
        new_inv_status_placeholder = st.empty()
        
        # Get portfolio data from session state
        portfolio_data = st.session_state.portfolio_data
        
        # Initialize the state with user inputs
        initial_state2 = AgentState2(
            portfolio_data=portfolio_data,
            risk_level=risk_level,
            investment_goals=investment_goals,
            technical_analysis={},
            news_analysis=[],
            fundamental_analysis={},
            high_rank_stocks=[],
            new_stock_analysis={},
            portfolio_fit={},
            rag_context=None,
            messages=[{
                "role": "human",
                "content": f"Please find new investment opportunities that complement my portfolio with risk level '{risk_level}' and investment goals: '{investment_goals}'."
            }],
            recommendations=[],
            portfolio_strengths=[],
            portfolio_weaknesses=[],
            new_investments=[],
            new_investment_summary="",
            allocation_advice="",
            risk_assessment="",
            final_report=""
        )
        
        # Create a custom callback to track progress
        def track_new_inv_progress(state):
            # Simple progress tracking based on what's in the state
            # Each node adds its output to the state, so we can use that to determine progress
            
            # Check which stage we're at based on what's in the state
            if "new_investments" in state and state["new_investments"]:
                # Everything completed
                new_inv_progress_placeholder.progress(100, "Complete")
                new_inv_status_placeholder.success("Analysis complete!")
            elif "portfolio_fit" in state and state["portfolio_fit"]:
                # Portfolio Fit Evaluation completed
                new_inv_progress_placeholder.progress(75, "Portfolio Fit Evaluation")
                new_inv_status_placeholder.info("Portfolio fit evaluation complete. Generating investment recommendations...")
            elif "new_stock_analysis" in state and state["new_stock_analysis"]:
                # New Stock Analysis completed
                new_inv_progress_placeholder.progress(50, "New Stock Analysis")
                new_inv_status_placeholder.info("Technical analysis of new stocks complete. Evaluating portfolio fit...")
            elif "high_rank_stocks" in state and state["high_rank_stocks"]:
                # Zacks Analysis completed
                new_inv_progress_placeholder.progress(25, "Finding High-Ranked Stocks")
                new_inv_status_placeholder.info("Found high-ranked stocks. Analyzing technical indicators...")
            else:
                # Just starting
                new_inv_progress_placeholder.progress(0, "Starting analysis...")
                new_inv_status_placeholder.info("Searching for high-ranked stocks...")
            
            return state
        
        # Run the graph with progress tracking
        new_inv_graph = setup_new_investments_graph(track_new_inv_progress)
        
        # Initialize progress
        new_inv_progress_placeholder.progress(0, "Starting analysis...")
        new_inv_status_placeholder.info("Finding high-ranked stocks...")
        
        # Run the graph
        new_inv_final_state = new_inv_graph.invoke(initial_state2)
        
        # Clear the progress indicators after completion
        new_inv_progress_placeholder.empty()
        new_inv_status_placeholder.empty()
        
        # Display the new investment recommendations
        st.subheader("New Investment Opportunities")
        
        # Display the summary
        new_investment_summary = new_inv_final_state.get("new_investment_summary", "")
        if new_investment_summary:
            st.write(new_investment_summary)
        
        # Display the new investment recommendations
        new_investments = new_inv_final_state.get("new_investments", [])
        if new_investments:
            # Sort recommendations by priority (lower number = higher priority)
            sorted_investments = sorted(new_investments, key=lambda x: x.get("priority", 999) if isinstance(x, dict) else 999)
            
            # Create a 3-column layout for recommendations
            cols = st.columns(3)
            
            # Display top recommendations
            for i, inv in enumerate(sorted_investments):
                if isinstance(inv, dict):
                    with cols[i % 3]:
                        action = inv.get("action", "")
                        ticker = inv.get("ticker", "")
                        reasoning = inv.get("reasoning", "")
                        
                        # Style based on action type
                        if action == "BUY":
                            st.success(f"**{action}: {ticker}**")
                        elif action == "SELL":
                            st.error(f"**{action}: {ticker}**")
                        else:  # HOLD
                            st.info(f"**{action}: {ticker}**")
                            
                        # Display reasoning with markdown formatting preserved
                        st.markdown(reasoning)
        else:
            st.info("No new investment recommendations were generated. Please try again.")
else:
    # Default display when app first loads
    st.info("Enter your portfolio details and click 'Generate Recommendations' to receive personalized financial advice.")
