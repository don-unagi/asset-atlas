from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from operator import itemgetter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .state import AgentState
import os

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l")

# Initialize Vector DB
vector_db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Define RAG prompt template for technical analysis interpretation
TECHNICAL_RAG_PROMPT = """
CONTEXT:
{context}

TECHNICAL INDICATORS:
{technical_data}

USER PORTFOLIO INFORMATION:
Risk Level: {risk_level}
Investment Goals: {investment_goals}

You are a financial advisor with expertise in both technical and fundamental analysis. Use the available context from "Security Analysis" by Graham and Dodd along with the technical indicators to provide insights on the stock.

For each stock, analyze:
1. What the technical indicators suggest about current market sentiment and potential price movements
2. How the fundamental data aligns with value investing principles
3. Whether the stock appears to be undervalued, fairly valued, or overvalued
4. How this stock fits within the user's risk profile and investment goals

Provide a balanced analysis that considers both technical signals and fundamental principles.
"""

technical_rag_prompt = ChatPromptTemplate.from_template(TECHNICAL_RAG_PROMPT)

# Define RAG prompt template for portfolio insights
PORTFOLIO_RAG_PROMPT = """
CONTEXT:
{context}

USER PORTFOLIO INFORMATION:
Risk Level: {risk_level}
Investment Goals: {investment_goals}
Portfolio: {portfolio}

You are a financial advisor with expertise in fundamental analysis. Use the available context from "Security Analysis" by Graham and Dodd to provide insights on the user's portfolio and investment strategy.
Focus on applying fundamental analysis principles to the user's specific situation.

Answer the following questions:
1. What would Graham and Dodd recommend for this investor with their risk level and goals?
2. What fundamental analysis principles should be applied to this portfolio?
3. How should this investor think about value investing in today's market?

Provide your answers in a clear, concise format.
"""

portfolio_rag_prompt = ChatPromptTemplate.from_template(PORTFOLIO_RAG_PROMPT)

# Define a batch analysis prompt for multiple stocks
BATCH_ANALYSIS_PROMPT = """
CONTEXT:
{context}

USER PORTFOLIO INFORMATION:
Risk Level: {risk_level}
Investment Goals: {investment_goals}

STOCKS TO ANALYZE:
{stocks_data}

You are a financial advisor with expertise in both technical and fundamental analysis. Use the available context from "Security Analysis" by Graham and Dodd along with the technical indicators to provide insights on each stock.

For each stock, provide a brief analysis that includes:
1. What the technical indicators suggest about current market sentiment
2. Whether the stock appears to be undervalued, fairly valued, or overvalued
3. How this stock fits within the user's risk profile and investment goals

Format your response as a JSON-like structure with ticker symbols as keys and analysis as values.
"""

batch_analysis_prompt = ChatPromptTemplate.from_template(BATCH_ANALYSIS_PROMPT)

def setup_technical_rag_chain():
    """Set up the RAG chain for technical analysis interpretation."""
    # Initialize LLM - using a faster model for individual stock analysis
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Create retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Create RAG chain for technical analysis
    technical_rag_chain = (
        {
            "context": lambda x: retriever.invoke("technical analysis indicators interpretation value investing"), 
            "technical_data": itemgetter("technical_data"),
            "risk_level": itemgetter("risk_level"),
            "investment_goals": itemgetter("investment_goals")
        }
        | technical_rag_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return technical_rag_chain

def setup_portfolio_rag_chain():
    """Set up the RAG chain for portfolio insights."""
    # Initialize LLM - using GPT-4o-mini for better performance while maintaining quality
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # Create retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    # Create RAG chain for portfolio insights
    portfolio_rag_chain = (
        {
            "context": lambda x: retriever.invoke("value investing portfolio analysis Graham Dodd"), 
            "risk_level": itemgetter("risk_level"),
            "investment_goals": itemgetter("investment_goals"),
            "portfolio": itemgetter("portfolio")
        }
        | portfolio_rag_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return portfolio_rag_chain

def setup_batch_analysis_chain():
    """Set up the RAG chain for batch stock analysis."""
    # Initialize LLM - using GPT-4o-mini for better performance while maintaining quality
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # Create retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    # Create RAG chain for batch analysis
    batch_analysis_chain = (
        {
            "context": lambda x: retriever.invoke("technical analysis fundamental analysis value investing"), 
            "stocks_data": itemgetter("stocks_data"),
            "risk_level": itemgetter("risk_level"),
            "investment_goals": itemgetter("investment_goals")
        }
        | batch_analysis_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return batch_analysis_chain

# Initialize the RAG chains
technical_rag_chain = setup_technical_rag_chain()
portfolio_rag_chain = setup_portfolio_rag_chain()
batch_analysis_chain = setup_batch_analysis_chain()

def get_stock_interpretation(technical_data_str, risk_level, investment_goals):
    """Get RAG-based interpretation for a stock's technical data."""
    return technical_rag_chain.invoke({
        "technical_data": technical_data_str,
        "risk_level": risk_level,
        "investment_goals": investment_goals
    })

def get_portfolio_insights(portfolio_str, risk_level, investment_goals):
    """Get RAG-based insights for a portfolio."""
    return portfolio_rag_chain.invoke({
        "risk_level": risk_level,
        "investment_goals": investment_goals,
        "portfolio": portfolio_str
    })

def rag_analyzer(state: AgentState) -> AgentState:
    """Performs RAG-based analysis on technical data and portfolio."""
    portfolio = state["portfolio_data"]
    risk_level = state["risk_level"]
    investment_goals = state["investment_goals"]
    technical_analysis = state.get("technical_analysis", {})
    
    # Format portfolio for RAG
    portfolio_str = ""
    for ticker, data in portfolio.items():
        portfolio_str += f"{ticker}: {data['shares']} shares at ${data['purchase_price']:.2f}, "
        portfolio_str += f"current value: ${data['value']:.2f}, "
        portfolio_str += f"gain/loss: {data['gain_loss_pct']:.2f}%\n"
    
    # Generate portfolio-level insights using RAG (single call instead of multiple)
    portfolio_insights_text = get_portfolio_insights(
        portfolio_str, 
        risk_level, 
        investment_goals
    )
    
    # Parse the portfolio insights into structured format
    lines = portfolio_insights_text.strip().split('\n')
    portfolio_insights = []
    
    # Extract questions and answers from the response
    current_question = None
    current_answer = []
    
    for line in lines:
        if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
            # If we have a previous question, save it
            if current_question is not None:
                portfolio_insights.append({
                    "question": current_question,
                    "response": '\n'.join(current_answer).strip()
                })
                current_answer = []
            
            # Extract new question
            parts = line.split(':', 1)
            if len(parts) > 1:
                current_question = parts[0].strip()
                current_answer.append(parts[1].strip())
            else:
                current_question = line.strip()
        elif current_question is not None:
            current_answer.append(line)
    
    # Add the last question/answer if exists
    if current_question is not None and current_answer:
        portfolio_insights.append({
            "question": current_question,
            "response": '\n'.join(current_answer).strip()
        })
    
    # If we couldn't parse properly, create a fallback structure
    if not portfolio_insights:
        portfolio_insights = [
            {"question": "Portfolio Analysis", "response": portfolio_insights_text}
        ]
    
    # Only process technical analysis if there are stocks to analyze
    if technical_analysis:
        # Prepare batch analysis data instead of individual calls
        stocks_data = ""
        for ticker, data in technical_analysis.items():
            if "error" not in data:
                stocks_data += f"Stock: {ticker}\n"
                stocks_data += f"Current Price: ${data['current_price']:.2f}\n"
                stocks_data += "Technical Indicators:\n"
                for key, value in data['technical_indicators'].items():
                    stocks_data += f"  {key}: {value}\n"
                stocks_data += "Fundamental Data:\n"
                for key, value in data['fundamental_data'].items():
                    if value is not None:
                        stocks_data += f"  {key}: {value}\n"
                stocks_data += "\n---\n\n"
        
        # Only make the batch analysis call if we have stocks to analyze
        if stocks_data:
            try:
                # Get batch analysis for all stocks in one call
                batch_analysis_result = batch_analysis_chain.invoke({
                    "stocks_data": stocks_data,
                    "risk_level": risk_level,
                    "investment_goals": investment_goals
                })
                
                # Parse the batch results and update technical_analysis
                # This is a simplified parsing - in production you might want more robust parsing
                current_ticker = None
                current_analysis = []
                
                for line in batch_analysis_result.split('\n'):
                    if ':' in line and line.split(':')[0].strip() in technical_analysis:
                        # If we have a previous ticker, save its analysis
                        if current_ticker is not None and current_ticker in technical_analysis:
                            technical_analysis[current_ticker]["rag_interpretation"] = '\n'.join(current_analysis).strip()
                            current_analysis = []
                        
                        # Start new ticker
                        current_ticker = line.split(':')[0].strip()
                        current_analysis.append(line.split(':', 1)[1].strip())
                    elif current_ticker is not None:
                        current_analysis.append(line)
                
                # Add the last ticker's analysis
                if current_ticker is not None and current_ticker in technical_analysis:
                    technical_analysis[current_ticker]["rag_interpretation"] = '\n'.join(current_analysis).strip()
            except Exception as e:
                # Fallback to a simpler approach if batch processing fails
                for ticker, data in technical_analysis.items():
                    if "error" not in data:
                        technical_analysis[ticker]["rag_interpretation"] = f"Analysis unavailable due to processing error: {str(e)}"
    
    # Combine portfolio insights
    combined_insights = "\n\n".join([f"Q: {insight['question']}\nA: {insight['response']}" for insight in portfolio_insights])
    
    # Update state with RAG insights
    state["technical_analysis"] = technical_analysis
    state["rag_context"] = combined_insights
    state["fundamental_analysis"] = state.get("fundamental_analysis", {})
    state["fundamental_analysis"]["security_analysis_insights"] = portfolio_insights
    
    # Add message to communication
    state["messages"] = state.get("messages", []) + [{
        "role": "ai",
        "content": f"[RAGAnalyzer] I've provided insights on your portfolio and individual stocks based on value investing principles from Security Analysis by Graham and Dodd."
    }]
    
    return state 