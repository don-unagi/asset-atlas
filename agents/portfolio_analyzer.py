import yfinance as yf
import numpy as np
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from .state import AgentState
import os

from dotenv import load_dotenv
load_dotenv()

def get_portfolio_data(portfolio):
    """Fetch historical data for portfolio assets."""
    tickers = list(portfolio.keys())
    
    # Get historical data for portfolio assets
    historical_data = {}
    
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="1y")
            historical_data[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            
    return historical_data

def calculate_portfolio_metrics(portfolio, historical_data):
    """Calculate portfolio metrics."""
    # Calculate portfolio metrics
    total_value = sum(portfolio[ticker]["value"] for ticker in portfolio)
    allocations = {ticker: portfolio[ticker]["value"] / total_value for ticker in portfolio}
    
    # Calculate volatility and returns
    returns = {}
    volatility = {}
    for ticker in historical_data:
        if not historical_data[ticker].empty:
            price_data = historical_data[ticker]['Close']
            daily_returns = price_data.pct_change().dropna()
            returns[ticker] = daily_returns.mean() * 252  # Annualized return
            volatility[ticker] = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    
    return {
        "total_value": total_value,
        "allocations": allocations,
        "returns": returns,
        "volatility": volatility
    }

def create_portfolio_analyzer_chain():
    """Create a chain for portfolio analysis using LLM."""
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a portfolio analysis expert. Analyze the following portfolio information and provide insights.
    
    Portfolio Data:
    {portfolio_data}
    
    Portfolio Metrics:
    {portfolio_metrics}
    
    Fundamental Data:
    {fundamental_data}
    
    Risk Level: {risk_level}
    Investment Goals: {investment_goals}
    
    Provide a comprehensive analysis of the portfolio including:
    1. Overall portfolio composition and diversification
    2. Risk assessment based on volatility and beta
    3. Valuation assessment based on fundamental metrics
    4. Alignment with the user's risk level and investment goals
    5. Areas of concern or potential improvement
    
    Your analysis should be detailed, insightful, and actionable.
    """)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Create LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, api_key=openai_api_key)
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    return chain

# Initialize portfolio analyzer chain
portfolio_analyzer_chain = create_portfolio_analyzer_chain()

def portfolio_analyzer(state: AgentState) -> AgentState:
    """Analyzes the current portfolio composition using LLM."""
    portfolio = state["portfolio_data"]
    risk_level = state["risk_level"]
    investment_goals = state["investment_goals"]
    
    # Get portfolio data
    historical_data = get_portfolio_data(portfolio)
    
    # Calculate portfolio metrics
    portfolio_metrics = calculate_portfolio_metrics(portfolio, historical_data)
    
    # Format data for LLM
    portfolio_data_str = ""
    for ticker, data in portfolio.items():
        portfolio_data_str += f"{ticker}: {data['shares']} shares at ${data['purchase_price']:.2f}, "
        portfolio_data_str += f"current value: ${data['value']:.2f}, "
        portfolio_data_str += f"gain/loss: {data['gain_loss_pct']:.2f}%\n"
    
    # Get fundamental data from technical analysis if available
    fundamental_data = {}
    if "technical_analysis" in state:
        for ticker, analysis in state["technical_analysis"].items():
            if "fundamental_data" in analysis:
                fundamental_data[ticker] = analysis["fundamental_data"]
    
    # Format fundamental data for LLM
    fundamental_data_str = ""
    for ticker, data in fundamental_data.items():
        fundamental_data_str += f"{ticker}:\n"
        for key, value in data.items():
            if value is not None:
                fundamental_data_str += f"  {key}: {value}\n"
        fundamental_data_str += "\n"
    
    # Format portfolio metrics for LLM
    portfolio_metrics_str = ""
    portfolio_metrics_str += f"Total Value: ${portfolio_metrics['total_value']:.2f}\n\n"
    
    portfolio_metrics_str += "Allocations:\n"
    for ticker, allocation in portfolio_metrics['allocations'].items():
        portfolio_metrics_str += f"  {ticker}: {allocation*100:.2f}%\n"
    
    portfolio_metrics_str += "\nAnnualized Returns:\n"
    for ticker, ret in portfolio_metrics['returns'].items():
        portfolio_metrics_str += f"  {ticker}: {ret*100:.2f}%\n"
    
    portfolio_metrics_str += "\nAnnualized Volatility:\n"
    for ticker, vol in portfolio_metrics['volatility'].items():
        portfolio_metrics_str += f"  {ticker}: {vol*100:.2f}%\n"
    
    # Get LLM analysis
    analysis = portfolio_analyzer_chain.invoke({
        "portfolio_data": portfolio_data_str,
        "portfolio_metrics": portfolio_metrics_str,
        "fundamental_data": fundamental_data_str,
        "risk_level": risk_level,
        "investment_goals": investment_goals
    })
    
    # Update state
    state["portfolio_analysis"] = {
        "total_value": portfolio_metrics["total_value"],
        "allocations": portfolio_metrics["allocations"],
        "returns": portfolio_metrics["returns"],
        "volatility": portfolio_metrics["volatility"],
        "fundamental_data": fundamental_data,
        "llm_analysis": analysis
    }
    
    # Add message to communication
    state["messages"] = state.get("messages", []) + [{
        "role": "ai",
        "content": f"[PortfolioAnalyzer] I've analyzed your portfolio. Here are my findings:\n\n{analysis}"
    }]
    
    return state
