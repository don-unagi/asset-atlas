import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from typing import Dict, List
from .state import AgentState, AgentState2

def create_new_investment_recommender():
    """Create a new investment recommender using LangChain."""
    # Define the function schema for structured output
    function_def = {
        "name": "generate_new_investment_recommendations",
        "description": "Generate new investment recommendations based on analyzed high-rank stocks and portfolio fit",
        "parameters": {
            "type": "object",
            "properties": {
                "new_investment_summary": {
                    "type": "string",
                    "description": "Summary of the new investment opportunities"
                },
                "new_investments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            },
                            "action": {
                                "type": "string",
                                "enum": ["BUY"],
                                "description": "Recommended action for this stock"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Detailed reasoning for the recommendation"
                            },
                            "priority": {
                                "type": "integer",
                                "description": "Priority level (1-5, where 1 is highest priority)",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["ticker", "action", "reasoning", "priority"]
                    }
                }
            },
            "required": ["new_investment_summary", "new_investments"]
        }
    }
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial advisor specializing in identifying new investment opportunities.
        
Your task is to recommend only high-ranked stocks that have been properly analyzed and evaluated for portfolio fit.

Consider the following when making recommendations:
1. Technical analysis signals (RSI, moving averages, etc.)
2. Fundamental analysis metrics (PE ratios, debt-to-equity, growth rates, etc.)
3. Prioritize stocks that have been evaluated as good fits for the portfolio
4. Consider the user's risk tolerance and investment goals
5. Focus on stocks that complement the existing portfolio and improve diversification
6. Pay close attention to the RAG interpretations

Provide clear, actionable recommendations with detailed reasoning.
         
IMPORTANT: For the reasoning of each recommendation, include a more detailed technical and fundamental analysis section:
    - For technical analysis: Include specific insights about RSI levels, MACD signals, moving average crossovers, and what these indicators suggest about momentum and trend direction.
    - For fundamental analysis: Include specific metrics like P/E ratio compared to industry average, debt-to-equity ratio, earnings growth, and what these metrics suggest about the company's valuation and financial health.
         

"""),
        ("human", """
I need recommendations for new investments based on the following information:

User's Portfolio:
{portfolio_data}

User's Risk Tolerance: {risk_level}
User's Investment Goals: {investment_goals}

High-Ranked Stocks:
{high_rank_stocks}

New Stock Analysis:
{new_stock_analysis}

Portfolio Fit Evaluation:
{portfolio_fit}

Please provide specific recommendations for new investments that would complement my existing portfolio.
Only recommend stocks that have been properly analyzed and evaluated for portfolio fit.
Pay special attention to the RAG interpretations in the new stock analysis.
""")
    ])
    
    # Create the LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create the structured output chain
    chain = prompt | llm.bind_functions(functions=[function_def], function_call={"name": "generate_new_investment_recommendations"}) | JsonOutputFunctionsParser()
    
    return chain

def new_investment_recommender(state: AgentState2) -> AgentState2:
    """Recommends new investments based on analyzed high-rank stocks and portfolio fit evaluation."""
    try:
        # Get the portfolio data
        portfolio = state["portfolio_data"]
        
        # Get the high-rank stocks
        high_rank_stocks = state.get("high_rank_stocks", [])
        
        # Get the new stock analysis
        new_stock_analysis = state.get("new_stock_analysis", {})
        
        # Get the portfolio fit evaluation
        portfolio_fit = state.get("portfolio_fit", {})
        
        # If no stocks have been analyzed or evaluated, return early
        if not new_stock_analysis or not portfolio_fit.get("evaluated_stocks"):
            state["messages"] = state.get("messages", []) + [{
                "role": "ai",
                "content": "[NewInvestmentRecommender] No properly analyzed stocks to recommend."
            }]
            state["new_investment_summary"] = "No properly analyzed stocks to recommend."
            state["new_investments"] = []
            return state
        
        # Get user preferences
        risk_level = state.get("risk_level", 5)
        investment_goals = state.get("investment_goals", "Growth")
        
        # Create the recommender
        recommender = create_new_investment_recommender()
        
        # Generate recommendations
        result = recommender.invoke({
            "portfolio_data": portfolio,
            "risk_level": risk_level,
            "investment_goals": investment_goals,
            "high_rank_stocks": high_rank_stocks,
            "new_stock_analysis": new_stock_analysis,
            "portfolio_fit": portfolio_fit
        })
        
        # Ensure we have the required fields
        if "new_investment_summary" not in result:
            result["new_investment_summary"] = f"Analysis of {len(high_rank_stocks)} high-ranked stocks."
        
        if "new_investments" not in result:
            result["new_investments"] = []
            
    except Exception as e:
        # Handle any errors in the recommendation engine
        state["messages"] = state.get("messages", []) + [{
            "role": "ai",
            "content": f"[NewInvestmentRecommender] Error generating recommendations: {str(e)}"
        }]
        
        result = {
            "new_investment_summary": "Error in analysis",
            "new_investments": []
        }
    
    # Update state with recommendations
    state["new_investment_summary"] = result.get("new_investment_summary", "")
    state["new_investments"] = result.get("new_investments", [])
    
    # Add message to communication
    state["messages"] = state.get("messages", []) + [{
        "role": "ai",
        "content": f"[NewInvestmentRecommender] I've analyzed the properly evaluated stocks and generated {len(result.get('new_investments', []))} recommendations for new investments."
    }]
    
    return state 