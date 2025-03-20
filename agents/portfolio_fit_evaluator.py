import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from typing import Dict, List
from .state import AgentState, AgentState2

def create_portfolio_fit_evaluator():
    """Create a portfolio fit evaluator using LangChain."""
    # Define the function schema for structured output
    function_def = {
        "name": "evaluate_portfolio_fit",
        "description": "Evaluate how new stocks fit into the existing portfolio",
        "parameters": {
            "type": "object",
            "properties": {
                "portfolio_fit_summary": {
                    "type": "string",
                    "description": "Summary of how the new stocks fit into the existing portfolio"
                },
                "evaluated_stocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            },
                            "portfolio_fit_score": {
                                "type": "integer",
                                "description": "Score from 1-10 indicating how well the stock fits in the portfolio (10 being best)",
                                "minimum": 1,
                                "maximum": 10
                            },
                            "diversification_impact": {
                                "type": "string",
                                "description": "How this stock would impact portfolio diversification"
                            },
                            "risk_impact": {
                                "type": "string",
                                "description": "How this stock would impact portfolio risk"
                            },
                            "sector_balance": {
                                "type": "string",
                                "description": "How this stock would affect sector balance in the portfolio"
                            },
                            "recommendation": {
                                "type": "string",
                                "enum": ["STRONG_FIT", "MODERATE_FIT", "POOR_FIT"],
                                "description": "Overall recommendation for portfolio fit"
                            }
                        },
                        "required": ["ticker", "portfolio_fit_score", "diversification_impact", "risk_impact", "sector_balance", "recommendation"]
                    }
                }
            },
            "required": ["portfolio_fit_summary", "evaluated_stocks"]
        }
    }
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert portfolio manager specializing in evaluating how new investments fit into existing portfolios.
        
Your task is to analyze new high-ranked stocks and evaluate how well they would fit into the user's existing portfolio.

Consider the following when evaluating portfolio fit:
1. Diversification across sectors, industries, and asset classes
2. Risk profile and how it aligns with the user's risk tolerance
3. Correlation with existing holdings
4. Impact on overall portfolio performance
5. Balance between growth and value investments
6. Geographic diversification if applicable

Provide a detailed evaluation of each stock's fit within the portfolio context.
"""),
        ("human", """
I need to evaluate how these new high-ranked stocks would fit into my existing portfolio:

User's Portfolio:
{portfolio_data}

User's Risk Tolerance: {risk_level}
User's Investment Goals: {investment_goals}

Technical Analysis of Existing Portfolio:
{technical_analysis}

New High-Ranked Stocks Analysis:
{new_stock_analysis}

Please evaluate how each new stock would fit into my existing portfolio.
""")
    ])
    
    # Create the LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create the structured output chain
    chain = prompt | llm.bind_functions(functions=[function_def], function_call={"name": "evaluate_portfolio_fit"}) | JsonOutputFunctionsParser()
    
    return chain

def portfolio_fit_evaluator(state: AgentState2) -> AgentState2:
    """Evaluates how new high-ranked stocks fit into the existing portfolio context."""
    try:
        # Get the portfolio data
        portfolio = state["portfolio_data"]
        
        # Get the technical analysis of the existing portfolio
        technical_analysis = state.get("technical_analysis", {})
        
        # Get the analysis of new stocks
        new_stock_analysis = state.get("new_stock_analysis", {})
        
        # If no new stocks to evaluate, return early
        if not new_stock_analysis:
            state["messages"] = state.get("messages", []) + [{
                "role": "ai",
                "content": "[PortfolioFitEvaluator] No new stocks to evaluate for portfolio fit."
            }]
            state["portfolio_fit"] = {
                "portfolio_fit_summary": "No new stocks to evaluate.",
                "evaluated_stocks": []
            }
            return state
        
        # Get user preferences
        risk_level = state.get("risk_level", 5)
        investment_goals = state.get("investment_goals", "Growth")
        
        # Create the evaluator
        evaluator = create_portfolio_fit_evaluator()
        
        # Generate evaluation
        result = evaluator.invoke({
            "portfolio_data": portfolio,
            "risk_level": risk_level,
            "investment_goals": investment_goals,
            "technical_analysis": technical_analysis,
            "new_stock_analysis": new_stock_analysis
        })
        
        # Ensure we have the required fields
        if "portfolio_fit_summary" not in result:
            result["portfolio_fit_summary"] = f"Evaluation of {len(new_stock_analysis)} new stocks for portfolio fit."
        
        if "evaluated_stocks" not in result:
            result["evaluated_stocks"] = []
            
    except Exception as e:
        # Handle any errors in the evaluation
        state["messages"] = state.get("messages", []) + [{
            "role": "ai",
            "content": f"[PortfolioFitEvaluator] Error evaluating portfolio fit: {str(e)}"
        }]
        
        result = {
            "portfolio_fit_summary": "Error in evaluation",
            "evaluated_stocks": []
        }
    
    # Update state with portfolio fit evaluation
    state["portfolio_fit"] = result
    
    # Add message to communication
    state["messages"] = state.get("messages", []) + [{
        "role": "ai",
        "content": f"[PortfolioFitEvaluator] I've evaluated how {len(result.get('evaluated_stocks', []))} new stocks would fit into your existing portfolio."
    }]
    
    return state 