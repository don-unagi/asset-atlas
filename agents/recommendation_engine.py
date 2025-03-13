import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from typing import Dict, List
from .state import AgentState

def create_recommendation_engine():
    """Create a recommendation engine using LangChain."""
    # Define the function schema for structured output
    function_def = {
        "name": "generate_recommendations",
        "description": "Generate investment recommendations based on portfolio analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "portfolio_summary": {
                    "type": "string",
                    "description": "Summary of the current portfolio status"
                },
                "recommendations": {
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
                                "enum": ["BUY", "SELL", "HOLD"],
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
                },
                "portfolio_strengths": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Key strength of the portfolio"
                    },
                    "description": "List of portfolio strengths"
                },
                "portfolio_weaknesses": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Key weakness or area for improvement in the portfolio"
                    },
                    "description": "List of portfolio weaknesses or areas for improvement"
                },
                "allocation_advice": {
                    "type": "string",
                    "description": "Advice on portfolio allocation and diversification"
                },
                "risk_assessment": {
                    "type": "string",
                    "description": "Assessment of portfolio risk relative to user's risk tolerance"
                },
                "final_report": {
                    "type": "string",
                    "description": "Comprehensive final report with all recommendations and analysis"
                }
            },
            "required": ["portfolio_summary", "recommendations", "portfolio_strengths", "portfolio_weaknesses", "allocation_advice", "risk_assessment", "final_report"]
        }
    }
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert financial advisor. Based on the following information, provide personalized investment recommendations.
    
    PORTFOLIO:
    {portfolio}
    
    RISK PROFILE:
    {risk_level}
    
    INVESTMENT GOALS:
    {investment_goals}
    
    TECHNICAL ANALYSIS:
    {technical_analysis}
    
    FUNDAMENTAL DATA:
    {fundamental_data}
    
    RELEVANT NEWS:
    {news}
    
    SECURITY ANALYSIS INSIGHTS (Graham & Dodd):
    {rag_insights}
    
    AGENT DISCUSSIONS:
    {agent_discussions}
    
    Provide comprehensive investment recommendations that incorporate:
    1. Technical analysis signals (RSI, moving averages, etc.)
    2. Fundamental analysis metrics (PE ratios, debt-to-equity, growth rates, etc.)
    3. Recent news sentiment and impact
    4. Value investing principles from Graham & Dodd's Security Analysis
    5. Alignment with the user's risk profile and investment goals
    
    For each holding, provide a clear BUY/SELL/HOLD recommendation with detailed reasoning and priority level.
    
    IMPORTANT: For the reasoning of each recommendation, include a more detailed technical and fundamental analysis section:
    - For technical analysis: Include specific insights about RSI levels, MACD signals, moving average crossovers, and what these indicators suggest about momentum and trend direction.
    - For fundamental analysis: Include specific metrics like P/E ratio compared to industry average, debt-to-equity ratio, earnings growth, and what these metrics suggest about the company's valuation and financial health.
    - Use technical jargon appropriately to demonstrate expertise while still being clear.
    - Format the reasoning to clearly separate technical insights from fundamental insights.
    
    Identify 3-5 key strengths and weaknesses of the current portfolio.
    Assess the overall risk and provide allocation advice.
    
    Remember that your recommendations will be presented to a busy executive who needs concise, actionable insights.
    Focus on clarity and brevity in your explanations.
    """)
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    
    # Create chain
    chain = (
        prompt 
        | llm.bind_functions(functions=[function_def], function_call="generate_recommendations")
        | JsonOutputFunctionsParser()
    )
    
    return chain

# Initialize recommendation engine
recommendation_engine_chain = create_recommendation_engine()

def recommendation_engine(state: AgentState) -> AgentState:
    """Generates investment recommendations based on all analyses including RAG insights."""
    portfolio = state["portfolio_data"]
    risk_level = state["risk_level"]
    investment_goals = state["investment_goals"]
    tech_analysis = state.get("technical_analysis", {})
    news = state.get("news_analysis", [])
    rag_insights = state.get("rag_context", "No RAG insights available.")
    messages = state.get("messages", [])
    fundamental_analysis = state.get("fundamental_analysis", {})
    
    # Extract fundamental data and RAG interpretations from technical analysis
    fundamental_data = {}
    rag_interpretations = {}
    
    for ticker in tech_analysis:
        # Extract fundamental data
        if isinstance(tech_analysis[ticker], dict):
            if 'fundamental_data' in tech_analysis[ticker]:
                fundamental_data[ticker] = tech_analysis[ticker]['fundamental_data']
            
            # Extract RAG interpretations
            if 'rag_interpretation' in tech_analysis[ticker]:
                rag_interpretations[ticker] = tech_analysis[ticker]['rag_interpretation']
    
    # Combine RAG interpretations with portfolio-level insights
    combined_rag_insights = rag_insights
    if rag_interpretations:
        combined_rag_insights += "\n\n## Stock-Specific Interpretations:\n\n"
        for ticker, interpretation in rag_interpretations.items():
            combined_rag_insights += f"### {ticker}:\n{interpretation}\n\n"
    
    # Format agent discussions
    agent_discussions = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # Invoke recommendation engine
    try:
        result = recommendation_engine_chain.invoke({
            "portfolio": str(portfolio),
            "risk_level": risk_level,
            "investment_goals": investment_goals,
            "technical_analysis": str(tech_analysis),
            "fundamental_data": str(fundamental_data),
            "news": str(news),
            "rag_insights": combined_rag_insights,
            "agent_discussions": agent_discussions
        })
        
        # Ensure we have all required fields
        if "recommendations" not in result or not result["recommendations"]:
            # Generate default recommendations if none were provided
            result["recommendations"] = []
            for ticker in portfolio:
                result["recommendations"].append({
                    "ticker": ticker,
                    "action": "HOLD",
                    "reasoning": "Default recommendation due to insufficient analysis data.",
                    "priority": 3
                })
        
        if "final_report" not in result or not result["final_report"]:
            # Generate a default final report
            result["final_report"] = f"""
            # Investment Portfolio Analysis Report
            
            ## Portfolio Overview
            Risk Level: {risk_level}
            Investment Goals: {investment_goals}
            
            ## Key Recommendations
            {', '.join([f"{rec['ticker']}: {rec['action']}" for rec in result.get("recommendations", [])])}
            
            ## Summary
            This is a default report generated due to insufficient analysis data.
            """
        
        # Ensure other fields exist
        if "portfolio_strengths" not in result:
            result["portfolio_strengths"] = ["Diversification across multiple assets"]
        
        if "portfolio_weaknesses" not in result:
            result["portfolio_weaknesses"] = ["Potential for improved sector allocation"]
        
        if "allocation_advice" not in result:
            result["allocation_advice"] = "Consider maintaining a balanced portfolio aligned with your risk tolerance."
        
        if "risk_assessment" not in result:
            result["risk_assessment"] = f"Your portfolio appears to be aligned with your {risk_level} risk tolerance."
        
        if "portfolio_summary" not in result:
            result["portfolio_summary"] = f"Portfolio with {len(portfolio)} assets analyzed."
        
    except Exception as e:
        # Handle any errors in the recommendation engine
        print(f"Error in recommendation engine: {str(e)}")
        result = {
            "recommendations": [],
            "final_report": f"Error generating recommendations: {str(e)}",
            "portfolio_strengths": [],
            "portfolio_weaknesses": [],
            "allocation_advice": "",
            "risk_assessment": "",
            "portfolio_summary": "Error in analysis"
        }
    
    # Update state with recommendations and enhanced data
    state["recommendations"] = result["recommendations"]
    state["final_report"] = result["final_report"]
    
    # Add new structured data to state
    state["portfolio_strengths"] = result.get("portfolio_strengths", [])
    state["portfolio_weaknesses"] = result.get("portfolio_weaknesses", [])
    state["allocation_advice"] = result.get("allocation_advice", "")
    state["risk_assessment"] = result.get("risk_assessment", "")
    
    # Add message to communication
    state["messages"] = state.get("messages", []) + [{
        "role": "ai",
        "content": f"[RecommendationEngine] I've generated the following recommendations:\n\n{result.get('portfolio_summary', '')}\n\n"
                   f"Risk Assessment: {result.get('risk_assessment', '')}\n\n"
                   f"Allocation Advice: {result.get('allocation_advice', '')}"
    }]
    
    return state
