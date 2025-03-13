from typing import TypedDict, Dict, Any, List, Optional, Annotated
import operator

# Define state types for LangGraph
class AgentState(TypedDict):
    # Input data
    portfolio_data: Dict[str, Any]
    risk_level: str
    investment_goals: str
    
    # Analysis data
    technical_analysis: Dict[str, Any]
    news_analysis: List[Dict[str, Any]]
    fundamental_analysis: Dict[str, Any]
    
    # RAG data
    rag_context: Optional[str]
    
    # Agent communication
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Output data
    recommendations: List[Dict[str, str]]
    portfolio_strengths: List[str]
    portfolio_weaknesses: List[str]
    new_investments: List[Dict[str, Any]]
    allocation_advice: str
    risk_assessment: str
    final_report: str 


    # Define state types for LangGraph
class AgentState2(TypedDict):
    # Input data
    portfolio_data: Dict[str, Any]
    risk_level: str
    investment_goals: str
    
    # Analysis data
    technical_analysis: Dict[str, Any]
    news_analysis: List[Dict[str, Any]]
    fundamental_analysis: Dict[str, Any]
    
    # New investment workflow data
    high_rank_stocks: List[Dict[str, Any]]
    new_stock_analysis: Dict[str, Any]
    portfolio_fit: Dict[str, Any]
    
    # RAG data
    rag_context: Optional[str]
    
    # Agent communication
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Output data
    recommendations: List[Dict[str, str]]
    portfolio_strengths: List[str]
    portfolio_weaknesses: List[str]
    new_investments: List[Dict[str, Any]]
    new_investment_summary: str
    allocation_advice: str
    risk_assessment: str
    final_report: str 