from langgraph.graph import StateGraph, END
from .state import AgentState, AgentState2
from .portfolio_analyzer import portfolio_analyzer
from .news_analyzer import news_analyzer
from .technical_analyzer import technical_analyzer
from .recommendation_engine import recommendation_engine
from .rag_analyzer import rag_analyzer
from .zacks_analyzer import zacks_analyzer
from .new_investment_recommender import new_investment_recommender
from .new_stock_analyzer import new_stock_analyzer
from .portfolio_fit_evaluator import portfolio_fit_evaluator

# def setup_graph():
#     """Set up and compile the LangGraph workflow with a direct sequential flow.
    
#     The workflow follows a fixed sequence without a supervisor:
#     PortfolioAnalyzer -> TechnicalAnalyzer -> NewsAnalyzer -> RAGAnalyzer -> RecommendationEngine -> END
#     """
#     # Define state schema
#     workflow = StateGraph(AgentState)
    
#     # Add nodes
#     workflow.add_node("PortfolioAnalyzer", portfolio_analyzer)
#     workflow.add_node("NewsAnalyzer", news_analyzer)
#     workflow.add_node("TechnicalAnalyzer", technical_analyzer)
#     workflow.add_node("RAGAnalyzer", rag_analyzer)
#     workflow.add_node("RecommendationEngine", recommendation_engine)
    
#     # Define direct sequential edges between agents
#     workflow.add_edge("PortfolioAnalyzer", "TechnicalAnalyzer")
#     workflow.add_edge("TechnicalAnalyzer", "NewsAnalyzer")
#     workflow.add_edge("NewsAnalyzer", "RAGAnalyzer")
#     workflow.add_edge("RAGAnalyzer", "RecommendationEngine")
#     workflow.add_edge("RecommendationEngine", END)
    
#     # Set entry point
#     workflow.set_entry_point("PortfolioAnalyzer")
    
#     return workflow.compile()

def setup_graph_with_tracking(progress_callback=None):
    """Set up and compile the LangGraph workflow with progress tracking and direct sequential flow.
    
    The workflow follows a fixed sequence without a supervisor:
    TechnicalAnalyzer -> PortfolioAnalyzer -> NewsAnalyzer -> RAGAnalyzer -> RecommendationEngine -> END
    """
    # Define state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes with progress tracking wrappers if callback is provided
    workflow.add_node("PortfolioAnalyzer", 
                     lambda state: progress_callback(portfolio_analyzer(state)) if progress_callback else portfolio_analyzer(state))
    
    workflow.add_node("NewsAnalyzer", 
                     lambda state: progress_callback(news_analyzer(state)) if progress_callback else news_analyzer(state))
    
    workflow.add_node("TechnicalAnalyzer", 
                     lambda state: progress_callback(technical_analyzer(state)) if progress_callback else technical_analyzer(state))
    
    workflow.add_node("RAGAnalyzer", 
                     lambda state: progress_callback(rag_analyzer(state)) if progress_callback else rag_analyzer(state))
    
    workflow.add_node("RecommendationEngine", 
                     lambda state: progress_callback(recommendation_engine(state)) if progress_callback else recommendation_engine(state))
    
    # Define direct sequential edges between agents
    workflow.add_edge("TechnicalAnalyzer", "PortfolioAnalyzer")
    workflow.add_edge("PortfolioAnalyzer", "NewsAnalyzer")
    workflow.add_edge("NewsAnalyzer", "RAGAnalyzer")
    workflow.add_edge("RAGAnalyzer", "RecommendationEngine")
    workflow.add_edge("RecommendationEngine", END)
    
    # Set entry point
    workflow.set_entry_point("TechnicalAnalyzer")
    
    return workflow.compile() 

def setup_new_investments_graph(progress_callback=None):
    """Set up and compile a LangGraph workflow for finding new investment opportunities.
    
    The workflow follows a fixed sequence:
    ZacksAnalyzer -> NewStockAnalyzer -> PortfolioFitEvaluator -> NewInvestmentRecommender -> END
    
    This workflow:
    1. Fetches high-ranked stocks from Zacks
    2. Performs technical analysis only on these new stocks
    3. Evaluates how these new stocks fit into the existing portfolio
    4. Recommends only stocks that have been properly analyzed
    """
    # Define state schema
    workflow = StateGraph(AgentState2)
    
    # Add nodes with progress tracking wrappers if callback is provided
    workflow.add_node("ZacksAnalyzer", 
                     lambda state: progress_callback(zacks_analyzer(state)) if progress_callback else zacks_analyzer(state))
    
    # This node will analyze only the new high-ranked stocks, not the entire portfolio
    workflow.add_node("NewStockAnalyzer", 
                     lambda state: progress_callback(new_stock_analyzer(state)) if progress_callback else new_stock_analyzer(state))
    
    # This node evaluates how new stocks fit into the existing portfolio context
    workflow.add_node("PortfolioFitEvaluator", 
                     lambda state: progress_callback(portfolio_fit_evaluator(state)) if progress_callback else portfolio_fit_evaluator(state))
    
    # This node recommends only stocks that have been properly analyzed
    workflow.add_node("NewInvestmentRecommender", 
                     lambda state: progress_callback(new_investment_recommender(state)) if progress_callback else new_investment_recommender(state))
    
    # Define direct sequential edges between agents
    workflow.add_edge("ZacksAnalyzer", "NewStockAnalyzer")
    workflow.add_edge("NewStockAnalyzer", "PortfolioFitEvaluator")
    workflow.add_edge("PortfolioFitEvaluator", "NewInvestmentRecommender")
    workflow.add_edge("NewInvestmentRecommender", END)
    
    # Set entry point
    workflow.set_entry_point("ZacksAnalyzer")
    
    return workflow.compile() 