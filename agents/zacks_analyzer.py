import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zacks import find_rank_1_stocks
from .state import AgentState, AgentState2

def zacks_analyzer(state: AgentState2) -> AgentState2:
    """Fetches high-rank stocks from Zacks for potential new investments."""
    try:
        # Get 5 rank 1 stocks from Zacks
        high_rank_stocks = find_rank_1_stocks(n=15)

        #TODO if really aggressive use sp400
        
        # Store the high-rank stocks in the state
        state["high_rank_stocks"] = high_rank_stocks
        
        # Add message to communication
        stock_symbols = [stock['symbol'] for stock in high_rank_stocks]
        state["messages"] = state.get("messages", []) + [{
            "role": "ai",
            "content": f"[ZacksAnalyzer] I've identified {len(high_rank_stocks)} high-ranked stocks from Zacks: {', '.join(stock_symbols)}"
        }]
        
    except Exception as e:
        # Handle any errors
        state["messages"] = state.get("messages", []) + [{
            "role": "ai",
            "content": f"[ZacksAnalyzer] Error fetching high-rank stocks: {str(e)}"
        }]
        state["high_rank_stocks"] = []
    
    return state 