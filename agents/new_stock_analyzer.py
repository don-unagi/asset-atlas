import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .state import AgentState, AgentState2
from .rag_analyzer import batch_analysis_chain

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence."""
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }

def get_technical_indicators(ticker_obj, hist):
    """Calculate technical indicators for a stock without making judgments."""
    if hist.empty:
        return None
    
    # Calculate basic technical indicators
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    hist['RSI'] = calculate_rsi(hist['Close'])
    
    # Calculate MACD
    macd = calculate_macd(hist['Close'])
    hist['MACD_Line'] = macd['macd_line']
    hist['MACD_Signal'] = macd['signal_line']
    hist['MACD_Histogram'] = macd['histogram']
    
    # Calculate Bollinger Bands
    hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
    std = hist['Close'].rolling(window=20).std()
    hist['BB_Upper'] = hist['BB_Middle'] + (std * 2)
    hist['BB_Lower'] = hist['BB_Middle'] - (std * 2)
    
    # Get latest values
    latest = hist.iloc[-1]
    
    # Get fundamental data
    info = ticker_obj.info
    
    return {
        'current_price': latest['Close'],
        'technical_indicators': {
            'ma20': latest['MA20'],
            'ma50': latest['MA50'],
            'ma200': latest['MA200'],
            'rsi': latest['RSI'],
            'macd_line': latest['MACD_Line'],
            'macd_signal': latest['MACD_Signal'],
            'macd_histogram': latest['MACD_Histogram'],
            'bb_upper': latest['BB_Upper'],
            'bb_middle': latest['BB_Middle'],
            'bb_lower': latest['BB_Lower'],
            'volume': latest['Volume']
        },
        'fundamental_data': {
            'symbol': ticker_obj.ticker,
            'price': info.get('currentPrice'),
            'pe_ratio': info.get('trailingPE'),
            'peg_ratio': info.get('pegRatio'),
            'debt_to_equity': info.get('debtToEquity'),
            'forward_pe': info.get('forwardPE'),
            'beta': info.get('beta'),
            'return_on_equity': info.get('returnOnEquity'),
            'free_cash_flow': info.get('freeCashflow'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'dividend_yield': info.get('dividendYield'),
            'market_cap': info.get('marketCap'),
            'profit_margins': info.get('profitMargins'),
            'price_to_book': info.get('priceToBook')
        }
    }

def new_stock_analyzer(state: AgentState2) -> AgentState2:
    """Performs technical analysis only on the new high-ranked stocks from Zacks."""
    # Get the high-rank stocks from the Zacks analyzer
    high_rank_stocks = state.get("high_rank_stocks", [])
    
    if not high_rank_stocks:
        state["messages"] = state.get("messages", []) + [{
            "role": "ai",
            "content": "[NewStockAnalyzer] No high-ranked stocks found to analyze."
        }]
        state["new_stock_analysis"] = {}
        return state
    
    # Extract tickers from high-rank stocks
    tickers = [stock['symbol'] for stock in high_rank_stocks]
    analysis_results = {}
    
    # Collect technical data for all new stocks
    for ticker in tickers:
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            
            if not hist.empty:
                # Get comprehensive technical indicators without judgments
                indicators = get_technical_indicators(stock, hist)
                
                if indicators:
                    analysis_results[ticker] = indicators
                else:
                    analysis_results[ticker] = {"error": "Failed to calculate indicators"}
            else:
                analysis_results[ticker] = {"error": "No historical data available"}
                
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            analysis_results[ticker] = {"error": str(e)}
    
    # Add RAG interpretation for new stocks
    risk_level = state.get("risk_level", 5)
    investment_goals = state.get("investment_goals", "Growth")
    
    # Prepare batch analysis data for new stocks
    stocks_data = ""
    for ticker, data in analysis_results.items():
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
            # Get batch analysis for all new stocks in one call
            batch_analysis_result = batch_analysis_chain.invoke({
                "stocks_data": stocks_data,
                "risk_level": risk_level,
                "investment_goals": investment_goals
            })
            
            # Try to parse as JSON first
            try:
                import json
                import re
                
                # Try to find JSON-like content in the response using regex
                json_match = re.search(r'\{[\s\S]*\}', batch_analysis_result)
                if json_match:
                    json_str = json_match.group(0)
                    analysis_data = json.loads(json_str)
                    
                    # Update analysis_results with the parsed JSON
                    for ticker, analysis in analysis_data.items():
                        if ticker in analysis_results:
                            analysis_results[ticker]["rag_interpretation"] = analysis
                else:
                    # Fallback to text parsing approach
                    current_ticker = None
                    current_analysis = []
                    
                    for line in batch_analysis_result.split('\n'):
                        if ':' in line and line.split(':')[0].strip() in analysis_results:
                            # If we have a previous ticker, save its analysis
                            if current_ticker is not None and current_ticker in analysis_results:
                                analysis_results[current_ticker]["rag_interpretation"] = '\n'.join(current_analysis).strip()
                                current_analysis = []
                            
                            # Start new ticker
                            current_ticker = line.split(':')[0].strip()
                            current_analysis.append(line.split(':', 1)[1].strip())
                        elif current_ticker is not None:
                            current_analysis.append(line)
                    
                    # Add the last ticker's analysis
                    if current_ticker is not None and current_ticker in analysis_results:
                        analysis_results[current_ticker]["rag_interpretation"] = '\n'.join(current_analysis).strip()
            
            except (json.JSONDecodeError, Exception) as json_err:
                print(f"Error parsing JSON response for new stocks: {str(json_err)}")
                print(f"Raw response: {batch_analysis_result}")
                
        except Exception as e:
            # Log any errors but continue execution
            import traceback
            print(f"Error in batch analysis for new stocks: {str(e)}")
            print(traceback.format_exc())
    
    # Update state with new stock analysis
    state["new_stock_analysis"] = analysis_results
    
    # Add message to communication
    num_with_rag = sum(1 for ticker in analysis_results if "rag_interpretation" in analysis_results[ticker])
    state["messages"] = state.get("messages", []) + [{
        "role": "ai",
        "content": f"[NewStockAnalyzer] I've calculated technical indicators, gathered fundamental data, and added RAG-based interpretations for {num_with_rag} of {len(analysis_results)} new high-ranked stocks."
    }]
    
    return state 