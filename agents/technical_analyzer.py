import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .state import AgentState

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

def technical_analyzer(state: AgentState) -> AgentState:
    """Performs comprehensive technical analysis on portfolio assets."""
    portfolio = state["portfolio_data"]
    
    tickers = list(portfolio.keys())
    analysis_results = {}
    
    # Collect technical data for all stocks
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
    
    # Update state with technical analysis
    state["technical_analysis"] = analysis_results
    
    # Add message to communication
    state["messages"] = state.get("messages", []) + [{
        "role": "ai",
        "content": f"[TechnicalAnalyzer] I've calculated technical indicators and gathered fundamental data for your portfolio stocks."
    }]
    
    return state
