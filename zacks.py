import pandas as pd
import urllib.request
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_zacks_rank(symbol, max_retries=3):
    """Get Zacks rank for a symbol."""
    for _ in range(max_retries):
        try:
            url = f'https://quote-feed.zacks.com/index?t={symbol}'
            with urllib.request.urlopen(url, timeout=10) as response:
                return json.loads(response.read().decode())[symbol]["zacks_rank"]
        except Exception as e:
            logger.error(f"Error fetching Zacks Rank for {symbol}: {e}")
            break
    return None

def find_rank_1_stocks(n=10, max_workers=60):
    """Find n stocks with Zacks rank 1 from SP500."""
    try:
        # Get and shuffle SP500 symbols
        sp500_symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        
        # Get SP400 (Mid Cap) symbols
        sp400_symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol'].tolist()
        
        # Combine and shuffle all symbols
        
        symbols = sp500_symbols + sp400_symbols
        # Get and shuffle SP500 symbols
        #symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        random.shuffle(symbols)
        
        rank_1_stocks = []
        processed = 0
        
        # Process all symbols in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_zacks_rank, symbol): symbol for symbol in symbols}
            
            for future in as_completed(futures):
                if len(rank_1_stocks) >= n:
                    break
                    
                symbol = futures[future]
                try:
                    rank = future.result()
                    processed += 1
                    
                    if rank == '1':
                        rank_1_stocks.append({'symbol': symbol, 'zacks_rank': rank})
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        
        return rank_1_stocks[:n]
        
    except Exception as e:
        logger.error(f"Error fetching SP500 symbols: {e}")
        return []

if __name__ == "__main__":
    stocks = find_rank_1_stocks()
    
    if stocks:
        print("\nFound Zacks rank 1 stocks:")
        for stock in stocks:
            print(f"Symbol: {stock['symbol']}")
        print(f"\nTotal found: {len(stocks)}")
    else:
        print("No rank 1 stocks found.")


