import os
import datetime
from newsapi import NewsApiClient
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from .state import AgentState

def fetch_news(tickers, days=7):
    """Fetch news for the given tickers from NewsAPI."""
    # You need to set NEWSAPI_KEY in your .env file
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if not newsapi_key:
        print("NewsAPI key not found. Please add NEWSAPI_KEY to your .env file.")
        return []
    
    newsapi = NewsApiClient(api_key=newsapi_key)
    news_items = []
    
    # Get current date and date from n days ago
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    days_ago = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    
    for ticker in tickers:
        try:
            # Search for news about the ticker
            all_articles = newsapi.get_everything(
                q=ticker,
                from_param=days_ago,
                to=today,
                language='en',
                sort_by='relevancy',
                page_size=3  # Get top 3 news items per ticker
            )
            
            # Process the articles
            if all_articles['status'] == 'ok' and all_articles['totalResults'] > 0:
                for article in all_articles['articles']:
                    news_items.append({
                        'ticker': ticker,
                        'title': article['title'],
                        'summary': article['description'] or "No description available.",
                        'url': article['url'],
                        'published_at': article['publishedAt'],
                        'content': article.get('content', '')
                    })
            
            # If no news found, add a placeholder
            if not all_articles['articles']:
                news_items.append({
                    'ticker': ticker,
                    'title': f"No recent news found for {ticker}",
                    'summary': "No relevant news articles were found for this ticker in the past week.",
                    'url': "",
                    'published_at': today,
                    'content': ""
                })
                
        except Exception as e:
            # Handle API errors or other exceptions
            print(f"Error fetching news for {ticker}: {str(e)}")
            news_items.append({
                'ticker': ticker,
                'title': f"Error fetching news for {ticker}",
                'summary': f"Could not retrieve news due to an error: {str(e)}",
                'url': "",
                'published_at': today,
                'content': ""
            })
    
    return news_items

def create_news_analyzer_chain():
    """Create a chain for news analysis using LLM."""
    # Define function for structured output
    function_def = {
        "name": "analyze_news",
        "description": "Analyze financial news articles for sentiment and impact",
        "parameters": {
            "type": "object",
            "properties": {
                "articles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            },
                            "title": {
                                "type": "string",
                                "description": "News article title"
                            },
                            "summary": {
                                "type": "string",
                                "description": "Summary of the news article"
                            },
                            "url": {
                                "type": "string",
                                "description": "URL of the news article"
                            },
                            "published_at": {
                                "type": "string",
                                "description": "Publication date of the article"
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                                "description": "Sentiment of the article towards the stock"
                            },
                            "impact_analysis": {
                                "type": "string",
                                "description": "Analysis of how this news might impact the stock"
                            },
                            "key_points": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Key points extracted from the news article"
                            },
                            "market_implications": {
                                "type": "string",
                                "description": "Broader market implications of this news"
                            }
                        },
                        "required": ["ticker", "title", "summary", "sentiment", "impact_analysis", "url"]
                    }
                },
                "overall_market_sentiment": {
                    "type": "string",
                    "description": "Overall market sentiment based on all news articles"
                },
                "key_trends": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Key trends identified across all news articles"
                }
            },
            "required": ["articles", "overall_market_sentiment"]
        }
    }
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a financial news analyst. Analyze the following news articles about stocks.
    
    For each article, determine the sentiment (positive, negative, or neutral) and provide a brief analysis of how this news might impact the stock.
    Focus on the news content itself and its potential market implications.
    
    IMPORTANT: Only include articles that are directly relevant to the stocks in the portfolio or to financial markets in general.
    Exclude any articles that are not related to financial markets, stocks, or investing.
    
    NEWS ARTICLES:
    {news_articles}
    
    PORTFOLIO CONTEXT:
    {portfolio_context}
    
    Provide a detailed sentiment analysis for each article and an overall market sentiment assessment.
    Focus on the news content and market implications rather than current portfolio performance.
    """)

    #CRITICAL: For each article, you MUST include the URL exactly as provided in the input. Do not modify or omit the URL.
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    
    # Create chain
    chain = (
        prompt 
        | llm.bind_functions(functions=[function_def], function_call="analyze_news")
        | JsonOutputFunctionsParser()
    )
    
    return chain

# Initialize news analyzer chain
news_analyzer_chain = create_news_analyzer_chain()

def news_analyzer(state: AgentState) -> AgentState:
    """Gathers and analyzes news for relevant assets using NewsAPI and LLM."""
    portfolio = state["portfolio_data"]
    tickers = list(portfolio.keys())
    
    try:
        # Fetch news
        news_items = fetch_news(tickers)
        
        if not news_items:
            # If no news found, return empty analysis
            state["news_analysis"] = []
            state["messages"] = state.get("messages", []) + [{
                "role": "ai",
                "content": "[NewsAnalyzer] I couldn't find any relevant news for your portfolio stocks."
            }]
            return state
        
        # Format portfolio context - simplified to focus on tickers only
        portfolio_context = "Portfolio contains the following tickers:\n"
        for ticker in tickers:
            portfolio_context += f"{ticker}\n"
        
        # Format news articles for LLM
        news_articles_str = ""
        for item in news_items:
            news_articles_str += f"TICKER: {item['ticker']}\n"
            news_articles_str += f"TITLE: {item['title']}\n"
            news_articles_str += f"SUMMARY: {item['summary']}\n"
            news_articles_str += f"URL: {item['url']}\n"
            news_articles_str += f"PUBLISHED: {item['published_at']}\n"
            if item['content']:
                news_articles_str += f"CONTENT: {item['content']}\n"
            news_articles_str += "\n---\n\n"
        
        # Analyze news with LLM
        try:
            result = news_analyzer_chain.invoke({
                "news_articles": news_articles_str,
                "portfolio_context": portfolio_context
            })
            
            # Update state with analyzed news
            state["news_analysis"] = result["articles"]
            
            # Add message to communication
            state["messages"] = state.get("messages", []) + [{
                "role": "ai",
                "content": f"[NewsAnalyzer] I've analyzed recent news for your portfolio stocks. Overall market sentiment: {result['overall_market_sentiment']}"
            }]
        except Exception as e:
            print(f"Error analyzing news with LLM: {str(e)}")
            # Create a default analysis if LLM fails
            default_analysis = []
            for item in news_items[:5]:  # Process up to 5 news items
                default_analysis.append({
                    "ticker": item["ticker"],
                    "title": item["title"],
                    "summary": item["summary"],
                    "url": item["url"],
                    "published_at": item.get("published_at", ""),
                    "sentiment": "neutral",
                    "impact_analysis": "Could not analyze impact due to processing error.",
                    "key_points": ["News content could not be analyzed"],
                    "market_implications": "Unknown due to processing limitations"
                })
            
            state["news_analysis"] = default_analysis
            state["messages"] = state.get("messages", []) + [{
                "role": "ai",
                "content": "[NewsAnalyzer] I found some news for your portfolio stocks, but couldn't perform detailed analysis."
            }]
    except Exception as e:
        print(f"Error in news analyzer: {str(e)}")
        state["news_analysis"] = []
        state["messages"] = state.get("messages", []) + [{
            "role": "ai",
            "content": f"[NewsAnalyzer] I encountered an error while analyzing news: {str(e)}"
        }]
    
    return state
