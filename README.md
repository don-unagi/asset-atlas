# AssetAtlas - AI Financial Advisor

AssetAtlas is an AI-powered financial advisor that provides personalized investment recommendations based on your portfolio, risk tolerance, and investment goals. It uses a multi-agent system with RAG (Retrieval Augmented Generation) to analyze your portfolio from multiple perspectives.

## Features

- **Portfolio Analysis**: Analyzes your current portfolio composition, diversification, and performance
- **Technical Analysis**: Evaluates technical indicators like RSI, MACD, and moving averages
- **News Analysis**: Gathers and analyzes recent news about your portfolio stocks
- **Fundamental Analysis**: Provides insights based on Graham and Dodd's "Security Analysis" book
- **Personalized Recommendations**: Generates tailored investment recommendations based on all analyses

## Multi-Agent System

The app uses a sophisticated multi-agent system powered by LangGraph:

1. **Supervisor Agent**: Manages the workflow and decides which agent should act next
2. **Portfolio Analyzer**: Analyzes the basic portfolio structure and allocation
3. **Technical Analyzer**: Performs technical analysis on stocks in the portfolio
4. **News Analyzer**: Gathers and analyzes recent news about portfolio stocks
5. **RAG Analyzer**: Provides fundamental analysis insights based on Graham and Dodd's "Security Analysis"
6. **Recommendation Engine**: Generates final recommendations based on all analyses

## Setup and Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   NEWSAPI_KEY=your_newsapi_key
   ```
6. Load the Security Analysis book into the vector database:
   ```
   python load_security_analysis.py
   ```
7. Run the app:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter your risk tolerance and investment goals in the sidebar
2. Add your portfolio stocks, shares, and purchase prices
3. Click "Generate Recommendations" to get personalized financial advice
4. View the analysis results, including technical indicators, news sentiment, and recommendations

## Dependencies

- Streamlit: Web interface
- LangChain: LLM orchestration
- LangGraph: Multi-agent system
- OpenAI: LLM provider
- HuggingFace: Embeddings model
- Chroma: Vector database
- yfinance: Stock data
- NewsAPI: Financial news

## License

MIT 