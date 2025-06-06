import yfinance as yf

def fetch_fundamentals(symbol):
    """
    Fetch fundamental data for a stock using yfinance.
    Returns a dictionary with key financial metrics.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    fundamentals = {
        'symbol': symbol,
        'longName': info.get('longName'),
        'sector': info.get('sector'),
        'marketCap': info.get('marketCap'),
        'trailingPE': info.get('trailingPE'),
        'forwardPE': info.get('forwardPE'),
        'priceToBook': info.get('priceToBook'),
        'dividendYield': info.get('dividendYield'),
        'returnOnEquity': info.get('returnOnEquity'),
        'debtToEquity': info.get('debtToEquity'),
        'revenueGrowth': info.get('revenueGrowth'),
        'grossMargins': info.get('grossMargins'),
        'operatingMargins': info.get('operatingMargins'),
        'profitMargins': info.get('profitMargins'),
    }
    return fundamentals


def generate_llm_insight(fundamentals, llm):
    """
    Generate LLM-driven insights from fundamental data.
    llm: a callable LLM function (e.g., OpenAI, Claude, etc.)
    """
    prompt = f"""
    Analyze the following stock fundamentals and provide a summary for a retail investor:
    {fundamentals}
    """
    return llm(prompt) 