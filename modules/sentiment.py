import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load FinBERT model and tokenizer
finbert_model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')

def fetch_news_headlines(symbol, api_key, num_headlines=10):
    """
    Fetch recent news headlines for a given stock symbol using NewsAPI.
    Improved query and filtering.
    """
    # Smarter query → look for stock-related headlines
    query = f'"{symbol} stock" OR "{symbol} shares" OR "{symbol} price" OR "{symbol} earnings" OR "NASDAQ:{symbol}"'

    url = (
        f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize={num_headlines}&apiKey={api_key}'
    )
    response = requests.get(url)
    data = response.json()

    # Extract clean headlines
    headlines = []
    for article in data.get('articles', []):
        headline = article['title']
        # Filter out very short or noisy headlines
        if len(headline.split()) >= 4:
            headlines.append(headline)

    return headlines

def analyze_sentiment(headlines, confidence_threshold=0.7):
    """
    Analyze sentiment of a list of headlines using FinBERT.
    Returns a list of (headline, sentiment, confidence) tuples.
    Applies optional confidence threshold.
    """
    results = []
    for headline in headlines:
        inputs = finbert_tokenizer(headline, return_tensors='pt', truncation=True, max_length=64)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
            sentiment_idx = np.argmax(probs)
            confidence = float(np.max(probs))

            # Map index to label
            if sentiment_idx == 0:
                label = 'NEGATIVE'
            elif sentiment_idx == 1:
                label = 'NEUTRAL'
            else:
                label = 'POSITIVE'

            # Apply confidence threshold
            if confidence >= confidence_threshold:
                results.append((headline, label, confidence))
            else:
                results.append((headline, 'UNCERTAIN', confidence))

    return results
