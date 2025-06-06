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
    """
    url = (
        f'https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&language=en&pageSize={num_headlines}&apiKey={api_key}'
    )
    response = requests.get(url)
    data = response.json()
    headlines = [article['title'] for article in data.get('articles', [])]
    return headlines


def analyze_sentiment(headlines):
    """
    Analyze sentiment of a list of headlines using FinBERT.
    Returns a list of (headline, sentiment) tuples.
    """
    results = []
    for headline in headlines:
        inputs = finbert_tokenizer(headline, return_tensors='pt', truncation=True, max_length=64)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
            sentiment = np.argmax(probs)
            if sentiment == 0:
                label = 'negative'
            elif sentiment == 1:
                label = 'neutral'
            else:
                label = 'positive'
        results.append((headline, label, float(np.max(probs))))
    return results 