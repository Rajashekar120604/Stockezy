# Stockezy

Stockezy is a real-time AI research agent for stock market analysis. It combines sentiment analysis, technical indicators, fundamental analysis, and expert perspectives into an interactive Streamlit dashboard.

## Features
- Sentiment Analysis (FinBERT for news & social media)
- Technical Indicators (RSI, MACD, SMA)
- Fundamental Analysis (Yahoo Finance API + LLM-driven insights)
- Expert Perspectives (Buffett, Lynch, Dalio-style analysis)
- Interactive UI (Streamlit-powered dashboards)

## Tech Stack
- Python, Streamlit, yfinance, pandas, numpy, transformers, langchain, requests, ta, pinecone-client, openai

## Setup
1. Clone the repository or copy the folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Folder Structure
```
Stockezy/
├── app.py
├── requirements.txt
├── modules/
│   ├── sentiment.py
│   ├── technicals.py
│   ├── fundamentals.py
│   ├── experts.py
├── utils/
│   └── data_fetch.py
└── README.md
``` 