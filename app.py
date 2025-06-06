import streamlit as st
import pandas as pd
from modules import sentiment, technicals, fundamentals, experts
import requests # Import requests for Together AI API calls
# import openai # Import the openai library
# import anthropic # Uncomment and import other libraries as needed
# from langchain.llms import YourOtherLLM # Example for LangChain integration

# --- API Keys (Replace with your actual keys) ---
NEWS_API_KEY = "d31aaa0ad9384840942091f6e040532a"
TOGETHER_AI_API_KEY = "tgp_v1_lJnmHnecCkQPf0QbkzDSa5WVzohj6EX5WFAoqXJf9NU"
# -----------------------------------------------

st.set_page_config(page_title="Stockezy Dashboard", layout="wide")
st.title("📈 Stockezy: AI Stock Research Dashboard")
st.markdown("""
A real-time AI research agent for stock market analysis. Enter a stock symbol in the sidebar to get started!
""")

# Sidebar for user input
st.sidebar.header("Configuration")
# Removed API key input fields
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL)")
run_analysis = st.sidebar.button("Analyze")

# Function to call Together AI LLM
def call_together_ai_llm(prompt, api_key, model="mistralai/Mixtral-8x7B-Instruct-v0.1"): # Defaulting to a common Together AI model
    if not api_key or api_key == "YOUR_TOGETHER_AI_API_KEY":
        return "Together AI API key not provided or is still the placeholder."

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error calling Together AI LLM: {e}"


# Tabs for each module
# Check only for symbol and trigger button click now that API keys are embedded
if symbol and run_analysis:
    # Check if API keys are provided (even though embedded, they might be placeholders)
    if NEWS_API_KEY == "YOUR_NEWS_API_KEY":
         st.error("Please replace 'YOUR_NEWS_API_KEY' with your actual NewsAPI key in app.py")
    if TOGETHER_AI_API_KEY == "YOUR_TOGETHER_AI_API_KEY":
         st.error("Please replace 'YOUR_TOGETHER_AI_API_KEY' with your actual Together AI API key in app.py")

    # Proceed only if keys are updated
    if NEWS_API_KEY != "YOUR_NEWS_API_KEY" and TOGETHER_AI_API_KEY != "YOUR_TOGETHER_AI_API_KEY":

        tab1, tab2, tab3, tab4 = st.tabs([
            "Sentiment Analysis", "Technical Indicators", "Fundamental Analysis", "Expert Perspectives"
        ])

        with tab1:
            st.subheader("📰 Sentiment Analysis (FinBERT)")
            with st.spinner("Fetching news and analyzing sentiment..."):
                headlines = sentiment.fetch_news_headlines(symbol, NEWS_API_KEY)
                if headlines:
                    results = sentiment.analyze_sentiment(headlines)
                    for headline, label, prob in results:
                        color = "green" if label == "positive" else ("red" if label == "negative" else "gray")
                        st.markdown(f"<span style='color:{color}'><b>{label.upper()}</b></span>: {headline} <small>({prob:.2f})</small>", unsafe_allow_html=True)
                else:
                    st.info(f"No recent news found for {symbol}")

        with tab2:
            st.subheader("📊 Technical Indicators (RSI, MACD, SMA)")
            with st.spinner("Fetching stock data and calculating indicators..."):
                df = technicals.fetch_stock_data(symbol)
                if not df.empty:
                    sma = technicals.calculate_sma(df)
                    rsi = technicals.calculate_rsi(df)
                    macd_line, macd_signal = technicals.calculate_macd(df)

                    # Plots
                    st.line_chart(df['Close'], height=200, use_container_width=True)
                    # Create a combined DataFrame for SMA and RSI for plotting
                    sma_rsi_df = pd.DataFrame({'SMA': sma, 'RSI': rsi})
                    st.line_chart(sma_rsi_df, height=200, use_container_width=True)
                    # Create a combined DataFrame for MACD lines for plotting
                    macd_df = pd.DataFrame({'MACD': macd_line, 'Signal': macd_signal})
                    st.line_chart(macd_df, height=200, use_container_width=True)
                else:
                    st.info(f"Could not fetch stock data for {symbol}.")

        with tab3:
            st.subheader("💼 Fundamental Analysis")
            with st.spinner("Fetching fundamentals and generating insights..."):
                fund = fundamentals.fetch_fundamentals(symbol)
                if fund:
                    st.json(fund)
                    st.markdown("**LLM-Driven Insight:**")
                    # Call the Together AI LLM function
                    llm_insight = fundamentals.generate_llm_insight(fund, lambda p: call_together_ai_llm(p, TOGETHER_AI_API_KEY))
                    st.write(llm_insight)
                else:
                     st.info(f"Could not fetch fundamental data for {symbol}.")

        with tab4:
            st.subheader("🧑‍💼 Expert Perspectives")
            expert = st.selectbox("Choose an expert template:", ["buffett", "lynch", "dalio"])
            st.markdown(f"**{expert.title()}-Style Analysis**")

            with st.spinner(f"Generating {expert.title()}-style analysis..."):
                fund = fundamentals.fetch_fundamentals(symbol)
                if fund:
                    # Call the Together AI LLM function with selected expert template
                    expert_analysis = experts.generate_expert_analysis(fund, lambda p: call_together_ai_llm(p, TOGETHER_AI_API_KEY), expert=expert)
                    st.write(expert_analysis)
                else:
                    st.info(f"Could not fetch fundamental data for {symbol}.")


    else:
         st.warning("Please update API keys in app.py to run the analysis.")

else:
    st.info("Enter a stock symbol and click 'Analyze' in the sidebar to begin.") 