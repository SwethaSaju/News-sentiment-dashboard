# app.py
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from streamlit_autorefresh import st_autorefresh

# ----------------------------
# Auto-refresh every 60 seconds
# ----------------------------
st_autorefresh(interval=60 * 1000, key="news_refresh")

# ----------------------------
# Page Title
# ----------------------------
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard")

# ----------------------------
# API Key
# ----------------------------
API_KEY = st.secrets.get("aa9f72d6bad1c3ffbd42dcba1e7b7ce5", "")  # Use Streamlit secrets

# ----------------------------
# Fetch News
# ----------------------------
def fetch_news():
    try:
        url = f"https://gnews.io/api/v4/top-headlines?token={API_KEY}&lang=en&country=us&max=10"
        response = requests.get(url).json()
        articles = response.get("articles", [])
        data = [{"headline": art["title"], "description": art["description"]} for art in articles]
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("API returned no articles")
        return df
    except:
        # Fallback dummy headlines if API fails
        return pd.DataFrame({
            "headline": [
                "Stock market hits record high",
                "Economic crisis causes panic selling",
                "New vaccine brings hope",
                "Natural disaster destroys thousands of homes"
            ],
            "description": [""]*4
        })

# ----------------------------
# Train Dummy ML Model
# ----------------------------
train_headlines = [
    "Stock market hits record high",
    "Economic crisis causes panic selling",
    "New vaccine brings hope",
    "Natural disaster destroys thousands of homes",
]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_headlines)
model = LogisticRegression()
model.fit(X_train, labels)

# ----------------------------
# Fetch news and classify sentiment
# ----------------------------
df_news = fetch_news()
X_test = vectorizer.transform(df_news["headline"])
preds = model.predict(X_test)
df_news["Sentiment"] = ["Positive" if p==1 else "Negative" for p in preds]

# ----------------------------
# Display Dashboard
# ----------------------------
st.subheader("Latest Headlines with Sentiment")
st.table(df_news[["headline", "Sentiment"]])

st.subheader("Sentiment Counts")
st.bar_chart(df_news["Sentiment"].value_counts())


            

        
