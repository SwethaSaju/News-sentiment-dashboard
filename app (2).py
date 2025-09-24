# app.py
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# 1. Fetch News
# ----------------------------
API_KEY = "aa9f72d6bad1c3ffbd42dcba1e7b7ce5"

def fetch_news():
    url = f"https://gnews.io/api/v4/top-headlines?token={API_KEY}&lang=en&country=us&max=10"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    data = [{"headline": art["title"], "description": art["description"]} for art in articles]
    return pd.DataFrame(data)

# ----------------------------
# 2. Train Dummy ML Model
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
# 3. Streamlit Dashboard
# ----------------------------
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard")
placeholder = st.empty()

while True:
    df_news = fetch_news()
    if not df_news.empty:
        X_test = vectorizer.transform(df_news["headline"])
        preds = model.predict(X_test)
        df_news["Sentiment"] = ["Positive" if x==1 else "Negative" for x in preds]

        with placeholder.container():
            st.subheader("Latest Headlines with Sentiment")
            st.table(df_news[["headline", "Sentiment"]])
            st.bar_chart(df_news["Sentiment"].value_counts())
    else:
        st.warning("No news fetched.")
    st.experimental_rerun()  # auto-refresh dashboard

            

        
