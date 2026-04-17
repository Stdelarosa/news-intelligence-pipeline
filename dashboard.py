import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

DB_URL = os.environ.get("DB_URL")
engine = create_engine(DB_URL)

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_sql("""
        SELECT * FROM news_articles
        ORDER BY published_at DESC
    """, engine)

df = load_data()

st.title("AI & Tech News Intelligence")
st.caption("Sentiment analysis and entity extraction on daily news headlines")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Articles", len(df))
col2.metric("Positive", len(df[df["sentiment_label"] == "positive"]))
col3.metric("Neutral", len(df[df["sentiment_label"] == "neutral"]))
col4.metric("Negative", len(df[df["sentiment_label"] == "negative"]))

st.divider()

st.subheader("Sentiment Over Time")
df["published_at"] = pd.to_datetime(df["published_at"])
df["date"] = df["published_at"].dt.date

sentiment_by_date = df.groupby(["date", "sentiment_label"]).size().unstack(
    fill_value=0
)
st.bar_chart(sentiment_by_date)

st.divider()

st.subheader("Most Mentioned Entities")
all_entities = []
for row in df["entities"].dropna():
    all_entities.extend([e.strip() for e in row.split(",") if e.strip()])

entity_counts = pd.Series(all_entities).value_counts().head(15).reset_index()
entity_counts.columns = ["Entity", "Mentions"]
st.bar_chart(entity_counts.set_index("Entity"))

st.divider()

st.subheader("Articles by Source")
source_counts = df["source"].value_counts().head(10).reset_index()
source_counts.columns = ["Source", "Articles"]
st.bar_chart(source_counts.set_index("Source"))

st.divider()

st.subheader("Latest Headlines")

sentiment_filter = st.selectbox(
    "Filter by Sentiment",
    ["All", "positive", "neutral", "negative"]
)

filtered = df.copy()
if sentiment_filter != "All":
    filtered = filtered[filtered["sentiment_label"] == sentiment_filter]

st.dataframe(
    filtered[[
        "published_at", "source", "title",
        "sentiment_label", "sentiment_score", "entities"
    ]].head(50),
    use_container_width=True,
    height=500
)