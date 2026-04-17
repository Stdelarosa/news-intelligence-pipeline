import os
import requests
import pandas as pd
import spacy
from textblob import TextBlob
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
DB_URL = os.environ.get("DB_URL")

engine = create_engine(DB_URL)
nlp = spacy.load("en_core_web_lg")

def create_table():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id              SERIAL PRIMARY KEY,
                fetched_at      TIMESTAMP DEFAULT NOW(),
                published_at    TIMESTAMP,
                source          VARCHAR(100),
                title           TEXT,
                description     TEXT,
                url             TEXT UNIQUE,
                sentiment_score NUMERIC(5,4),
                sentiment_label VARCHAR(10),
                entities        TEXT
            )
        """))
        conn.commit()
    print("Table ready")

def fetch_news():
    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"category=technology&"
        f"language=en&"
        f"country=us&"
        f"pageSize=50&"
        f"apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if data["status"] != "ok":
        print(f"API error: {data}")
        return []

    articles = data.get("articles", [])
    print(f"Fetched {len(articles)} articles")
    return articles

def analyze_article(article):
    title = article.get("title") or ""
    description = article.get("description") or ""
    text = f"{title}. {description}"

    blob = TextBlob(text)
    score = round(blob.sentiment.polarity, 4)
    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    doc = nlp(text)
    entities = list(set([
        ent.text for ent in doc.ents
        if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT"]
    ]))
    entities_str = ", ".join(entities[:10])

    return score, label, entities_str

def load_articles(articles):
    loaded = 0
    skipped = 0

    for article in articles:
        try:
            score, label, entities = analyze_article(article)

            published = article.get("publishedAt")
            if published:
                published = datetime.strptime(
                    published, "%Y-%m-%dT%H:%M:%SZ"
                )

            row = pd.DataFrame([{
                "published_at":    published,
                "source":          article.get("source", {}).get("name"),
                "title":           article.get("title"),
                "description":     article.get("description"),
                "url":             article.get("url"),
                "sentiment_score": score,
                "sentiment_label": label,
                "entities":        entities
            }])

            row.to_sql(
                name="news_articles",
                con=engine,
                if_exists="append",
                index=False
            )
            loaded += 1

        except Exception as e:
            if "unique" in str(e).lower():
                skipped += 1
            else:
                print(f"Error loading article: {e}")

    print(f"Loaded {loaded} new articles, skipped {skipped} duplicates")

if __name__ == "__main__":
    create_table()
    articles = fetch_news()
    if articles:
        load_articles(articles)
    print("Pipeline complete")