import os
import pandas as pd
import requests
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# ---------------------------
# Config & secrets
# ---------------------------
# Prefer Streamlit Secrets > env var > text input fallback
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", None))

st.set_page_config(page_title="Real-Time News Sentiment", page_icon="📰", layout="wide")
st.title("📰 Real-Time News Sentiment (PySpark + Streamlit)")

with st.sidebar:
    st.header("Configuration")
    region = st.text_input("Top-headlines country (2-letter, optional)", value="")
    pagesize = st.slider("Page size", min_value=10, max_value=100, value=30, step=5)
    if not NEWSAPI_KEY:
        NEWSAPI_KEY = st.text_input("NewsAPI key", type="password", help="Set in Streamlit Secrets as NEWSAPI_KEY for deployments")

    st.caption("Tip: On Streamlit Cloud, add NEWSAPI_KEY in app secrets.")

# ---------------------------
# Helpers
# ---------------------------
def fetch_latest_news(apikey: str, pagesize: int = 30, country: str = "") -> pd.DataFrame:
    base = "https://newsapi.org/v2/top-headlines"
    params = {
        "language": "en",
        "pageSize": pagesize,
        "apiKey": apikey,
    }
    if country.strip():
        params["country"] = country.strip().lower()

    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("articles", []) or []
    rows = []
    for a in data:
        title = a.get("title")
        desc = a.get("description")
        if title:
            rows.append((title, desc))
    return pd.DataFrame(rows, columns=["headline", "description"])


def build_pipeline():
    tokenizer = Tokenizer(inputCol="headline", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    return Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])


def get_example_training_df(spark):
    # Minimal illustrative training data from the notebook
    data = [
        ("Stocks are soaring today after positive market news", 1),
        ("Company profits hit record highs", 1),
        ("Global markets are crashing amid inflation fears", 0),
        ("Investors are worried about economic slowdown", 0),
        ("Tech sector reports major growth", 1),
        ("Unemployment rates rise unexpectedly", 0),
    ]
    return spark.createDataFrame(data, ["headline", "label"])

# ---------------------------
# Main flow
# ---------------------------
if not NEWSAPI_KEY:
    st.warning("Provide a NewsAPI key to fetch headlines.")
    st.stop()

# Initialize Spark lazily and cache via session state
if "spark" not in st.session_state:
    st.session_state.spark = SparkSession.builder.appName("RealTimeNewsSentiment").getOrCreate()

spark = st.session_state.spark

# Train or load model in session
if "model" not in st.session_state:
    with st.spinner("Training model pipeline..."):
        pipeline = build_pipeline()
        train_df = get_example_training_df(spark)
        st.session_state.model = pipeline.fit(train_df)

model = st.session_state.model

col_left, col_right = st.columns([1, 1])
with col_left:
    if st.button("Fetch latest headlines"):
        st.session_state.news_df = None  # reset

# Fetch data
if "news_df" not in st.session_state or st.session_state.news_df is None:
    try:
        news_pdf = fetch_latest_news(NEWSAPI_KEY, pagesize=pagesize, country=region)
        if news_pdf.empty:
            st.info("No headlines returned. Try another country or larger page size.")
            st.stop()
        st.session_state.news_df = news_pdf
    except Exception as e:
        st.error(f"Failed to fetch headlines: {e}")
        st.stop()

news_df = st.session_state.news_df

# Convert to Spark, predict, return to pandas
spark_df = spark.createDataFrame(news_df)
predictions = model.transform(spark_df).select("headline", "prediction")

preds_pdf = predictions.toPandas()
preds_pdf["Sentiment"] = preds_pdf["prediction"].map({1.0: "Positive", 0.0: "Negative"})

# ---------------------------
# Display
# ---------------------------
with col_right:
    pos_rate = (preds_pdf["prediction"] == 1.0).mean()
    st.metric("Positive share", f"{pos_rate*100:.1f}%")

st.subheader("Predicted sentiments")
st.dataframe(preds_pdf[["headline", "Sentiment"]], use_container_width=True)

with st.expander("Show raw predictions"):
    st.dataframe(preds_pdf, use_container_width=True)

st.caption("Note: This demo uses a tiny illustrative training set; replace with a real, labeled dataset for production-quality results.")
