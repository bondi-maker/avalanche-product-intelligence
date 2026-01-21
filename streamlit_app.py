import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Avalanche App", layout="wide")

# --- Snowflake session (native auth inside Snowflake) ---
session = get_active_session()

# --- Load data (your cell2) ---
df = session.sql("""
    SELECT
        ORDER_ID,
        PRODUCT,
        REVIEW_DATE,
        REVIEW_TEXT,
        SNOWFLAKE.CORTEX.SENTIMENT(REVIEW_TEXT) AS SENTIMENT_SCORE,
        SHIPPING_DATE,
        CARRIER,
        STATUS,
        DELIVERY_DAYS,
        LATE,
        REGION
    FROM CLEANED_REVIEWS
    WHERE REVIEW_TEXT IS NOT NULL
      AND TRIM(REVIEW_TEXT) <> ''
""").to_pandas()

st.title("Avalanche Customer Sentiment Dashboard")

# --- Sidebar filters (your cell3) ---
st.sidebar.header("Filters")

products = st.sidebar.multiselect(
    "Product",
    options=sorted(df["PRODUCT"].dropna().unique()),
    default=sorted(df["PRODUCT"].dropna().unique())
)

late_only = st.sidebar.selectbox("Delivery Status", ["ALL", "On Time", "Late"])

filtered_df = df[df["PRODUCT"].isin(products)]

if late_only == "Late":
    filtered_df = filtered_df[filtered_df["LATE"] == True]
elif late_only == "On Time":
    filtered_df = filtered_df[filtered_df["LATE"] == False]

# --- Data preview ---
st.subheader("Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)

# --- Charts (your cell4) ---
st.subheader("Mean Sentiment by Product")
by_product = (
    filtered_df.groupby("PRODUCT")["SENTIMENT_SCORE"]
      .mean()
      .sort_values(ascending=False)
)
st.bar_chart(by_product)

st.subheader("Average Sentiment: Late vs On-Time")
by_late = filtered_df.groupby("LATE")["SENTIMENT_SCORE"].mean()
st.bar_chart(by_late)

# --- Quick stats (your cell5) ---
st.subheader("Quick Stats")
c1, c2, c3 = st.columns(3)
c1.metric("Total Reviews", len(filtered_df))
c2.metric("Average Sentiment", round(filtered_df["SENTIMENT_SCORE"].mean(), 3))
c3.metric("Late %", round(filtered_df["LATE"].mean() * 100, 1))
