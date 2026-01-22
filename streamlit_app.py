import streamlit as st
import pandas as pd

st.set_page_config(page_title="Avalanche App", layout="wide")

# --- Snowflake connection via Streamlit Secrets (Community Cloud) ---
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    conn = st.connection("snowflake")

    # 🔎 TEMP DEBUG — confirms Snowflake identity + context
    debug_df = conn.query("""
        SELECT
            CURRENT_ACCOUNT()  AS account,
            CURRENT_USER()     AS user,
            CURRENT_ROLE()     AS role,
            CURRENT_WAREHOUSE() AS warehouse,
            CURRENT_DATABASE() AS db,
            CURRENT_SCHEMA()   AS schema
    """)
    st.subheader("Snowflake Session Context (debug)")
    st.dataframe(debug_df, use_container_width=True)

    # --- Main query (fully qualified, no context ambiguity) ---
    query = """
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
        FROM AVALANCHE_DB.AVALANCHE_SCHEMA.CLEANED_REVIEWS
        WHERE REVIEW_TEXT IS NOT NULL
          AND TRIM(REVIEW_TEXT) <> ''
    """

    return conn.query(query)


# --- Load data ---
df = load_data()

st.title("Avalanche Customer Sentiment Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")

products_all = sorted(df["PRODUCT"].dropna().unique().tolist())
products = st.sidebar.multiselect(
    "Product",
    options=products_all,
    default=products_all
)

late_only = st.sidebar.selectbox(
    "Delivery Status",
    ["ALL", "On Time", "Late"]
)

filtered_df = df[df["PRODUCT"].isin(products)].copy()

if late_only == "Late":
    filtered_df = filtered_df[filtered_df["LATE"] == True]
elif late_only == "On Time":
    filtered_df = filtered_df[filtered_df["LATE"] == False]

# --- Data preview ---
st.subheader("Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)

# --- Charts ---
st.subheader("Mean Sentiment by Product")
by_product = (
    filtered_df
    .groupby("PRODUCT")["SENTIMENT_SCORE"]
    .mean()
    .sort_values(ascending=False)
)
st.bar_chart(by_product)

st.subheader("Average Sentiment: Late vs On-Time")
by_late = filtered_df.groupby("LATE")["SENTIMENT_SCORE"].mean()
st.bar_chart(by_late)

# --- Quick stats ---
st.subheader("Quick Stats")
c1, c2, c3 = st.columns(3)

c1.metric("Total Reviews", int(len(filtered_df)))

avg_sent = float(filtered_df["SENTIMENT_SCORE"].mean()) if len(filtered_df) else 0.0
c2.metric("Average Sentiment", round(avg_sent, 3))

late_pct = float(filtered_df["LATE"].mean()) * 100 if len(filtered_df) else 0.0
c3.metric("Late %", round(late_pct, 1))
