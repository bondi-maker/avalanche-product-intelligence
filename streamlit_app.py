import streamlit as st
import pandas as pd

st.set_page_config(page_title="Avalanche Customer Sentiment", layout="wide")

# -----------------------------
# Data load
# -----------------------------
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    conn = st.connection("snowflake")

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
    df = conn.query(query)

    # Normalize types for Streamlit filters/plots
    if "REVIEW_DATE" in df.columns:
        df["REVIEW_DATE"] = pd.to_datetime(df["REVIEW_DATE"], errors="coerce")

    if "SHIPPING_DATE" in df.columns:
        df["SHIPPING_DATE"] = pd.to_datetime(df["SHIPPING_DATE"], errors="coerce")

    # Snowflake can return booleans or 0/1 depending on source—normalize
    if "LATE" in df.columns:
        if df["LATE"].dtype != bool:
            df["LATE"] = df["LATE"].astype(str).str.lower().isin(["true", "1", "t", "yes"])

    return df


# -----------------------------
# LLM helper (prompt augmentation)
# -----------------------------
@st.cache_data(ttl=300)
def cortex_complete(prompt: str, model: str = "snowflake-arctic") -> str:
    """
    Calls Snowflake Cortex COMPLETE via SQL.
    Works in Snowflake-hosted Streamlit and can work in Streamlit Cloud
    if the Snowflake user/role has Cortex permissions.
    """
    conn = st.connection("snowflake")

    # Escape single quotes for SQL string literal safety
    safe_prompt = prompt.replace("'", "''")

    sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{safe_prompt}'
        ) AS RESPONSE
    """
    res = conn.query(sql)
    return str(res.iloc[0]["RESPONSE"])


def build_dataset_brief(df: pd.DataFrame) -> str:
    """Small, cheap 'cheat sheet' describing what the dataset contains."""
    if df is None or df.empty:
        return "Dataset is empty."

    cols = list(df.columns)

    # Date range
    dr_min, dr_max = None, None
    if "REVIEW_DATE" in df.columns:
        non_null = df["REVIEW_DATE"].dropna()
        if len(non_null):
            dr_min = non_null.min().date()
            dr_max = non_null.max().date()

    # Product list (top 25 by count)
    top_products = (
        df["PRODUCT"].dropna().value_counts().head(25).index.tolist()
        if "PRODUCT" in df.columns else []
    )

    lines = []
    lines.append("DATASET BRIEF (Avalanche reviews)")
    lines.append(f"- Rows available: {len(df)}")
    if dr_min and dr_max:
        lines.append(f"- Review date range: {dr_min} to {dr_max}")
    lines.append(f"- Columns: {', '.join(cols)}")
    if top_products:
        lines.append(f"- Top products (by review count, max 25): {', '.join(top_products)}")

    # Definitions (keep short)
    defs = []
    if "SENTIMENT_SCORE" in df.columns:
        defs.append("SENTIMENT_SCORE: numeric sentiment from Snowflake Cortex SENTIMENT() (higher = more positive).")
    if "LATE" in df.columns:
        defs.append("LATE: True means late delivery, False means on-time delivery.")
    if "DELIVERY_DAYS" in df.columns:
        defs.append("DELIVERY_DAYS: number of days between shipping and delivery (if provided).")
    if defs:
        lines.append("- Definitions:")
        for d in defs:
            lines.append(f"  - {d}")

    return "\n".join(lines)


def fetch_review_snippets(
    product: str | None,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
    max_rows: int = 12
) -> str:
    """
    Lightweight retrieval (not full RAG):
    pulls a small set of recent reviews for the selected product/date window to ground answers.
    """
    conn = st.connection("snowflake")

    where = ["REVIEW_TEXT IS NOT NULL", "TRIM(REVIEW_TEXT) <> ''"]

    if product and product != "ALL":
        safe_product = product.replace("'", "''")
        where.append(f"PRODUCT = '{safe_product}'")

    if start_dt is not None:
        where.append(f"REVIEW_DATE >= '{start_dt.date()}'")

    if end_dt is not None:
        where.append(f"REVIEW_DATE <= '{end_dt.date()}'")

    where_clause = " AND ".join(where)

    sql = f"""
        SELECT
            ORDER_ID,
            PRODUCT,
            REVIEW_DATE,
            LEFT(REVIEW_TEXT, 300) AS REVIEW_SNIPPET,
            SNOWFLAKE.CORTEX.SENTIMENT(REVIEW_TEXT) AS SENTIMENT_SCORE,
            LATE
        FROM AVALANCHE_DB.AVALANCHE_SCHEMA.CLEANED_REVIEWS
        WHERE {where_clause}
        ORDER BY REVIEW_DATE DESC
        LIMIT {int(max_rows)}
    """

    snippet_df = conn.query(sql)

    if snippet_df.empty:
        return "No review snippets found for the current selection."

    lines = ["REVIEW SNIPPETS (most recent, truncated):"]
    for _, r in snippet_df.iterrows():
        rd = r.get("REVIEW_DATE")
        rd_str = str(rd)[:10] if rd is not None else "N/A"
        lines.append(
            f"- {rd_str} | ORDER_ID={r.get('ORDER_ID')} | "
            f"PRODUCT={r.get('PRODUCT')} | LATE={r.get('LATE')} | "
            f"SENTIMENT={r.get('SENTIMENT_SCORE')} | "
            f"TEXT=\"{str(r.get('REVIEW_SNIPPET', '')).replace(chr(10), ' ')}\""
        )

    return "\n".join(lines)


def build_chat_prompt(user_question: str, dataset_brief: str, review_snippets: str) -> str:
    """
    Prompt augmentation: role + constraints + dataset brief + concrete snippets + question.
    """
    return f"""
You are an analyst helping Avalanche staff understand customer sentiment from the provided dataset.
Rules:
- Use ONLY the dataset brief and review snippets below.
- If the answer cannot be supported from the provided context, say: "Not enough information in the provided reviews to answer that."
- Be concise. Prefer bullet points.
- If you make a claim, point to evidence by referencing ORDER_ID(s) from the snippets.

{dataset_brief}

{review_snippets}

QUESTION:
{user_question}
""".strip()


# -----------------------------
# Load and UI
# -----------------------------
df = load_data()

st.title("Avalanche Customer Sentiment")
st.caption("Analyze customer review sentiment by product, date range, and delivery performance. Filter → interpret → act.")

tab_dashboard, tab_ask = st.tabs(["Dashboard", "Ask Reviews"])

# -----------------------------
# Dashboard tab
# -----------------------------
with tab_dashboard:
    st.sidebar.header("Filters")
    st.sidebar.markdown("Use these filters to answer: **What do customers think about X?**")

    if df.empty:
        st.error("No data returned from Snowflake. Check table access and filters.")
        st.stop()

    products_all = sorted(df["PRODUCT"].dropna().unique().tolist()) if "PRODUCT" in df.columns else []
    selected_products = st.sidebar.multiselect(
        "Product(s)",
        options=products_all,
        default=products_all
    )

    if "REVIEW_DATE" in df.columns and df["REVIEW_DATE"].notna().any():
        min_date = df["REVIEW_DATE"].dropna().min().date()
        max_date = df["REVIEW_DATE"].dropna().max().date()
        start_date, end_date = st.sidebar.date_input(
            "Review date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        start_date, end_date = None, None

    late_only = st.sidebar.selectbox("Delivery Status", ["ALL", "On Time", "Late"])

    filtered_df = df.copy()

    if selected_products:
        filtered_df = filtered_df[filtered_df["PRODUCT"].isin(selected_products)]

    if start_date and end_date and "REVIEW_DATE" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["REVIEW_DATE"].dt.date >= start_date) &
            (filtered_df["REVIEW_DATE"].dt.date <= end_date)
        ]

    if late_only == "Late":
        filtered_df = filtered_df[filtered_df["LATE"] == True]
    elif late_only == "On Time":
        filtered_df = filtered_df[filtered_df["LATE"] == False]

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews (filtered)", int(len(filtered_df)))

    avg_sent = float(filtered_df["SENTIMENT_SCORE"].mean()) if len(filtered_df) else 0.0
    c2.metric("Avg sentiment", round(avg_sent, 3))

    late_pct = (float(filtered_df["LATE"].mean()) * 100) if len(filtered_df) else 0.0
    c3.metric("Late %", round(late_pct, 1))

    if len(filtered_df):
        if avg_sent > 0.4:
            verdict = "Strongly positive overall."
        elif avg_sent > 0.1:
            verdict = "Generally positive overall."
        elif avg_sent > -0.1:
            verdict = "Mixed / neutral overall."
        else:
            verdict = "Largely negative overall."
    else:
        verdict = "No data in this slice."

    c4.metric("Interpretation", verdict)

    st.subheader("Data Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    st.subheader("Mean Sentiment by Product")
    if len(filtered_df):
        by_product = (
            filtered_df.groupby("PRODUCT")["SENTIMENT_SCORE"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(by_product)
    else:
        st.info("No rows to chart. Adjust filters.")

    st.subheader("Average Sentiment: Late vs On-Time")
    if len(filtered_df):
        by_late = filtered_df.groupby("LATE")["SENTIMENT_SCORE"].mean()
        st.bar_chart(by_late)
    else:
        st.info("No rows to chart. Adjust filters.")

    st.divider()
    st.subheader("Feedback (quick)")
    fb = st.text_area("What was confusing or missing?", placeholder="Example: I expected a winter-only filter / I can’t tell what sentiment score means.")
    if st.button("Submit feedback"):
        st.success("Thanks — noted. (Prototype tip: wire this into a table later.)")


# -----------------------------
# Ask Reviews tab (prompt augmentation)
# -----------------------------
with tab_ask:
    st.subheader("Ask Reviews")
    st.caption("Answers are based on the dataset brief + a small set of recent review snippets (not full RAG yet).")

    if df.empty:
        st.error("No data available for Q&A.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        user_question = st.text_input(
            "Ask a question",
            placeholder="Example: What are customers complaining about for Winter goggles this month?"
        )

    with col2:
        product_options = ["ALL"] + (sorted(df["PRODUCT"].dropna().unique().tolist()) if "PRODUCT" in df.columns else [])
        scope_product = st.selectbox("Scope product", options=product_options, index=0)

    if "REVIEW_DATE" in df.columns and df["REVIEW_DATE"].notna().any():
        min_date = df["REVIEW_DATE"].dropna().min().date()
        max_date = df["REVIEW_DATE"].dropna().max().date()
        scope_start, scope_end = st.date_input(
            "Scope review date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        scope_start_dt = pd.to_datetime(scope_start)
        scope_end_dt = pd.to_datetime(scope_end)
    else:
        scope_start_dt = None
        scope_end_dt = None

    model = st.selectbox(
        "Model (Cortex)",
        options=["snowflake-arctic", "llama3.1-70b", "mixtral-8x7b"],
        index=0
    )

    if st.button("Answer from reviews", type="primary", disabled=not bool(user_question.strip())):
        with st.spinner("Thinking..."):
            dataset_brief = build_dataset_brief(df)
            snippets = fetch_review_snippets(
                product=None if scope_product == "ALL" else scope_product,
                start_dt=scope_start_dt,
                end_dt=scope_end_dt,
                max_rows=12
            )
            full_prompt = build_chat_prompt(
                user_question=user_question,
                dataset_brief=dataset_brief,
                review_snippets=snippets
            )

            try:
                answer = cortex_complete(full_prompt, model=model)
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Evidence used (snippets)")
                st.code(snippets)

            except Exception as e:
                st.error(f"Chat failed: {e}")
                st.info("If this is Streamlit Cloud, confirm the Snowflake user/role has Cortex permissions for COMPLETE().")
