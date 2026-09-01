import time

import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Amazon Fashion Review Streaming",
    page_icon="📊",
    layout="wide",
)

SAMPLE_REVIEWS = [
    {"overall": 5.0, "reviewText": "Exactly what I wanted. It fits perfectly and looks great."},
    {"overall": 4.0, "reviewText": "Good quality for the price and the color is beautiful."},
    {"overall": 1.0, "reviewText": "Very disappointing. The material feels cheap and uncomfortable."},
    {"overall": 3.0, "reviewText": "The product is acceptable, but the size is slightly different."},
    {"overall": 5.0, "reviewText": "I love this item and would definitely purchase it again."},
    {"overall": 2.0, "reviewText": "It arrived late and did not look like the photos."},
    {"overall": 4.0, "reviewText": "Comfortable, stylish, and suitable for everyday use."},
    {"overall": 1.0, "reviewText": "The stitching came apart after only one use."},
    {"overall": 3.0, "reviewText": "Average product. Nothing special, but it works as expected."},
    {"overall": 5.0, "reviewText": "Fantastic quality. I am extremely happy with this purchase."},
    {"overall": 2.0, "reviewText": "The fabric is too thin and the fit is not good."},
    {"overall": 4.0, "reviewText": "Nice design and good value. I would recommend it."},
]

EMOTION_ORDER = ["Rất tích cực", "Tích cực", "Trung lập", "Tiêu cực", "Rất tiêu cực"]
EMOTION_COLORS = {
    "Rất tích cực": "#198754",
    "Tích cực": "#51cf66",
    "Trung lập": "#adb5bd",
    "Tiêu cực": "#ff6b6b",
    "Rất tiêu cực": "#c92a2a",
}


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )


def classify_review(review_text: str, amazon_rating: float) -> tuple[str, float]:
    analyzer = load_sentiment_model()
    result = analyzer(review_text[:500])[0]
    label = result["label"].lower()
    confidence = float(result["score"])

    if amazon_rating <= 2.0 and "pos" in label:
        sentiment = "negative"
    elif amazon_rating >= 4.0 and "neg" in label:
        sentiment = "negative"
    elif "pos" in label:
        sentiment = "positive"
    elif "neg" in label:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    if sentiment == "positive":
        emotion = "Rất tích cực" if amazon_rating >= 4.5 else "Tích cực"
    elif sentiment == "negative":
        emotion = "Rất tiêu cực" if amazon_rating <= 1.5 else "Tiêu cực"
    else:
        emotion = "Trung lập"

    return emotion, confidence


def reset_stream():
    st.session_state.rows = []
    st.session_state.position = 0
    st.session_state.started_at = None


if "rows" not in st.session_state:
    reset_stream()

st.title("Ứng dụng Big Data Streaming phân tích độ hài lòng khách hàng")
st.caption(
    "Dữ liệu Amazon Fashion • Trường reviewText • Phân tích cảm xúc bằng RoBERTa"
)

with st.sidebar:
    st.header("Điều khiển streaming")
    batch_size = st.slider("Số review mỗi lượt", 1, 5, 3)
    delay = st.slider("Độ trễ mô phỏng (giây)", 0.0, 1.0, 0.2, 0.1)
    start = st.button("▶ Xử lý lượt tiếp theo", type="primary", use_container_width=True)
    run_all = st.button("⚡ Xử lý toàn bộ", use_container_width=True)
    st.button("↻ Làm mới dữ liệu", on_click=reset_stream, use_container_width=True)
    st.divider()
    st.info(
        "Ứng dụng mô phỏng dữ liệu đến liên tục. Mỗi review được phân tích từ "
        "trường reviewText bằng mô hình RoBERTa."
    )

if start or run_all:
    if st.session_state.started_at is None:
        st.session_state.started_at = time.monotonic()

    remaining = len(SAMPLE_REVIEWS) - st.session_state.position
    amount = remaining if run_all else min(batch_size, remaining)

    if amount <= 0:
        st.toast("Đã xử lý toàn bộ dữ liệu.")
    else:
        progress = st.progress(0, text="Đang nhận và phân tích review...")
        with st.spinner("Đang tải/chạy mô hình RoBERTa..."):
            for index in range(amount):
                source = SAMPLE_REVIEWS[st.session_state.position]
                emotion, confidence = classify_review(
                    source["reviewText"], float(source["overall"])
                )
                st.session_state.rows.append(
                    {
                        "amazon_rating": float(source["overall"]),
                        "reviewText": source["reviewText"],
                        "emotion": emotion,
                        "confidence": confidence,
                    }
                )
                st.session_state.position += 1
                progress.progress(
                    (index + 1) / amount,
                    text=f"Đã xử lý {index + 1}/{amount} review trong lượt này",
                )
                if delay:
                    time.sleep(delay)
        progress.empty()

rows = st.session_state.rows
processed = len(rows)
remaining = len(SAMPLE_REVIEWS) - st.session_state.position
elapsed = (
    time.monotonic() - st.session_state.started_at
    if st.session_state.started_at is not None
    else 0.0
)

positive = sum(r["emotion"] in ("Rất tích cực", "Tích cực") for r in rows)
negative = sum(r["emotion"] in ("Rất tiêu cực", "Tiêu cực") for r in rows)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Đã xử lý", processed)
metric_2.metric("Tích cực", positive)
metric_3.metric("Tiêu cực", negative)
metric_4.metric("Còn lại", remaining)

if rows:
    dataframe = pd.DataFrame(rows)
    left, right = st.columns([1.45, 1])

    with left:
        st.subheader("Các review gần nhất")
        display_df = dataframe.iloc[::-1].copy()
        display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.2%}")
        display_df = display_df.rename(
            columns={
                "amazon_rating": "Rating",
                "reviewText": "Nội dung reviewText",
                "emotion": "Cảm xúc AI",
                "confidence": "Độ tin cậy",
            }
        )
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Phân phối cảm xúc")
        counts = dataframe["emotion"].value_counts().reindex(EMOTION_ORDER, fill_value=0)
        chart_df = counts.rename_axis("Cảm xúc").reset_index(name="Số lượng")
        figure = px.bar(
            chart_df,
            x="Cảm xúc",
            y="Số lượng",
            color="Cảm xúc",
            color_discrete_map=EMOTION_COLORS,
        )
        figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(figure, use_container_width=True)

    st.success(
        f"Streaming đang hoạt động — đã phân tích {processed} review trong {elapsed:.1f} giây."
    )
else:
    st.info("Nhấn “Xử lý lượt tiếp theo” để bắt đầu nhận dữ liệu review.")

st.divider()
st.caption(
    "Big Data Streaming Demo | Amazon Fashion Reviews | "
    "Sentiment model: cardiffnlp/twitter-roberta-base-sentiment-latest"
)
