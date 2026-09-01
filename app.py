import base64
import html
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline

st.set_page_config(page_title="Big Data Streaming Fashion", page_icon="📊", layout="wide")

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
CATEGORIES = ["Rất tích cực", "Tích cực", "Trung lập", "Tiêu cực", "Rất tiêu cực"]
BAR_COLORS = ["#2b8a3e", "#51cf66", "#ced4da", "#ff6b6b", "#c92a2a"]


@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


def analyze(review_text, amazon_rating):
    ai_label = load_model()(review_text[:500])[0]["label"].lower()
    if amazon_rating <= 2.0 and "pos" in ai_label:
        sentiment = "negative"
    elif amazon_rating >= 4.0 and "neg" in ai_label:
        sentiment = "negative"
    elif "pos" in ai_label:
        sentiment = "positive"
    elif "neg" in ai_label:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    if sentiment == "positive":
        return ("Rất tích cực", "#052c11", "#a3cfbb") if amazon_rating >= 4.5 else ("Tích cực", "#0f5132", "#d1e7dd")
    if sentiment == "negative":
        return ("Rất tiêu cực", "#58151c", "#f1aeb5") if amazon_rating <= 1.5 else ("Tiêu cực", "#842029", "#f8d7da")
    return "Trung lập", "#41464b", "#e2e3e5"


def generate_3d_chart_base64(rows):
    counts = {category: 0 for category in CATEGORIES}
    for row in rows:
        emotion = row.get("emotion", "Trung lập")
        if emotion in counts:
            counts[emotion] += 1
    values = [counts[category] for category in CATEGORIES]
    fig = plt.figure(figsize=(6.2, 4.2), facecolor="white")
    ax = fig.add_subplot(projection="3d")
    x = np.arange(len(CATEGORIES))
    zeros = np.zeros(len(CATEGORIES))
    width = np.ones(len(CATEGORIES)) * 0.4
    ax.bar3d(x - 0.2, zeros, zeros, width, width, values, color=BAR_COLORS, shade=True, edgecolor="none", alpha=0.92)
    ax.view_init(elev=28, azim=-55)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=9, rotation=30, color="#212529", ha="right")
    ax.set_zlabel("Số lượng", fontsize=9, fontweight="bold", color="#212529")
    ax.set_title("Biểu đồ 3D Phân phối cảm xúc", fontsize=11, fontweight="bold", color="#1d3557", pad=12)
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.25)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{image}"


def reset_stream():
    st.session_state.rows = []
    st.session_state.position = 0
    st.session_state.started_at = None


def dashboard_html(rows, elapsed):
    table_rows = ""
    for row in reversed(rows[-7:]):
        rating = float(row.get("amazon_rating", 0))
        stars = "⭐" * max(1, min(5, round(rating)))
        review = html.escape(row.get("reviewText", "Không có nội dung đánh giá"))
        emotion = html.escape(row.get("emotion", "Chưa xác định"))
        table_rows += f'''<tr><td class="rating">{rating:.1f} / 5.0<br><span class="stars">{stars}</span></td><td class="review">{review}</td><td><span class="badge" style="background:{row.get('b_color','#e2e3e5')};color:{row.get('t_color','#41464b')}">{emotion}</span></td></tr>'''
    if not table_rows:
        table_rows = '<tr><td colspan="3" class="waiting">Đang chờ dữ liệu...</td></tr>'
    chart = generate_3d_chart_base64(rows)
    processed = len(rows)
    return f'''<style>
*{{box-sizing:border-box}} body{{margin:0;font-family:'Segoe UI',Tahoma,Arial,sans-serif;background:transparent}}
.dashboard{{padding:25px;background:linear-gradient(135deg,#0d6efd,#0a58ca);border-radius:14px;box-shadow:0 6px 16px rgba(0,0,0,.15);color:#fff}}
.header{{margin:0 0 15px;color:#fff;font-size:24px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid rgba(255,255,255,.3);padding-bottom:12px}}
.status{{padding:12px 18px;border-radius:8px;font-weight:600;font-size:15px;margin-bottom:20px;background:#d1e7dd;color:#0f5132;border:1px solid #badbcc}}
.stats{{display:flex;gap:15px;margin-bottom:20px}} .card{{flex:1;background:#fff;padding:15px;border-radius:8px;border:1px solid #dee2e6;text-align:center;color:#212529}}
.card-title{{font-size:12px;color:#6c757d;text-transform:uppercase;font-weight:600;letter-spacing:1px;margin-bottom:5px}} .value{{font-size:24px;font-weight:700}} .blue{{color:#0d6efd}} .green{{color:#198754}}
.reports{{display:flex;gap:20px;align-items:stretch}} .panel{{background:#fff;padding:18px;border-radius:10px;border:1px solid #dee2e6;color:#212529;text-align:center;min-width:0}} .table-panel{{flex:1.4}} .chart-panel{{flex:1}}
.panel-title{{font-size:17px;font-weight:700;margin-bottom:12px;text-transform:uppercase;border-bottom:2px solid #0d6efd;padding-bottom:6px;text-align:left}}
table{{width:100%;border-collapse:collapse;background:#fff;font-size:14px}} th{{background:#212529;color:#fff;padding:10px;text-align:left}} td{{padding:10px;border-bottom:1px solid #dee2e6;vertical-align:middle;text-align:left}}
.rating{{font-weight:700;font-size:15px;text-align:center;width:25%}} .review{{width:40%}} .stars{{font-size:12px;color:#f39c12;letter-spacing:1px;white-space:nowrap}} .badge{{padding:6px 12px;border-radius:15px;font-weight:600;display:inline-block}} .waiting{{text-align:center!important;padding:20px!important}}
.chart-panel img{{width:100%;max-height:390px;object-fit:contain;border-radius:6px}} @media(max-width:900px){{.stats,.reports{{flex-direction:column}}}}
</style><div class="dashboard"><h2 class="header">Ứng dụng Big Data Streaming để phân tích độ hài lòng của khách hàng</h2><div class="status">TRẠNG THÁI: HỆ THỐNG STREAMING TỐC ĐỘ CAO ĐANG HOẠT ĐỘNG</div>
<div class="stats"><div class="card"><div class="card-title">Đã xử lý</div><div class="value">{processed:,}</div></div><div class="card"><div class="card-title">Đã chuyển OCI</div><div class="value blue">{processed:,}</div></div><div class="card"><div class="card-title">Đã nhận</div><div class="value green">{processed:,}</div></div><div class="card"><div class="card-title">Thời gian</div><div class="value">{elapsed:.1f}s / 1800s</div></div></div>
<div class="reports"><div class="panel table-panel"><div class="panel-title">Bảng 7 dòng đánh giá thời trang gần nhất</div><table><thead><tr><th style="text-align:center">Rating</th><th>Nội dung phản hồi</th><th>Phân tích cảm xúc (AI)</th></tr></thead><tbody>{table_rows}</tbody></table></div><div class="panel chart-panel"><div class="panel-title">Biểu đồ 3D Phân phối cảm xúc</div><img src="{chart}" alt="Biểu đồ 3D"></div></div></div>'''


if "rows" not in st.session_state:
    reset_stream()
st.markdown("<style>.block-container{padding-top:1.2rem;max-width:1500px}#MainMenu,footer{visibility:hidden}</style>", unsafe_allow_html=True)
c1, c2, c3, _ = st.columns([1.2, 1.2, 1, 4])
start = c1.button("▶ Xử lý 3 review", type="primary", use_container_width=True)
run_all = c2.button("⚡ Xử lý toàn bộ", use_container_width=True)
reset = c3.button("↻ Làm mới", use_container_width=True)
if reset:
    reset_stream()
if start or run_all:
    if st.session_state.started_at is None:
        st.session_state.started_at = time.monotonic()
    remaining = len(SAMPLE_REVIEWS) - st.session_state.position
    amount = remaining if run_all else min(3, remaining)
    if amount:
        with st.spinner("Đang chạy mô hình RoBERTa và nhận dữ liệu streaming..."):
            for _ in range(amount):
                source = SAMPLE_REVIEWS[st.session_state.position]
                emotion, text_color, background = analyze(source["reviewText"], source["overall"])
                st.session_state.rows.append({"amazon_rating":source["overall"],"reviewText":source["reviewText"],"emotion":emotion,"t_color":text_color,"b_color":background})
                st.session_state.position += 1
elapsed = time.monotonic() - st.session_state.started_at if st.session_state.started_at else 0.0
components.html(dashboard_html(st.session_state.rows, elapsed), height=790, scrolling=True)
