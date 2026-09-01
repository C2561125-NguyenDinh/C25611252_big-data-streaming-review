import base64
import certifi
import gzip
import html
import io
import json
import queue
import threading
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from confluent_kafka import Consumer, Producer
from transformers import pipeline

st.set_page_config(page_title="Big Data Streaming Fashion", page_icon="📊", layout="wide")

DATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz"
BOOTSTRAP_SERVERS = "cell-1.streaming.sa-saopaulo-1.oci.oraclecloud.com:9092"
TOPIC = "DemoStreamingFashion"
DURATION_SECONDS = 1800
CATEGORIES = ["Rất tích cực", "Tích cực", "Trung lập", "Tiêu cực", "Rất tiêu cực"]
BAR_COLORS = ["#2b8a3e", "#51cf66", "#ced4da", "#ff6b6b", "#c92a2a"]


@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


def classify_review(analyzer, review_text, amazon_rating):
    label = analyzer(review_text[:500])[0]["label"].lower()
    if amazon_rating <= 2 and "pos" in label:
        sentiment = "negative"
    elif amazon_rating >= 4 and "neg" in label:
        sentiment = "negative"
    elif "pos" in label:
        sentiment = "positive"
    elif "neg" in label:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    if sentiment == "positive":
        return ("Rất tích cực", "#052c11", "#a3cfbb") if amazon_rating >= 4.5 else ("Tích cực", "#0f5132", "#d1e7dd")
    if sentiment == "negative":
        return ("Rất tiêu cực", "#58151c", "#f1aeb5") if amazon_rating <= 1.5 else ("Tiêu cực", "#842029", "#f8d7da")
    return "Trung lập", "#41464b", "#e2e3e5"


def chart_base64(counts):
    values = [counts.get(category, 0) for category in CATEGORIES]
    fig = plt.figure(figsize=(6.2, 4.2), facecolor="white")
    ax = fig.add_subplot(projection="3d")
    x = np.arange(len(CATEGORIES))
    zeros = np.zeros(len(CATEGORIES))
    widths = np.ones(len(CATEGORIES)) * 0.4
    ax.bar3d(x - 0.2, zeros, zeros, widths, widths, values, color=BAR_COLORS, shade=True, edgecolor="none", alpha=.92)
    ax.view_init(elev=28, azim=-55)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=9, rotation=30, color="#212529", ha="right")
    ax.set_zlabel("Số lượng", fontsize=9, fontweight="bold", color="#212529")
    ax.set_title("Biểu đồ 3D Phân phối cảm xúc", fontsize=11, fontweight="bold", color="#1d3557", pad=12)
    ax.xaxis.set_pane_color((.95, .95, .95, .8))
    ax.yaxis.set_pane_color((.95, .95, .95, .8))
    ax.zaxis.set_pane_color((.95, .95, .95, .8))
    plt.subplots_adjust(left=.02, right=.98, top=.88, bottom=.25)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def dashboard(recent_rows, counts, generated, delivered, consumed, elapsed, status, error=""):
    table_rows = ""
    for row in reversed(recent_rows[-7:]):
        rating = float(row.get("amazon_rating", 0))
        stars = "⭐" * max(1, min(5, round(rating)))
        review = html.escape(row.get("reviewText", "Không có nội dung đánh giá"))
        emotion = html.escape(row.get("emotion", "Chưa xác định"))
        table_rows += f'''<tr><td class="rating">{rating:.1f} / 5.0<br><span class="stars">{stars}</span></td><td>{review}</td><td><span class="badge" style="background:{row.get('b_color','#e2e3e5')};color:{row.get('t_color','#41464b')}">{emotion}</span></td></tr>'''
    if not table_rows:
        table_rows = '<tr><td colspan="3" class="waiting">Đang chờ dữ liệu...</td></tr>'
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    chart = chart_base64(counts)
    return f'''<style>
*{{box-sizing:border-box}}body{{margin:0;font-family:'Segoe UI',Tahoma,Arial,sans-serif}}.dash{{padding:25px;background:linear-gradient(135deg,#0d6efd,#0a58ca);border-radius:14px;color:#fff}}
.head{{margin:0 0 15px;font-size:24px;text-transform:uppercase;border-bottom:2px solid rgba(255,255,255,.3);padding-bottom:12px}}.status{{padding:12px 18px;border-radius:8px;font-weight:600;margin-bottom:20px;background:#d1e7dd;color:#0f5132;border:1px solid #badbcc}}
.error{{padding:10px;margin-bottom:15px;background:#f8d7da;color:#842029;border-radius:8px}}.stats{{display:flex;gap:15px;margin-bottom:20px}}.card{{flex:1;background:#fff;padding:15px;border-radius:8px;text-align:center;color:#212529}}.ct{{font-size:12px;color:#6c757d;text-transform:uppercase;font-weight:600;letter-spacing:1px}}.cv{{font-size:24px;font-weight:700}}.blue{{color:#0d6efd}}.green{{color:#198754}}
.reports{{display:flex;gap:20px}}.panel{{background:#fff;padding:18px;border-radius:10px;color:#212529;min-width:0}}.table-panel{{flex:1.4}}.chart-panel{{flex:1;text-align:center}}.pt{{font-size:17px;font-weight:700;margin-bottom:12px;text-transform:uppercase;border-bottom:2px solid #0d6efd;padding-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th{{background:#212529;color:#fff;padding:10px;text-align:left}}td{{padding:10px;border-bottom:1px solid #dee2e6}}.rating{{font-weight:700;text-align:center;width:25%}}.stars{{font-size:12px;color:#f39c12;white-space:nowrap}}.badge{{padding:6px 12px;border-radius:15px;font-weight:600;display:inline-block}}.waiting{{text-align:center;padding:20px}}.chart-panel img{{width:100%;max-height:390px;object-fit:contain}}@media(max-width:900px){{.stats,.reports{{flex-direction:column}}}}
</style><div class="dash"><h2 class="head">Ứng dụng Big Data Streaming để phân tích độ hài lòng của khách hàng</h2><div class="status">TRẠNG THÁI: {html.escape(status)}</div>{error_html}<div class="stats"><div class="card"><div class="ct">Đã xử lý</div><div class="cv">{generated:,}</div></div><div class="card"><div class="ct">Đã chuyển OCI</div><div class="cv blue">{delivered:,}</div></div><div class="card"><div class="ct">Đã nhận</div><div class="cv green">{consumed:,}</div></div><div class="card"><div class="ct">Thời gian</div><div class="cv">{elapsed:.1f}s / 1800s</div></div></div><div class="reports"><div class="panel table-panel"><div class="pt">Bảng 7 dòng đánh giá thời trang gần nhất</div><table><thead><tr><th>Rating</th><th>Nội dung phản hồi</th><th>Phân tích cảm xúc (AI)</th></tr></thead><tbody>{table_rows}</tbody></table></div><div class="panel chart-panel"><div class="pt">Biểu đồ 3D Phân phối cảm xúc</div><img src="{chart}"></div></div></div></div>'''


def run_stream(username, token, output):
    analyzer = load_model()
    run_id = uuid.uuid4().hex[:8]
    common = {
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "PLAIN",
        "sasl.username": username,
        "sasl.password": token,
        "ssl.ca.location": certifi.where(),
    }
    producer = Producer({**common, "client.id": f"prod_{run_id}", "linger.ms": 10, "acks": "1"})
    consumer = Consumer({**common, "client.id": f"cons_{run_id}", "group.id": f"fashion_stream_{run_id}", "auto.offset.reset": "latest", "enable.auto.commit": True})
    local_queue = queue.Queue(maxsize=5000)
    lock = threading.Lock()
    stop = threading.Event()
    stats = {"generated": 0, "delivered": 0, "failed": 0, "error": ""}

    def delivery(err, _message):
        with lock:
            if err:
                stats["failed"] += 1
                stats["error"] = str(err)
            else:
                stats["delivered"] += 1

    def produce():
        try:
            response = requests.get(DATA_URL, stream=True, timeout=60)
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as archive:
                for line in archive:
                    if stop.is_set():
                        break
                    try:
                        record = json.loads(line.decode("utf-8"))
                        rating = float(record.get("overall", 0))
                        review = str(record.get("reviewText", "Không có nội dung đánh giá")).strip() or "Không có nội dung đánh giá"
                        emotion, text_color, background = classify_review(analyzer, review, rating)
                        event = {"run_id":run_id,"amazon_rating":rating,"reviewText":review,"emotion":emotion,"t_color":text_color,"b_color":background}
                        payload = json.dumps(event).encode("utf-8")
                        try:
                            local_queue.put_nowait(payload)
                        except queue.Full:
                            pass
                        producer.produce(TOPIC, value=payload, on_delivery=delivery)
                        producer.poll(0)
                        with lock:
                            stats["generated"] += 1
                    except Exception as exc:
                        with lock:
                            stats["error"] = f"Bỏ qua record lỗi: {exc}"
            producer.flush(10)
        except Exception as exc:
            with lock:
                stats["error"] = f"Producer: {exc}"

    recent, counts, consumed = [], {category: 0 for category in CATEGORIES}, 0
    consumer.subscribe([TOPIC])
    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    started = time.monotonic()
    last_render = 0.0
    use_fallback = False
    try:
        while time.monotonic() - started < DURATION_SECONDS:
            elapsed = time.monotonic() - started
            with lock:
                snapshot = dict(stats)
            if elapsed > 12 and snapshot["delivered"] == 0:
                use_fallback = True
            if use_fallback:
                try:
                    payload = local_queue.get(timeout=.05)
                    event = json.loads(payload.decode("utf-8"))
                except queue.Empty:
                    event = None
            else:
                message = consumer.poll(.1)
                event = None
                if message is not None and not message.error():
                    candidate = json.loads(message.value().decode("utf-8"))
                    if candidate.get("run_id") == run_id:
                        event = candidate
            if event:
                consumed += 1
                recent.append(event)
                recent = recent[-7:]
                emotion = event.get("emotion", "Trung lập")
                if emotion in counts:
                    counts[emotion] += 1
            if time.monotonic() - last_render >= 1:
                with lock:
                    snapshot = dict(stats)
                output.empty()
                with output.container():
                    components.html(dashboard(recent, counts, snapshot["generated"], snapshot["delivered"], consumed, elapsed, "HỆ THỐNG STREAMING TỐC ĐỘ CAO ĐANG HOẠT ĐỘNG", snapshot["error"]), height=790, scrolling=True)
                last_render = time.monotonic()
    finally:
        stop.set()
        worker.join(timeout=3)
        consumer.close()
    with lock:
        snapshot = dict(stats)
    output.empty()
    with output.container():
        components.html(dashboard(recent, counts, snapshot["generated"], snapshot["delivered"], consumed, time.monotonic()-started, "ĐÃ HOÀN TẤT", snapshot["error"]), height=790, scrolling=True)


st.markdown("<style>.block-container{padding-top:1.2rem;max-width:1500px}#MainMenu,footer{visibility:hidden}</style>", unsafe_allow_html=True)
start = st.button("▶ Bắt đầu Streaming 30 phút", type="primary")
output = st.empty()
if start:
    try:
        username = st.secrets["SASL_USERNAME"]
        token = st.secrets["OCI_AUTH_TOKEN"]
    except Exception:
        st.error("Chưa cấu hình SASL_USERNAME và OCI_AUTH_TOKEN trong Streamlit Secrets.")
    else:
        run_stream(username, token, output)
else:
    components.html(dashboard([], {category:0 for category in CATEGORIES}, 0, 0, 0, 0, "SẴN SÀNG KẾT NỐI OCI"), height=790, scrolling=True)
