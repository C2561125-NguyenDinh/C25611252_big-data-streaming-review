# Big Data Streaming – Amazon Fashion Reviews

Ứng dụng Streamlit chạy luồng Amazon Fashion trong 1.800 giây giống notebook:
đọc `reviewText`, phân tích RoBERTa, gửi qua OCI/Kafka và nhận lại bằng Consumer.

Trước khi deploy, thêm `SASL_USERNAME` và `OCI_AUTH_TOKEN` vào Streamlit Secrets.

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Upload toàn bộ các file trong thư mục này lên một GitHub repository public.
2. Mở Streamlit Community Cloud và chọn **Create app**.
3. Chọn repository, branch `main`, main file `app.py`.
4. Nhấn **Deploy**.

Phiên bản Streamlit không chứa OCI username hoặc auth token, vì vậy người chấm
có thể mở và thao tác trực tiếp mà không cần thông tin bí mật.
