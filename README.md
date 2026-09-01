# Big Data Streaming – Amazon Fashion Reviews

Ứng dụng Streamlit mô phỏng luồng dữ liệu Amazon Fashion, đọc nội dung từ
`reviewText`, phân tích cảm xúc bằng RoBERTa và tái tạo giao diện dashboard
trong Google Colab, gồm bảng 7 review gần nhất và biểu đồ cột 3D.

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
