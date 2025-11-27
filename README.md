# 🖐️ Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu (ASL Recognition System)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Đồ án môn học:** Xây dựng ứng dụng chuyển đổi ngôn ngữ ký hiệu thành văn bản và giọng nói theo thời gian thực sử dụng Computer Vision và Machine Learning.

---

## 📖 Giới thiệu

Dự án này phát triển một hệ thống hỗ trợ giao tiếp cho người khiếm thính. Hệ thống sử dụng camera để nhận diện cử chỉ tay và khuôn mặt, sau đó chuyển đổi chúng thành văn bản và âm thanh (Text-to-Speech) ngay lập tức.

Giải pháp kết hợp sức mạnh của **Google MediaPipe** (trích xuất đặc trưng xương khớp) và thuật toán **SVM** (Support Vector Machine) để đảm bảo tốc độ xử lý cao, hoạt động mượt mà trên cả các máy tính cấu hình thấp (không cần GPU rời).

## ✨ Tính năng nổi bật

*   **⚡ Nhận diện thời gian thực (Real-time):** Tốc độ xử lý cao (>30 FPS).
*   **🗣️ Chuyển văn bản thành giọng nói (TTS):**
    *   Hỗ trợ giọng đọc Offline (pyttsx3) phản hồi tức thì.
    *   Hỗ trợ giọng đọc Online (Google TTS) chất lượng cao, tự nhiên.
*   **📊 Giao diện trực quan:**
    *   Hiển thị khung xương tay/mặt lên màn hình.
    *   Thanh đo độ tin cậy (Confidence Bar) cho biết độ chính xác của dự đoán.
*   **🛠️ Dễ dàng mở rộng:** Có sẵn công cụ để tự thu thập dữ liệu và huấn luyện thêm các cử chỉ mới chỉ trong vài phút.

## 🚀 Cài đặt

Đảm bảo bạn đã cài đặt **Python 3.10** hoặc mới hơn.

### 1. Clone dự án
```bash
git clone https://github.com/thanhphandev/nhan-dien-ngon-ngu-ki-hieu.git
cd nhan-dien-ngon-ngu-ki-hieu
```

### 2. Tạo môi trường ảo (Khuyến nghị)
```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

---

## 🎮 Hướng dẫn sử dụng

### 1. Chạy ứng dụng ngay lập tức
Nếu bạn chỉ muốn trải nghiệm mô hình có sẵn:
```bash
streamlit run main.py
```
*   Truy cập vào đường dẫn hiển thị trên terminal (thường là `http://localhost:8501`).
*   Cấp quyền truy cập Camera cho trình duyệt.

### 2. Quy trình huấn luyện cử chỉ mới (Tùy chọn)

Nếu bạn muốn dạy AI hiểu thêm cử chỉ mới (ví dụ: "Tạm biệt"), hãy làm theo 3 bước sau:

**Bước 1: Thu thập dữ liệu**
Đứng trước camera và thực hiện cử chỉ.
```bash
# Thu thập cử chỉ "tam_biet" trong 60 giây
python scripts/capture_pose_data.py --pose_name="tam_biet" --duration=60
```
*Dữ liệu sẽ được lưu vào thư mục `data/tam_biet.npy`.*

**Bước 2: Huấn luyện mô hình**
Chạy script để AI học tất cả dữ liệu trong thư mục `data/`.
```bash
python scripts/train.py --model_name=my_custom_model
```
*Sau khi chạy xong, bạn sẽ thấy file `my_custom_model.pkl` trong thư mục `models/` và biểu đồ đánh giá độ chính xác (Confusion Matrix).*

**Bước 3: Cập nhật cấu hình**
Mở file `config.py` và sửa tên mô hình:
```python
MODEL_NAME = "my_custom_model.pkl"
```
Sau đó chạy lại `streamlit run main.py` để kiểm tra kết quả.

---

## 📂 Cấu trúc dự án

```
├── data/                   # Chứa dữ liệu thô (.npy) đã thu thập
├── docs/                   # Tài liệu hướng dẫn chi tiết
├── models/                 # Chứa file mô hình đã huấn luyện (.pkl) & biểu đồ báo cáo
├── scripts/                # Các script công cụ
│   ├── capture_pose_data.py  # Tool thu thập dữ liệu
│   ├── train.py              # Tool huấn luyện AI
│   └── test_model.py         # Tool test nhanh (không cần Streamlit)
├── utils/                  # Các module chức năng
│   ├── feature_extraction.py # Trích xuất đặc trưng (MediaPipe)
│   ├── model.py              # Class xử lý AI
│   ├── visualizer.py         # Class vẽ đồ họa (xương khớp)
│   ├── tts.py                # Class xử lý giọng nói
│   └── strings.py            # Xử lý văn bản hiển thị
├── config.py               # File cấu hình chung
├── main.py                 # File chính (Giao diện Streamlit)
└── requirements.txt        # Danh sách thư viện
```

## 🛠️ Công nghệ sử dụng

*   **Ngôn ngữ:** Python
*   **Computer Vision:** OpenCV, MediaPipe
*   **Machine Learning:** Scikit-learn (SVM Kernel RBF)
*   **Giao diện:** Streamlit
*   **Xử lý dữ liệu:** NumPy
*   **Trực quan hóa:** Matplotlib, Seaborn

---

**Lưu ý:** Dự án này được thiết kế cho mục đích học tập và nghiên cứu. Độ chính xác có thể bị ảnh hưởng bởi điều kiện ánh sáng và góc quay camera.