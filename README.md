# Eye Open–Close Detection in Real-time

Hệ thống nhận diện trạng thái **mắt mở / mắt nhắm** trong thời gian thực, phục vụ cho các bài toán như phát hiện buồn ngủ, giám sát hành vi người dùng và tương tác người–máy. Dự án bao gồm hai phần chính: **nghiên cứu – đánh giá mô hình** và **triển khai hệ thống demo với giao diện trực quan**.

---

## 👥 Thành viên thực hiện
* **Dương Đình Phương Dao** - 22520202
* **Phương Hoàng Yến** - 22521716
---

## 1. Cấu trúc thư mục
Thư mục `source_code` chứa hai thư mục chính:
* `inference_model`: Chứa mã nguồn huấn luyện, đánh giá và suy luận các mô hình theo nhiều hướng tiếp cận.
* `Predict-eye-state-streamlit`: Ứng dụng demo nhận diện trạng thái mắt thời gian thực với giao diện Streamlit.

---

## 2. Thư mục `inference_model`

Thư mục này phục vụ cho việc **huấn luyện, đánh giá và so sánh các mô hình** theo nhiều hướng tiếp cận:

- **EAR**: Dựa trên các đặc trưng hình học của mắt.
- **OCEC**: Mô hình CNN chuyên biệt cho phân loại mắt mở/nhắm.
- **BlinkLinMulT**: Mô hình dựa trên Transformer khai thác thông tin theo chuỗi thời gian.

Các mô hình được **đánh giá trên bộ dữ liệu CEW (Closed Eyes in the Wild)** nhằm so sánh độ chính xác và hiệu năng thực tế.

🔗 **Link bộ dữ liệu CEW**: [ClosedEyeDatabases](https://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html)

---

## 3. Thư mục `Predict-eye-state-streamlit`

Triển khai hệ thống nhận diện với giao diện trực quan, bao gồm:
- **Nhận diện Offline**: Tải ảnh lên để phân tích.
- **Nhận diện Online**: Sử dụng camera để theo dõi thời gian thực.
- **Visual Feedback**: Hiển thị khung bao (bounding box) và nhãn trạng thái.
- **Thống kê**: Đếm số lần nháy mắt (blink count).

---

## 4. Cài đặt và chạy hệ thống demo

### Bước 1: Di chuyển vào thư mục hệ thống
```bash
cd Predict-eye-state-streamlit
```

### Bước 2: Cài đặt các thư viện cần thiết

```bash
pip install streamlit torch torchvision mediapipe blinklinmult opencv-python numpy pillow
```

### Bước 3: Chạy hệ thống
```bash
streamlit run app.py
```
Sau khi chạy, giao diện hệ thống sẽ được mở trên trình duyệt, cho phép người dùng:

- Tải ảnh để nhận diện trạng thái mắt

- Mở camera để nhận diện theo thời gian thực và thống kê số lần nháy mắt

Ghi chú

- Hệ thống yêu cầu webcam để sử dụng chế độ nhận diện online

- Tốc độ xử lý phụ thuộc vào cấu hình phần cứng và môi trường chạy

- Dự án phục vụ mục đích học tập, nghiên cứu và demo đồ án
