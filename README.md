# PREPROCESSING METHODS

Đây là một dự án minh họa các kỹ thuật tiền xử lý dữ liệu (preprocessing) cho 3 loại dữ liệu phổ biến trong Data Mining và Machine Learning:

* Dữ liệu ảnh (image dataa)
* Dữ liệu bảng (tabular)
* Dữ liệu văn bản (text)

### Thông tin nhóm

| Họ và Tên          | Mã Số Sinh Viên |
| ------------------ | --------------- |
| Nguyễn Hữu Anh Trí | 23127130        |
| Cao Trần Bá Đạt    | 23127168        |
| Cao Tấn Hoàng Huy  | 23127051        |
| Tô Trần Hoàng Triệu| 23127133        |

---

###  Cấu trúc thư mục

```
Preprocessing_Methods/
│
├── LICENSE                      # Giấy phép sử dụng dự án
├── README.md                    # File mô tả tổng quan dự án và hướng dẫn sử dụng
├── requirements.txt             # Danh sách các thư viện Python cần cài đặt
│
├── data/                        # Thư mục lưu trữ dataset (tự động tạo khi chạy notebook)
│   ├── images/                  # Dữ liệu ảnh (Chest X-Ray Pneumonia)
│   ├── tabular/                 # Dữ liệu bảng (Credit Card Fraud Detection)
│   └── text/                    # Dữ liệu văn bản (IMDB Reviews)
│
├── docs/                        # Thư mục tài liệu
│   └── Report.pdf               # Báo cáo chính thức của dự án (file nộp)
│
└── notebooks/                   # Các notebook thực hành tiền xử lý
    ├── 01_image_preprocessing.ipynb    # Tiền xử lý dữ liệu ảnh
    ├── 02_tabular_preprocessing.ipynb  # Tiền xử lý dữ liệu bảng
    └── 03_text_preprocessing.ipynb     # Tiền xử lý dữ liệu văn bản
```

### Quá trình xử lí dữ liệu:

#### Quá trình xử lí dữ liệu ảnh (Image Preprocessing):

- Có thể coi chi tiết ở file `01_image_preprocessing.ipynb` trong folder `notebooks`
- Dataset sử dụng:Chest X-Ray Pneumonia (Kaggle)
- Nội dung thực hiện:
  * Tải dữ liệu từ Kaggle bằng `kagglehub`
  * Resize ảnh
  * Chuẩn hóa pixel (Normalization)
  * Feature Scaling (MinMaxScaler, StandardScaler)
  * Trực quan hóa ảnh trước và sau xử lý

---

#### Quá trình xử lí dữ liệu bảng (Tabular Data Preprocessing):

- Có thể coi chi tiết ở file `02_tabular_preprocessing.ipynb` trong folder `notebooks`
- Dataset sử dụng: Credit Card Fraud Detection (Kaggle)
- Nội dung thực hiện:
  * Phân tích dữ liệu ban đầu (EDA)
  * Xử lý missing values
  * Feature Selection:

    * Variance Threshold
    * RFE (Recursive Feature Elimination)
  * Feature Scaling:

    * MinMaxScaler
    * StandardScaler
    * RobustScaler
  * Trực quan hóa thống kê

---

#### Quá trùnh xử lí dữ liệu văn bản (Text Preprocessing):

- Có thể coi chi tiết ở file `03_text_preprocessing.ipynb` trong folder `notebooks`
- Dataset sử dụng: IMDB Movie Reviews (50,000 reviews)
- Nội dung thực hiện:
  * Tokenization:

    * Word Tokenize
    * Sentence Tokenize
    * Subword (BPE)
  * Stopword Removal
  * Stemming (Porter, Snowball)
  * Lemmatization
  * Vectorization:

    * Bag of Words
    * TF-IDF
---

### Hướng dẫn cài đặt

#### Yêu cầu hệ thống

* Python ≥ 3.7
* Jupyter Notebook hoặc VS Code
* Internet (lần đầu tải dataset)

---

#### Clone Repository

```bash
git clone https://github.com/AnhTtis/Preprocessing_Methods.git
cd Preprocessing_Methods
```

---

#### Các bước để chạy được project

- Bước 1: Tạo môi trường ảo (Có thể thực hiện hoặc sài thẳng môi trường Python có sẵn trong máy)

    - Windows

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

    - macOS / Linux

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

---

- Bước 2: Cài đặt thư viện cần thiết

```bash
pip install -r requirements.txt
```

- Các thư viện chính:
    - opencv-python
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - kagglehub
    - scikit-learn
    - nltk
    - tokenizers

---

- Bước 3: Chạy Notebook: Vì các notebooks đều xử lí các công việc độc lập nên ta có thể chạy chúng một các độc lập với nhau
    - Cách 1: Jupyter Notebook

    ```bash
    jupyter notebook
    ```

    Vào thư mục `notebooks/` và mở file cần chạy.

    ---

    - Cách 2: VS Code
        1. Mở thư mục chưa s project
        2. Cài extension Jupyter
        3. Chọn interpreter là môi trường `venv`
        4. Run từng cell bằng Shift + Enter hoặc nút bắt đầu ở góc phải trên cùng màn hình
---


#### Link Dataset (Google Drive Backup)

> Trường hợp không dùng KaggleHub, có thể tải tại:

* [Chest X-Ray Dataset](https://drive.google.com/drive/folders/13Pl54LPO3r0L8W95IMqxyygoBo7jDzVG?usp=sharing)

* [Credit Card Fraud Dataset](https://drive.google.com/drive/folders/1w87OCBt4dsMCtCCQ1RmCApcH2BS2fK5t?usp=sharing)

* [IMDB Reviews Dataset](https://drive.google.com/drive/folders/1pPbTk4_cNVlEahsT3LSQEcqjD48kwbEV?usp=sharing)

---

#### Lưu ý quan trọng

* Lần đầu chạy cần Internet để tải dataset
* Dữ liệu sẽ được lưu trong thư mục `data/`
* Text preprocessing sẽ tự tải NLTK resources

Nếu lỗi NLTK, chạy:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```


