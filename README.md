# 🛒 Smart Supermarket Billing System using Computer Vision

An automated supermarket checkout and product recognition system powered by **Deep Learning (MobileNetV2)**, **OpenCV**, and a **Tkinter** desktop graphical user interface. Designed to eliminate manual barcode scanning by recognizing packaged grocery items in real-time through a camera feed and generating instant itemized receipts.

---

## 📌 Project Overview
Traditional retail checkout counters rely heavily on manual barcode scanning, which causes long queues, scanning delays, and barcode damage issues. This project introduces a computer vision-powered checkout assistant that recognizes grocery products from live video frames, looks up their pricing metadata, and updates the customer's cart in real time.

### 🌟 Key Features
- **Real-Time Webcam Product Recognition**:
  - Live video stream captured via OpenCV.
  - Interactive "Capture Product" button triggers instant inference.
- **Transfer Learning with MobileNetV2**:
  - Pre-trained on ImageNet and fine-tuned for grocery item classification.
  - Data augmentation (rotation, zoom, horizontal flip, width/height shifts) for robust invariant feature detection.
  - Light-weight architecture optimized for fast inference on consumer edge devices.
- **Automated Billing & Inventory Lookup**:
  - Automatically matches recognized items to product metadata (`sample products.csv`).
  - Displays product name, category, brand, pack size, and unit price in INR (₹).
  - Calculates dynamic item totals and cumulative bill summary.
- **Dataset Collection Pipeline**:
  - Built-in crawler script (`image download.py`) to systematically curate training imagery using `icrawler`.

---

## 📦 Supported Product Classes
The model is trained to recognize standard Indian supermarket items:
- **Staples**: Tata Salt, Aashirvaad Atta, Fortune Sunflower Oil, Freedom Rice Bran Oil
- **Dairy & Packaged Foods**: Amul Butter, Maggi Noodles, Kellogg's Oats
- **Snacks & Biscuits**: Parle-G, Lays Magic Masala, Oreo Biscuits, Britannia Milk Bikis
- **Confectionery**: Snickers Chocolate, 5 Star Oreo
- **Personal Care & Household**: Dettol Handwash, Surf Excel Detergent, Brooke Bond Red Label Tea

---

## 🗂️ Project Structure

```text
├── GUI app.py                         # Tkinter + OpenCV interactive billing application
├── model training.py                 # MobileNetV2 transfer learning & training script
├── image download.py                 # Automated web crawler for dataset gathering
├── sample products.csv               # Product database (pricing, brand, category)
├── model_trained/
│   └── product_classifier.h5        # Serialized pre-trained MobileNetV2 weights (28MB)
├── products_images/                  # Curated product image dataset organized by class
├── Smart billing system report.pages # Detailed project report documentation
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.9+** and a working webcam connected to your system.

### 2. Clone the Repository
```bash
git clone https://github.com/Yashu0809/supermarket-billing-system.git
cd supermarket-billing-system
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python "GUI app.py"
```

1. Hold the grocery item in front of your webcam.
2. Click **Capture Product**.
3. The system predicts the item, displays the price, and adds it to your bill automatically.

---

## 🏋️ Training the Model
To re-train or fine-tune the classifier with new products:
1. Place product images in `products_images/<product_folder_name>/` (or use `image download.py`).
2. Run the training script:
```bash
python "model training.py"
```
The newly trained weights will be saved into `model_trained/product_classifier.h5`.

---

## 🛠️ Tech Stack
- **Deep Learning**: TensorFlow, Keras (MobileNetV2)
- **Computer Vision**: OpenCV (`cv2`)
- **GUI Framework**: Tkinter, PIL (Pillow)
- **Data Handling**: Pandas, NumPy
- **Dataset Scraping**: icrawler

---

## 👤 Author
**Yashwanth Jillella**
- GitHub: [@Yashu0809](https://github.com/Yashu0809)
