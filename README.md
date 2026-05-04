# 👤 FaceID Analytics: High-Performance PCA Face Recognition

[![C++](https://img.shields.io/badge/Language-C%2B%2B14%2F17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B)
[![OpenCV](https://img.shields.io/badge/Library-OpenCV%204.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art, lightweight Face Recognition system built from the ground up in **C++**. This project implements the classic **Eigenfaces** method (Principal Component Analysis) combined with **Support Vector Machines (SVM)** to provide a robust identification pipeline with a modern web-based analytics dashboard.

---

## 🌟 Key Features

### ⚡ Ultra-Fast PCA Engine
We've implemented a custom **Dual Space Covariance Optimization**. Instead of processing a massive $10,000 \times 10,000$ pixel matrix, our system uses the $XX^T$ trick to perform eigendecomposition in a reduced $320 \times 320$ space.
- **Result**: Training time reduced from minutes to **less than 1 second**.

### 📊 Comprehensive Analytics Dashboard
A modern, glassmorphism-inspired web interface powered by a native C++ `httplib` server.
- **Real-time Recognition**: Drag & drop image testing.
- **ROC Curves**: Native multi-class Receiver Operating Characteristic plots generated in C++.
- **Performance Metrics**: Per-class Precision, Recall, and F1-Score reports.

### 🎯 Robust Recognition Pipeline
- **Automated Face Detection**: Integrated OpenCV Haar Cascades for cropping and alignment.
- **Manual Implementation**: PCA and preprocessing steps implemented manually for maximum control and transparency.
- **Stratified Validation**: Automatic data splitting ensures balanced training across all subjects.

---

## 🛠️ Tech Stack

- **Backend**: C++ (OpenCV 4, httplib, nlohmann-json)
- **Frontend**: Modern Vanilla JS, CSS3 (Glassmorphism, Outfit Typography)
- **ML Models**: Principal Component Analysis (Manual) + RBF-Kernel SVM

---

## 📂 Project Structure

```text
Face-Recognition-PCA/
├── main.cpp            # API Endpoints & Server Logic
├── model.cpp/h         # Optimized PCA Math & SVM Training
├── dataset.cpp/h       # Image Preprocessing & Dataset Management
├── metrics.cpp/h       # ROC Analytics & Native Plotting
├── web/                # High-end Dashboard Frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── libs/               # Header-only Dependencies
└── CMakeLists.txt      # Build Configuration
```

---

## 🔧 Installation & Build

### 1. Prerequisites
Ensure you have the following installed:
- **CMake** ≥ 3.10
- **OpenCV** 4.x (`libopencv-dev`)
- A **C++17** compatible compiler (GCC/Clang)

### 2. Clone & Build
```bash
git clone https://github.com/sandy-khalil/Face-Recognition-PCA.git
cd Face-Recognition-PCA
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## 💻 How to Use

### 🌐 Method A: The Web Dashboard (Recommended)
Start the server and access the interactive UI:
```bash
./face_recognition
```
Open your browser to: **[http://localhost:8000](http://localhost:8000)**

### ⌨️ Method B: Command Line Interface
Run a quick analysis and generate a JSON report:
```bash
./face_recognition att_faces 80 0.8 --json
```

---

## 📈 Performance & Results
Based on the **AT&T Database of Faces** (400 images):
- **Accuracy**: 90% – 96%
- **Recognition Latency**: ~30ms
- **Training Time**: ~0.8s

---

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request to improve the PCA math, UI aesthetics, or recognition accuracy.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed with ❤️ for Advanced Computer Vision and Machine Learning.*
