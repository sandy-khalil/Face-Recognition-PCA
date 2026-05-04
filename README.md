# FaceID Analytics | PCA & SVM Face Recognition

A high-performance, lightweight Face Recognition system built with **C++**, **OpenCV**, and **httplib**. This project implements facial recognition using Principal Component Analysis (PCA) and Support Vector Machines (SVM) with a real-time web dashboard.

## 🚀 Key Features

-   **Ultra-Lightweight PCA**: Optimized manual implementation using the covariance trick ($XX^T$), capable of training on high-dimensional image data in milliseconds.
-   **Native C++ Web Server**: Powered by `httplib` for a responsive, single-binary backend.
-   **Modern Web Dashboard**: A glassmorphism-inspired UI for real-time recognition, configuration tuning, and performance analytics.
-   **Optimized Pipeline**: Stripped of "heavy" redundant features like unnecessary face detection for pre-aligned datasets, making it extremely fast.
-   **C++ ROC Analytics**: Native generation and plotting of ROC curves without external Python dependencies.

## 🛠️ Tech Stack

-   **Backend**: C++14/17, OpenCV 4.x
-   **Web Server**: `httplib` (Header-only)
-   **Frontend**: Vanilla HTML5/CSS3/JS (Outfit Font, Glassmorphism)
-   **Data Storage**: `nlohmann/json` (Header-only)

## 📂 Project Structure

```text
Face-Recognition-PCA/
├── main.cpp            # Server logic and API endpoints
├── model.cpp/h         # Optimized PCA and SVM training
├── dataset.cpp/h       # Data loading and stratified splitting
├── metrics.cpp/h       # ROC Curve generation and plotting
├── web/                # Frontend dashboard assets
│   ├── index.html
│   ├── style.css
│   └── script.js
├── libs/               # Header-only dependencies
└── CMakeLists.txt
```

## 🔧 Installation & Build

### Prerequisites
-   CMake ≥ 3.10
-   OpenCV 4.x (`libopencv-dev`)
-   C++14/17 compatible compiler

### Build Steps
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 💻 Usage

### 1. Start the Web Dashboard
Simply run the executable without arguments:
```bash
./build/face_recognition
```
Then navigate to **http://localhost:8000** in your browser.

### 2. CLI Analysis
Run a full accuracy report directly from the terminal:
```bash
./build/face_recognition att_faces 80 0.8 --json
```

## 📊 Performance
On the standard AT&T Database of Faces (400 images), the system achieves:
-   **Accuracy**: ~90-96%
-   **Training Time**: < 1.0s (Optimized)
-   **Recognition Latency**: < 50ms

---
*Built with ❤️ using C++ and OpenCV.*
