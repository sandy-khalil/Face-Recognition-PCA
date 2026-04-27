# Face Recognition using Eigenfaces (PCA) & SVM

A computer vision project that implements facial recognition using **Principal Component Analysis (PCA)** and **Support Vector Machines (SVM)**. This repository includes a data analysis pipeline and a **Flask web dashboard** for real-time testing.

## 🚀 Features
*   **Automated Face Detection:** Uses OpenCV Haar Cascades to detect and crop faces from raw images.
*   **Dimensionality Reduction:** Implements the "Eigenfaces" method via PCA to compress image data while retaining key features.
*   **Machine Learning Classification:** Uses an RBF-kernel SVM to identify subjects with high accuracy.
*   **Web Dashboard:** An interactive Flask interface to upload photos and view recognition results (Subject ID and Confidence Score).
*   **Performance Metrics:** Generates ROC curves and visualizes the top Principal Components (Eigenfaces).

## 🛠️ Tech Stack
*   **Language:** Python 3.x
*   **Libraries:** Scikit-Learn, OpenCV, Flask, NumPy, Matplotlib
*   **Dataset:** AT&T Database of Faces (ORL Dataset)

## 📂 Project Structure
```text
Face-Recognition-PCA/
├── app.py              # Flask Web Application
├── att_faces.py        # Model training & Evaluation script
├── templates/          # HTML files for the dashboard
│   └── index.html
├── static/             # Generated plots (Eigenfaces, ROC)
├── requirements.txt    # List of dependencies
└── README.md
