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

🔧 Setup & Installation

    Clone the Repository:
    code Bash

    git clone https://github.com/sandy-khalil/Face-Recognition-PCA.git
    cd Face-Recognition-PCA

    Install Dependencies:
    code Bash

    pip install -r requirements.txt

    Download the Dataset:
    The dataset is not included in this repo due to size.

        Download the AT&T Database of Faces.

        Extract it into a folder named att_faces in the root directory.

💻 Usage
1. Training and Analysis

To train the model and see the performance report (Accuracy and ROC Curve):
code Bash

python att_faces.py

2. Run the Web Dashboard

To launch the interactive interface:
code Bash

python app.py

Then navigate to http://127.0.0.1:5000 in your web browser.
📊 Results

The model typically achieves 80-90% accuracy on the AT&T dataset. The "Eigenfaces" represent the statistical variations across faces, allowing the SVM to classify individuals in a reduced 100-dimensional space instead of raw pixel space.
📝 License

Distributed under the MIT License. See LICENSE for more information.
code Code

---

### Part 2: How to add it to your GitHub

Since you have already pushed your project, follow these steps in your terminal to add the README:

1.  **Create the file:**
    ```bash
    nano README.md
    ```
    *(Or just create a new file named `README.md` in VS Code/Notepad and paste the content above into it).*

2.  **Add the file to Git:**
    ```bash
    git add README.md
    ```

3.  **Commit the change:**
    ```bash
    git commit -m "Add professional README"
    ```

4.  **Push to GitHub:**
    ```bash
    git push origin main
    ```

### Why this README is good for your portfolio:
1.  **Context:** It explains *what* Eigenfaces and PCA are.
2.  **Instructions:** It clearly tells a recruiter or another developer how to run your code.
3.  **Visuals:** It mentions the results and tech stack, making it easy to skim.

**Note:** If you want the images (ROC curve and Eigenfaces) to show up on the GitHub page, remember to upload the `.png` files from your `static/` folder directly to GitHub using the "Upload Files" button on the website!
