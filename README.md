[Readme.md](https://github.com/user-attachments/files/27134544/Readme.md)
Face Recognition using Eigenfaces (PCA) & SVM

A computer vision project that implements facial recognition using Principal Component Analysis (PCA) and Support Vector Machines (SVM). This repository includes a data analysis pipeline and a Flask web dashboard for real-time testing.
🚀 Features

    Automated Face Detection: Uses OpenCV Haar Cascades to detect and crop faces from raw images.

    Dimensionality Reduction: Implements the "Eigenfaces" method via PCA to compress image data while retaining key features.

    Machine Learning Classification: Uses an RBF-kernel SVM to identify subjects with high accuracy.

    Web Dashboard: An interactive Flask interface to upload photos and view recognition results (Subject ID and Confidence Score).

    Performance Metrics: Generates ROC curves and visualizes the top Principal Components (Eigenfaces).

🛠️ Tech Stack

    Language: Python 3.x

    Libraries: Scikit-Learn, OpenCV, Flask, NumPy, Matplotlib

    Dataset: AT&T Database of Faces (ORL Dataset)

📂 Project Structure
code Text

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
2. How to Paste (Recommended Method: VS Code)

If you are using VS Code, it is much easier than the terminal:

    Open your Face-Recognition-PCA folder in VS Code.

    Click the "New File" icon and name it README.md.

    Paste the text you just copied.

    Save it (Ctrl + S).

    In your terminal, run:
    code Bash

    git add README.md
    git commit -m "Add README"
    git push origin main

3. If you are using the Terminal (Nano)

If you are using nano README.md in the terminal:

    After typing nano README.md, right-click to paste.

    The text might look green, white, or blue depending on your terminal's "Syntax Highlighting." This is normal.

    Once you upload it to GitHub, GitHub will read the # and ** symbols and turn it into a beautiful webpage automatically.

Check your GitHub URL now: If you have pushed the file, go to your browser. GitHub will automatically render that "gray" code into a clean, white-background document!
