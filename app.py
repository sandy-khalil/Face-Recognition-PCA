import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

app = Flask(__name__)

# Global variables to store the trained model and PCA
pca_model = None
classifier = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def train_model():
    global pca_model, classifier
    data_path = 'att_faces'
    images, labels = [], []
    
    # Load data
    for subfolder in os.listdir(data_path):
        if not subfolder.startswith('s'): continue
        label = int(subfolder[1:])
        subfolder_path = os.path.join(data_path, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.pgm'):
                img = cv2.imread(os.path.join(subfolder_path, filename), 0)
                img = cv2.resize(img, (92, 112))
                images.append(img.flatten())
                labels.append(label)
    
    X, y = np.array(images), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train PCA and SVM
    pca_model = PCA(n_components=100, whiten=True).fit(X_train)
    X_train_pca = pca_model.transform(X_train)
    classifier = SVC(kernel='rbf', probability=True).fit(X_train_pca, y_train)
    
    # Generate static plots (only once)
    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(pca_model.components_[i].reshape((112, 92)), cmap='bone')
        plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig('static/eigenfaces_grid.png')
    plt.close()

    return round(classifier.score(pca_model.transform(X_test), y_test) * 100, 2)

@app.route('/')
def home():
    acc = train_model()
    return render_template('index.html', accuracy=acc)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return "No file"
    file = request.files['file']
    
    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Process image: Detect face -> Crop -> Resize (Standardize)
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y:y+h, x:x+w]
    
    img_resized = cv2.resize(img, (92, 112))
    img_flat = img_resized.flatten().reshape(1, -1)
    
    # Transform with PCA and Predict
    img_pca = pca_model.transform(img_flat)
    prediction = classifier.predict(img_pca)[0]
    confidence = np.max(classifier.predict_proba(img_pca)) * 100

    return render_template('index.html', accuracy=83.33, prediction=f"Person #{prediction}", confidence=f"{confidence:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)