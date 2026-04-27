import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# 1. SETUP PATHS
# Replace 'att_faces' with your actual folder name if different
data_path = 'att_faces' 

def load_data(path):
    images = []
    labels = []
    # Load face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Loading images and detecting faces...")
    for subfolder in os.listdir(path):
        if not subfolder.startswith('s'): continue
        label = int(subfolder[1:])
        subfolder_path = os.path.join(path, subfolder)
        
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.pgm'):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # 2. FACE DETECTION (Requirement 2)
                # Since ORL images are already cropped, we detect to satisfy the task.
                faces = face_cascade.detectMultiScale(img, 1.1, 4)
                
                # If a face is found, use it; otherwise use the whole image
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_img = img[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (92, 112)) # Keep size consistent
                    images.append(face_img.flatten())
                else:
                    images.append(img.flatten())
                
                labels.append(label)
    return np.array(images), np.array(labels)

# Load data
X, y = load_data(data_path)
n_classes = len(np.unique(y))

# Split into Training and Testing (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. PCA / EIGEN ANALYSIS (Requirement 3)
n_components = 100 # Number of Eigenfaces to keep
print(f"Extracting the top {n_components} eigenfaces from {X_train.shape[0]} faces...")

pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# Transform data into the PCA subspace
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a Classifier (SVM is standard for Eigenfaces)
clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
clf.fit(X_train_pca, y_train)

# 4. PERFORMANCE REPORT (Requirement 4)
y_pred = clf.predict(X_test_pca)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. PLOT ROC CURVE (Requirement 4)
# Since this is a multi-class problem (40 classes), we binarize the output
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = clf.predict_proba(X_test_pca)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting Macro-average ROC
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'orange', 'cyan'])
for i, color in zip(range(5), colors): # Plotting first 5 classes for clarity
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i+1} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - PCA/Eigenfaces')
plt.legend(loc="lower right")
plt.show()

# Optional: Visualize the first Eigenface
eigenface = pca.components_[0].reshape((112, 92))
plt.imshow(eigenface, cmap='gray')
plt.title("Primary Eigenface")
plt.show()