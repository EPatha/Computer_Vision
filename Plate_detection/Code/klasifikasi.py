import cv2
import os
import glob
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import pytesseract

# Step 1: Feature Extraction
def extract_hog_features(image):
    """Extract HOG features from an image."""
    try:
        hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        return hog_features
    except Exception as e:
        raise ValueError(f"HOG extraction failed: {e}")

# Step 2: OCR Classification
def classify_with_ocr(image):
    """Classify characters using OCR."""
    text = pytesseract.image_to_string(image, config='--psm 10')
    return text.strip()

# Step 3: SVM Classification
def train_and_classify_svm(features, labels):
    """Train and classify using SVM."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print(classification_report(y_test, predictions))
    return svm

# Step 4: CNN Classification
def train_and_classify_cnn(images, labels):
    """Train and classify using CNN."""
    images = np.array(images).reshape(-1, 28, 28, 1)  # Assuming images are resized to 28x28
    labels = to_categorical(labels, num_classes=36)  # Assuming 36 classes (0-9, A-Z)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(36, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)
    return model

# Main Function to Process Dataset
def process_dataset_and_classify(dataset_path, save_path, classification_method="OCR"):
    image_paths = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
    features, labels = [], []

    for image_path in image_paths:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            # Resize for CNN if needed
            resized_image = cv2.resize(image, (28, 28))

            if classification_method == "OCR":
                label = classify_with_ocr(image)
                print(f"OCR result for {image_path}: {label}")

            elif classification_method == "SVM":
                hog_features = extract_hog_features(image)
                label = os.path.basename(os.path.dirname(image_path))  # Assuming folder names are labels
                features.append(hog_features)
                labels.append(label)

            elif classification_method == "CNN":
                label = os.path.basename(os.path.dirname(image_path))  # Assuming folder names are labels
                features.append(resized_image)
                labels.append(label)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Train classifier based on the method
    if classification_method == "SVM":
        train_and_classify_svm(features, labels)
    elif classification_method == "CNN":
        train_and_classify_cnn(features, labels)

if __name__ == "__main__":
    dataset_path = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/segmentasi/"
    save_path = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/ekstrak/"

    # Choose classification method: "OCR", "SVM", or "CNN"
    classification_method = "OCR"  # Change to "SVM" or "CNN" as needed

    process_dataset_and_classify(dataset_path, save_path, classification_method)
