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

def ocr_recognition(image):
    """Recognize characters using OCR (Tesseract)."""
    text = pytesseract.image_to_string(image, config='--psm 6')
    print(f"Recognized Text: {text}")  # Debugging step
    return text
features_all = []  # Menyimpan semua fitur untuk PCA

# Contoh kode untuk ekstraksi fitur
for image_path in image_paths:
    # Ekstraksi HOG atau template features
    features_all.append(extracted_features)

if len(features_all) > 1:
    pca = PCA(n_components=50)
    features_all_reduced = pca.fit_transform(features_all)
# Dalam bagian rekonstruksi string
predicted_text = ocr_recognition(image)  # Gunakan hasil OCR untuk mengidentifikasi plat nomor
if predicted_text.strip():  # Pastikan tidak kosong
    plate_string = predicted_text.strip()
else:
    print("OCR failed to recognize text.")

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

# Step 5: Plate Number Reconstruction
def reconstruct_plate_string(predicted_chars):
    """Reconstruct plate string from classified characters."""
    plate_string = ''.join(predicted_chars)  # Gabungkan hasil karakter yang telah dikenali
    return plate_string


def save_reconstructed_plate(image_name, plate_string, save_path):
    """Save reconstructed plate string as a text file."""
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"{os.path.splitext(image_name)[0]}.txt")
    with open(output_file, 'w') as f:
        f.write(plate_string)

# Main Function to Process Dataset
def process_dataset_and_reconstruct(dataset_path, detected_plate_path, classification_method="OCR"):
    image_paths = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)

    for image_path in image_paths:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            # Classify each character
            if classification_method == "OCR":
                classified_chars = []
                for char_image_path in glob.glob(os.path.join(image_path, "*")):
                    char_image = cv2.imread(char_image_path, cv2.IMREAD_GRAYSCALE)
                    if char_image is not None:
                        classified_chars.append(classify_with_ocr(char_image))

            elif classification_method in ["SVM", "CNN"]:
                # Assume features and labels have been trained for SVM or CNN
                raise NotImplementedError(f"Classification method '{classification_method}' not yet implemented in this step.")

            # Reconstruct plate string
            plate_string = reconstruct_plate_string(classified_chars)
            print(f"Reconstructed plate for {image_path}: {plate_string}")

            # Save reconstructed plate string
            save_reconstructed_plate(os.path.basename(image_path), plate_string, detected_plate_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    dataset_path = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/segmentasi/"
    detected_plate_path = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/detected_plate/"

    # Choose classification method: "OCR", "SVM", or "CNN"
    classification_method = "OCR"  # Change to "SVM" or "CNN" as needed

    process_dataset_and_reconstruct(dataset_path, detected_plate_path, classification_method)
