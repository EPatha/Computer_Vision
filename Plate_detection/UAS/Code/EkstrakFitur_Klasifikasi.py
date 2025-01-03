import cv2
import os
import glob
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
import pickle

def extract_hog_features(image):
    """Extract HOG features from an image."""
    try:
        hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        return hog_features
    except Exception as e:
        raise ValueError(f"HOG extraction failed: {e}")

def apply_pca(features, n_components=50):
    """Reduce feature dimensions using PCA."""
    try:
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)
        return reduced_features, pca
    except Exception as e:
        raise ValueError(f"PCA failed: {e}")

def extract_template_features(image, template):
    """Extract template matching score as features."""
    try:
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        return np.max(res)
    except Exception as e:
        raise ValueError(f"Template matching failed: {e}")

def process_and_save_features(dataset_path, save_path, template_path=None, use_pca=False, n_components=50):
    # Get all image paths
    image_paths = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
    os.makedirs(save_path, exist_ok=True)

    # Load template if provided
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) if template_path else None

    for image_path in image_paths:
        try:
            print(f"Processing: {image_path}")  # Debug: Log file being processed

            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to read the image: {image_path}")

            # Extract HOG features
            hog_features = extract_hog_features(image)

            # Extract template features if template is available
            template_features = []
            if template is not None:
                template_features = extract_template_features(image, template)

            # Combine features
            features = np.hstack([hog_features, template_features]) if template_features else hog_features

            # Apply PCA if needed
            if use_pca:
                features, pca_model = apply_pca([features], n_components)

            # Get original folder structure
            relative_path = os.path.relpath(image_path, dataset_path)
            feature_save_dir = os.path.join(save_path, os.path.dirname(relative_path))
            os.makedirs(feature_save_dir, exist_ok=True)

            # Save extracted features
            feature_save_path = os.path.join(feature_save_dir, os.path.basename(image_path).replace('.png', '.pkl'))
            with open(feature_save_path, 'wb') as f:
                pickle.dump(features, f)

            print(f"Features extracted and saved for: {image_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    # Dataset and save paths
    dataset_path = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/segmentasi/"
    save_path = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/ekstrak/"

    # Optional: Path to the template image for template matching
    template_path = None  # Replace with the actual template image path if needed

    # Process dataset
    process_and_save_features(dataset_path, save_path, template_path=template_path, use_pca=True, n_components=50)
