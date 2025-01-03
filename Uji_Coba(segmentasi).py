import cv2
import numpy as np
import os
import glob

def segment_characters(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Perform connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    # Filter out small components based on area
    min_area = 50  # Adjust based on your dataset
    valid_components = [i for i, stat in enumerate(stats) if stat[cv2.CC_STAT_AREA] >= min_area]

    segmented_chars = []

    for i in valid_components:
        if i == 0:  # Skip background
            continue

        x, y, w, h, area = stats[i, :]

        # Extract the character region
        char_img = binary_img[y:y+h, x:x+w]
        segmented_chars.append((x, char_img))

    # Sort characters based on their x-coordinate
    segmented_chars = sorted(segmented_chars, key=lambda x: x[0])

    return [char[1] for char in segmented_chars]

def process_dataset(dataset_path):
    # Get all image paths
    image_paths = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)

    for image_path in image_paths:
        try:
            # Segment characters
            chars = segment_characters(image_path)

            # Get the original folder structure
            folder_name = os.path.basename(os.path.dirname(image_path))
            base_name = os.path.basename(image_path).split('.')[0]

            # Define the save directory with original folder structure
            save_dir = os.path.join("/home/ep/Documents/Github/myenv/plate-detection-env/dataset/segmentasi", folder_name, base_name)
            os.makedirs(save_dir, exist_ok=True)

            # Save segmented characters
            for i, char in enumerate(chars):
                save_path = os.path.join(save_dir, f"char_{i+1}.png")
                cv2.imwrite(save_path, char)

            print(f"Processed: {image_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    dataset_path = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_rlt/"
    process_dataset(dataset_path)
