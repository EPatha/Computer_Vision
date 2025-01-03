import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import glob
import re

# Path to the folder containing images
image_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_plat_numbers/**/*"

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Get a list of all image paths in the folder
image_paths = glob.glob(image_folder, recursive=True)

# List to store detection results
detection_results = []
not_detected_files = []

# Define a regular expression pattern for the Indonesian license plate format
plate_pattern = r'^(AA|AD|K|R|G|H|AB|D|F|E|Z|T|A|B|AG|AE|L|M|N|S|W|P|DK|ED|EA|EB|DH|DR|KU|KT|DA|KB|KH|DC|DD|DN|DT|DL|DM|DB|BA|BB|BD|BE|BG|BH|BK|BL|BM|BN|DE|DG|PA|PB)\d{1,4}[A-Z0-9]{0,4}$'

# Process each image
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        not_detected_files.append(image_path)
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply filters and edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Detect the location of the contour (license plate)
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:  # Looking for a quadrilateral (license plate)
            location = approx
            break

    if location is None:
        not_detected_files.append(image_path)
        continue

    # Create a mask for the detected contour
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Extract the region of interest
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # Use EasyOCR to read text from the cropped image
    result = reader.readtext(cropped_image)

    if result:
        text = result[0][-2]  # Extract detected text (plate number)
        
        # Check if the detected text matches the license plate pattern
        if re.match(plate_pattern, text):
            detection_results.append((image_path, text))
        else:
            print(f"Plate in image {image_path} does not match expected format: {text}")
            not_detected_files.append(image_path)
    else:
        not_detected_files.append(image_path)

# Count detected plates
detected_count = len(detection_results)

# Display results of detected plates
print(f"Total detected plates: {detected_count}")
print("Detected Plates and Their File Paths:")
for file, text in detection_results:
    print(f"File: {file}, Plate: {text}")
    
# Display files with no detections
print("\nFiles with no detections:")
for file in not_detected_files:
    print(f"File: {file}")

# Summary of results
print("\n===== Summary =====")
print(f"Total images processed: {len(image_paths)}")
print(f"Total valid plates detected: {detected_count}")

print("\nDetected Plates (matching pattern):")
for file, text in detection_results:
    print(f"File: {file}, Plate: {text}")

print("\nFiles with no detections or invalid plates:")
for file in not_detected_files:
    print(f"File: {file}")
