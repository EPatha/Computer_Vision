import cv2
import glob
import os
import numpy as np
import imutils

# Path to the folder containing images
image_folder = "/home/ep/Documents/Github/Computer_Vision/ESRGAN-master/results/**/*"

# Base output folder where cropped plates will be saved
output_base_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_plat_numbers"

# Ensure the output folder exists
os.makedirs(output_base_folder, exist_ok=True)

# Get a list of all image paths in the folder
image_paths = glob.glob(image_folder, recursive=True)

# Process each image
for image_path in image_paths:
    print(f"Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
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
        # If no plate is detected, save the original image with a new name
        not_detected_name = f"i(not detected)_{os.path.basename(image_path)}"
        output_path = os.path.join(output_base_folder, not_detected_name)
        cv2.imwrite(output_path, img)
        print(f"No plate detected. Saved as: {not_detected_name}")
        continue

    # Create a mask for the detected contour
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Extract the region of interest (license plate area)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = img[x1:x2+1, y1:y2+1]

    # Save the cropped plate in the same folder structure
    relative_path = os.path.relpath(image_path, start=os.path.dirname(image_folder))
    folder_structure = os.path.dirname(relative_path).replace(os.sep, "_")

    # Create the output folder path based on the folder structure
    output_folder = os.path.join(output_base_folder, folder_structure)
    os.makedirs(output_folder, exist_ok=True)

    output_name = f"{os.path.basename(image_path)}"
    output_path = os.path.join(output_folder, output_name)

    # Save the cropped plate image
    cv2.imwrite(output_path, cropped_image)
    print(f"Plate detected. Cropped plate saved as: {output_path}")
