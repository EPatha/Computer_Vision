import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import glob
import re
import nbformat

# Load the notebook file
file_path = '/home/ep/Documents/Github/Computer_Vision/ANPR_Loop_Cropped.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Regex pattern for validating license plates
plate_pattern = r"^(?:AA|AD|K|R|G|H|AB|D|F|E|Z|T|A|B|AG|AE|L|M|N|S|W|P|DK|ED|EA|EB|DH|DR|KU|KT|DA|KB|KH|DC|DD|DN|DT|DL|DM|DB|BA|BB|BD|BE|BG|BH|BK|BL|BM|BN|DE|DG|PA|PB)\s?[1-9]\d{0,3}\s?[A-Z0-9]{0,4}$"

# Additional code to validate license plates and save only valid ones
validation_code = """
import re

# Function to validate license plate
def is_valid_plate(plate):
    pattern = r"^(?:AA|AD|K|R|G|H|AB|D|F|E|Z|T|A|B|AG|AE|L|M|N|S|W|P|DK|ED|EA|EB|DH|DR|KU|KT|DA|KB|KH|DC|DD|DN|DT|DL|DM|DB|BA|BB|BD|BE|BG|BH|BK|BL|BM|BN|DE|DG|PA|PB)\\s?[1-9]\\d{0,3}\\s?[A-Z0-9]{0,4}$"
    return re.match(pattern, plate.strip())

# Validate detected plate numbers and save valid ones
valid_plate_count = 0
for img_path, detected_plate in detected_plates.items():
    if is_valid_plate(detected_plate):
        valid_plate_count += 1
        # Save cropped image in corresponding folder
        save_cropped_image(img_path, detected_plate)
        print(f"Valid Plate Detected and Saved: {detected_plate}")
    else:
        print(f"Invalid Plate Detected: {detected_plate}")

print(f"Total Valid Plates Detected: {valid_plate_count}")
"""

# Append the validation code as a new cell in the notebook
new_cell = nbformat.v4.new_code_cell(validation_code)
notebook.cells.append(new_cell)

# Save the modified notebook
output_path = '/home/ep/Documents/Github/Computer_Vision/ANPR_Loop_Cropped_Validated.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

output_path



# Path to the folder containing images
image_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_plat_numbers/**/*"

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Get a list of all image paths in the folder
image_paths = glob.glob(image_folder, recursive=True)

# List to store detection results
detection_results = []
not_detected_files = []

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
        detection_results.append((image_path, text))
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
print(f"Total detected plates: {detected_count}")

print("\nDetected Plates:")
for file, text in detection_results:
    print(f"File: {file}, Plate: {text}")

print("\nFiles with no detections:")
for file in not_detected_files:
    print(f"File: {file}")



import re

def is_valid_license_plate(plate):
    # Define the patterns for each part of the license plate
    region_pattern = r'^(AA|AD|K|R|G|H|AB|D|F|E|Z|T|A|B|AG|AE|L|M|N|S|W|P|DK|ED|EA|EB|DH|DR|KU|KT|DA|KB|KH|DC|DD|DN|DT|DL|DM|DB|BA|BB|BD|BE|BG|BH|BK|BL|BM|BN|DE|DG|PA|PB)$'
    category_pattern = r'^[1-9][0-9]{0,3}$'  # 1 to 9999
    type_pattern = r'^[A-Z0-9]{0,4}$'  # 0 to 4 alphanumeric characters
    
    # Split the license plate into its components
    parts = plate.split(" ")
    if len(parts) != 3:
        return False
    
    region, category, type_ = parts
    
    # Validate each part of the license plate
    if not re.match(region_pattern, region):
        return False
    if not re.match(category_pattern, category):
        return False
    if not re.match(type_pattern, type_):
        return False
    
    return True

# Example usage
plates = [
    "AB 1234 XY",
    "A 1 X",
    "DK 9999",
    "ZZ 1234 AB",  # Invalid region
    "AA 10000 AB",  # Invalid category
    "AB 1234 XYZ12"  # Invalid type
]

for plate in plates:
    print(f"Plate {plate} is valid: {is_valid_license_plate(plate)}")
