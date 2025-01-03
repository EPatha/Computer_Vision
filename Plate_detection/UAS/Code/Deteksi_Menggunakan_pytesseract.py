import cv2
import pytesseract
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend like Agg
import matplotlib.pyplot as plt

# Set Tesseract path if necessary
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Uncomment and set path if needed

# Load the image
img = cv2.imread('/home/ep/Documents/Github/Computer_Vision/ANPRwithPython-main/9_rlt.png')

# Resize the image for better processing (optional, but helpful for clearer results)
img_resized = cv2.resize(img, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding for better contrast
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)

# Optionally, apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

# Detect edges using Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Find contours to locate the plate
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize location variable to store the license plate contour
location = None

# Loop through contours and find the rectangular one (plate)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:  # A rectangle has 4 corners
        location = approx
        break

# If a plate is found, process it
if location is not None:
    # Mask the plate area
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0, 255, -1)
    plate = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
    # Convert the plate area to grayscale
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance the plate area
    _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations (erosion and dilation) to remove noise and enhance characters
    kernel = np.ones((3, 3), np.uint8)
    plate_thresh = cv2.morphologyEx(plate_thresh, cv2.MORPH_CLOSE, kernel)

    # Use pytesseract to extract text with a custom configuration
    custom_config = r'--psm 8 --oem 3'  # PSM 8 is for single word, and OEM 3 uses both LSTM and legacy OCR
    plate_number = pytesseract.image_to_string(plate_thresh, config=custom_config)

    # Print the detected plate number
    print(f"Detected Plate Number: {plate_number.strip()}")

    # Visualize the detected plate area
    plt.imshow(cv2.cvtColor(plate_thresh, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Plate: {plate_number.strip()}")
    plt.axis('off')
    plt.savefig('detected_plate.png', bbox_inches='tight', pad_inches=0)  # Save image
else:
    print("No plate detected")
