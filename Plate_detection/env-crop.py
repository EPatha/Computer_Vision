import cv2
import torch
import os
from PIL import Image

# Path to YOLOv5 weights and extracted images
yolo_weights = 'yolov5s.pt'  # Replace with your custom-trained weights if available
images_dir = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/3.TL.Kartini-20241212T144324Z-001/3.TL.Kartini/'
cropped_output_dir = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/3.TL.Kartini_cropped/'

# Create output directory for cropped images
os.makedirs(cropped_output_dir, exist_ok=True)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)

# Define the classes of interest
classes_of_interest = ['car', 'motorcycle', 'truck', 'bus']

# Process each image in the directory
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)

    # Load the image
    img = Image.open(image_path)

    # Perform detection
    results = model(img)

    # Convert results to pandas dataframe
    detections = results.pandas().xyxy[0]

    # Filter detections by class
    for _, detection in detections.iterrows():
        if detection['name'] in classes_of_interest:
            # Crop the detected area
            x_min, y_min, x_max, y_max = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            cropped_img = img.crop((x_min, y_min, x_max, y_max))

            # Save cropped image
            cropped_image_path = os.path.join(cropped_output_dir, f"cropped_{image_name}")
            cropped_img.save(cropped_image_path)

print("Deteksi dan crop selesai! Semua hasil disimpan di:", cropped_output_dir)
