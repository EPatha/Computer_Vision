#install package
import cv2 as cv
from glob import glob
import numpy as np
#ultralytics itu perusahaan yang buat yolo
from ultralytics import YOLO
#pakai model YOLO8 cari diinternet

# Regular pre-trained yolov8 model for car recognition
coco_model = YOLO('yolov8s.pt')

# YOLOv8 model trained to detect number plates (replace with your custom model path)
np_model = YOLO('/home/sembarang/Documents/license_plate_detector.pt')

# Read in test video paths
videos = glob('inputs/*.mp4')
print(videos)

# Read video by index
video = cv.VideoCapture(videos[0])

ret = True
frame_number = -1
# All vehicle class IDs from the COCO dataset (car, motorbike, truck)
vehicles = [2, 3, 5]  # Car, Motorbike, Truck class IDs
vehicle_bounding_boxes = []

# Loop over the video frames
while ret:
    frame_number += 1
    ret, frame = video.read()

    if ret:
        # Use YOLO to detect and track objects in the frame
        detections = coco_model.track(frame, persist=True)[0]

        # Iterate through detections
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection

            # Filter detections based on vehicle class IDs and confidence score
            if int(class_id) in vehicles and score > 0.5:
                vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
                # Draw bounding box on the frame
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv.putText(frame, f'ID: {track_id} Score: {score:.2f}', (int(x1), int(y1)-10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame with detected bounding boxes
        cv.imshow('Vehicle Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
video.release()
cv.destroyAllWindows()
