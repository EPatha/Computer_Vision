import cv2
import torch
import os
from PIL import Image

# Path to YOLOv5 weights and extracted images
yolo_weights = 'yolov5s.pt'  # Gantilah dengan model YOLOv5 yang sudah dilatih
images_dir = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/3.TL.Kartini-20241212T144324Z-001/3.TL.Kartini/'
cropped_output_dir = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/3.TL.Kartini_cropped/'

# Membuat direktori output untuk gambar cropped
os.makedirs(cropped_output_dir, exist_ok=True)

# Memuat model YOLO
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)

# Definisikan kelas yang diminati
classes_of_interest = ['car', 'motorcycle', 'truck', 'bus']

# Proses setiap gambar dalam direktori
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)
    
    # Memuat gambar menggunakan PIL
    img = Image.open(image_path)

    # Melakukan deteksi menggunakan model YOLO
    results = model(img)

    # Mengonversi hasil deteksi ke dataframe pandas
    detections = results.pandas().xyxy[0]

    # Membuat folder untuk gambar yang sedang diproses
    image_folder = os.path.join(cropped_output_dir, os.path.splitext(image_name)[0])
    os.makedirs(image_folder, exist_ok=True)

    # Inisialisasi indeks untuk cropping objek yang terdeteksi pada gambar yang sama
    index = 1

    # Menyaring deteksi dan crop berdasarkan kelas yang diminati
    for _, detection in detections.iterrows():
        if detection['name'] in classes_of_interest:
            # Mendapatkan koordinat bounding box
            x_min, y_min, x_max, y_max = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])

            # Crop area yang terdeteksi
            cropped_img = img.crop((x_min, y_min, x_max, y_max))

            # Membuat nama unik untuk gambar cropped
            cropped_image_name = f"{index}.jpg"
            cropped_image_path = os.path.join(image_folder, cropped_image_name)

            # Menyimpan gambar cropped
            cropped_img.save(cropped_image_path)

            # Menambah indeks untuk deteksi berikutnya
            index += 1

print("Deteksi dan crop selesai! Semua hasil disimpan di:", cropped_output_dir)
