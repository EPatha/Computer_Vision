import torch
import os
from PIL import Image

# Path to YOLOv5 weights and input images
yolo_weights = 'yolov5s.pt'  # Gantilah dengan model YOLO yang sudah dilatih
cropped_images_dir = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/3.TL.Kartini_cropped/'
plat_number_output_dir = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/plat_numbers/'

# Membuat direktori output untuk plat nomor
os.makedirs(plat_number_output_dir, exist_ok=True)

# Memuat model YOLO
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)

# Definisikan kelas plat nomor (sesuaikan dengan model kamu)
classes_of_interest = ['license_plate']

# Proses setiap folder dalam direktori cropped images
for folder_name in os.listdir(cropped_images_dir):
    folder_path = os.path.join(cropped_images_dir, folder_name)

    if os.path.isdir(folder_path):  # Pastikan hanya folder yang diproses
        # Buat folder output untuk plat nomor
        output_folder = os.path.join(plat_number_output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Proses setiap gambar dalam folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Memuat gambar menggunakan PIL
            img = Image.open(image_path)

            # Melakukan deteksi menggunakan model YOLO
            results = model(img)

            # Mengonversi hasil deteksi ke dataframe pandas
            detections = results.pandas().xyxy[0]

            # Memfilter deteksi untuk plat nomor
            plat_number_detected = False
            for _, detection in detections.iterrows():
                if detection['name'] == 'license_plate':  # Deteksi plat nomor
                    # Mendapatkan koordinat bounding box
                    x_min, y_min, x_max, y_max = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])

                    # Crop gambar untuk plat nomor
                    cropped_img = img.crop((x_min, y_min, x_max, y_max))

                    # Membuat nama unik untuk gambar cropped plat nomor
                    cropped_image_name = f"{os.path.splitext(image_name)[0]}-plat.jpg"
                    cropped_image_path = os.path.join(output_folder, cropped_image_name)

                    # Menyimpan gambar cropped plat nomor
                    cropped_img.save(cropped_image_path)
                    plat_number_detected = True
                    print(f"Plat nomor ditemukan dan disimpan: {cropped_image_name}")

            if not plat_number_detected:
                print(f"Tidak ada plat nomor yang terdeteksi di {image_name}")

print("Proses deteksi dan crop plat nomor selesai! Semua hasil disimpan di:", plat_number_output_dir)
