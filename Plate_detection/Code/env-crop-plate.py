import os
import cv2
import torch

# Path ke folder input dan output
input_dir = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/3.TL.Kartini_cropped/'
output_dir = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/plat_numbers/'

# Buat direktori output utama jika belum ada
os.makedirs(output_dir, exist_ok=True)

# Memuat model YOLO (pastikan model sudah dilatih untuk mendeteksi plat nomor)
yolo_weights = 'yolov5s.pt'  # Ganti dengan path model YOLO Anda
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)

# Fungsi untuk proses crop kotak plat nomor (tanpa membaca teks)
def process_image(image_path, output_folder):
    # Load gambar menggunakan OpenCV
    img = cv2.imread(image_path)

    # Periksa apakah gambar berhasil dimuat
    if img is None:
        print(f"Error: Tidak dapat memuat gambar {image_path}")
        return

    # Konversi BGR ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Deteksi menggunakan YOLO
    results = model(img_rgb)

    # Ambil deteksi kotak plat nomor
    detections = results.pandas().xyxy[0]
    plat_number_detected = False

    for _, detection in detections.iterrows():
        if detection['name'] == 'license_plate':  # Deteksi kotak plat nomor
            # Mendapatkan koordinat bounding box
            x_min, y_min, x_max, y_max = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])

            # Crop gambar kotak plat nomor
            cropped_img = img[y_min:y_max, x_min:x_max]

            # Simpan hasil crop
            os.makedirs(output_folder, exist_ok=True)  # Buat folder output jika belum ada
            output_path = os.path.join(output_folder, os.path.basename(image_path).replace('.jpg', '-plat.jpg'))
            cv2.imwrite(output_path, cropped_img)
            print(f"Plat nomor disimpan: {output_path}")
            plat_number_detected = True

    if not plat_number_detected:
        print(f"Tidak ada plat nomor yang terdeteksi di {image_path}")

# Iterasi rekursif melalui folder dan subfolder
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        # Proses hanya file gambar
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file_name)

            # Buat path folder output yang sesuai
            relative_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, relative_path)

            # Proses gambar
            process_image(file_path, output_folder)

print("Proses deteksi dan crop selesai!")
