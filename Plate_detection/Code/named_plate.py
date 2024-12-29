import os
import pytesseract
from PIL import Image
import re
import shutil

# Fungsi untuk membaca dan mendeteksi plat nomor
def detect_plate_number(image_path):
    # Membaca gambar
    img = Image.open(image_path)
    
    # Menggunakan pytesseract untuk mengekstrak teks dari gambar
    extracted_text = pytesseract.image_to_string(img, config='--psm 8').strip()
    
    # Menentukan pola untuk plat nomor sesuai dengan yang Anda jelaskan
    # Pola plat nomor: Digit awal (asal kota), Digit tengah (kategori kendaraan), Digit belakang (jenis kendaraan)
    pattern = r"([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z0-9]{1,4})"
    match = re.match(pattern, extracted_text)
    
    if match:
        asal_kota = match.group(1)
        kategori_kendaraan = match.group(2)
        jenis_kendaraan = match.group(3)
        
        # Menggabungkan hasil deteksi menjadi nama file baru
        detected_plate_number = f"{asal_kota}{kategori_kendaraan}{jenis_kendaraan}"
        return detected_plate_number
    else:
        return None

# Fungsi untuk memindahkan dan mengganti nama gambar
def process_images(input_folder, output_folder):
    # Membaca semua subfolder dan gambar dalam folder input
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                
                # Deteksi plat nomor dari gambar
                plate_number = detect_plate_number(image_path)
                
                if plate_number:
                    # Membuat folder output berdasarkan folder asal
                    relative_path = os.path.relpath(root, input_folder)
                    new_folder = os.path.join(output_folder, 'named_plate', relative_path, plate_number)
                    
                    # Membuat folder jika belum ada
                    os.makedirs(new_folder, exist_ok=True)
                    
                    # Membuat path baru untuk gambar dengan nama hasil deteksi
                    new_image_path = os.path.join(new_folder, f"{plate_number}.jpg")
                    
                    # Menyalin gambar ke folder tujuan dengan nama baru
                    shutil.copy(image_path, new_image_path)
                    print(f"Processed {image_path} -> {new_image_path}")
                else:
                    print(f"Failed to detect plate number in {image_path}")
            
# Path ke folder input dan output
input_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_rlt"
output_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset"

# Menjalankan proses
process_images(input_folder, output_folder)
