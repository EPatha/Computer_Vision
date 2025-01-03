import cv2
import numpy as np
import os
import easyocr
import re

# Pola untuk validasi digit awal (Asal Kota)
valid_prefixes = [
    'AA', 'AD', 'K', 'R', 'G', 'H', 'AB', 'D', 'F', 'E', 'Z', 'T', 'A', 'B', 'AG', 'AE', 'L', 'M', 'N', 
    'S', 'W', 'P', 'DK', 'ED', 'EA', 'EB', 'DH', 'DR', 'KU', 'KT', 'DA', 'KB', 'KH', 'DC', 'DD', 'DN', 
    'DT', 'DL', 'DM', 'DB', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'DE', 'DG', 'PA', 'PB'
]

# Fungsi untuk memvalidasi digit plat nomor sesuai pola
def validate_plate_number(detected_text):
    # Pisahkan teks menjadi bagian-bagian plat nomor
    parts = detected_text.split()
    
    if len(parts) < 3:
        return False  # Tidak ada cukup bagian (prefix, middle, suffix)

    # Validasi digit awal (Asal Kota)
    prefix = parts[0]
    if prefix not in valid_prefixes:
        return False
    
    # Validasi digit tengah (Kategori Kendaraan)
    try:
        middle = int(parts[1])
        if middle < 1 or middle > 9999:
            return False
    except ValueError:
        return False
    
    # Validasi digit belakang (Jenis Kendaraan)
    suffix = parts[2]
    if len(suffix) > 4 or not re.match(r'^[A-Za-z0-9]*$', suffix):
        return False

    return True

# Fungsi untuk meningkatkan gambar
def preprocess_image(image_path):
    # Membaca gambar
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Histogram Equalization untuk meningkatkan kontras
    img = cv2.equalizeHist(img)

    # Adaptive Thresholding untuk memisahkan teks dari latar belakang
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Canny Edge Detection untuk menemukan tepi-tepi penting
    img = cv2.Canny(img, 100, 200)

    # Morphological Transformations untuk memperbaiki objek dalam gambar
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Resize untuk menjaga konsistensi ukuran gambar
    img = cv2.resize(img, (640, 480))
    
    return img

# Fungsi untuk deteksi plat nomor menggunakan kontur dan pola persegi panjang
def detect_plate(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding untuk mendapatkan biner gambar
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Menemukan kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for cnt in contours:
        # Menghitung panjang dan lebar kontur
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        
        # Filter kontur berbentuk persegi panjang (plat nomor)
        if len(approx) == 4:  # Plat nomor biasanya berbentuk persegi panjang
            plates.append(approx)

    return plates, img

# Fungsi untuk melakukan OCR dengan EasyOCR
def ocr_plate(plate_image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(plate_image)
    
    # Mengambil teks yang terdeteksi
    detected_text = " ".join([text[1] for text in result])
    return detected_text

# Fungsi untuk menyimpan gambar yang terdeteksi dan menamai gambar sesuai dengan hasil deteksi
def save_detected_plate(image_path, detected_text, output_dir):
    # Membaca gambar asli
    img = cv2.imread(image_path)
    
    # Menyimpan gambar dengan nama sesuai dengan hasil OCR
    file_name = os.path.basename(image_path)
    new_file_name = f"{detected_text.replace(' ', '_')}.jpg"  # Menggunakan hasil OCR sebagai nama file
    output_path = os.path.join(output_dir, new_file_name)

    cv2.imwrite(output_path, img)
    print(f"Image saved at {output_path}")

# Main function untuk memproses gambar
def process_image(image_path, output_dir):
    # Preprocessing gambar untuk meningkatkan deteksi
    processed_img = preprocess_image(image_path)
    
    # Deteksi plat nomor dengan kontur dan pola persegi panjang
    plates, original_img = detect_plate(image_path)

    # Jika plat nomor ditemukan, lakukan OCR pada tiap plat nomor yang terdeteksi
    if plates:
        for plate in plates:
            # Membuat bounding box untuk plat nomor
            x, y, w, h = cv2.boundingRect(plate)
            plate_img = original_img[y:y+h, x:x+w]
            
            # Menggunakan EasyOCR untuk membaca plat nomor
            detected_text = ocr_plate(plate_img)
            print(f"Detected text: {detected_text}")
            
            # Validasi plat nomor berdasarkan pola
            if validate_plate_number(detected_text):
                print(f"Valid plate number: {detected_text}")
                # Menyimpan gambar hasil deteksi
                save_detected_plate(image_path, detected_text, output_dir)
            else:
                print(f"Invalid plate number: {detected_text}")

# Tentukan folder input dan output
input_folder = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_rlt/'
output_folder = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/named_plate/'

# Proses setiap gambar dalam folder input
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)
    if os.path.isdir(folder_path):
        output_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        
        # Proses setiap gambar dalam subfolder
        for image_name in os.listdir(folder_path):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(folder_path, image_name)
                process_image(image_path, output_subfolder)
