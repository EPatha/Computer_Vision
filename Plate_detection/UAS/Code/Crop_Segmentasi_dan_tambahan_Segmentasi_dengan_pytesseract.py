import os
import cv2
import pytesseract
import numpy as np
import re

# Direktori input dan output
input_dir = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/3.TL.Kartini_cropped"
output_dir = "Crop_with_pattern"

# Pola plat nomor
pattern_asal_kota = r"^(AA|AD|K|R|G|H|AB|D|F|E|Z|T|A|B|AG|AE|L|M|N|S|W|P|DK|ED|EA|EB|DH|DR|KU|KT|DA|KB|KH|DC|DD|DN|DT|DL|DM|DB|BA|BB|BD|BE|BG|BH|BK|BL|BM|BN|DE|DG|PA|PB)"
pattern_kategori_kendaraan = r"\d{1,4}"
pattern_jenis_kendaraan = r"[A-Z0-9]{0,4}$"

# Fungsi untuk membuat direktori jika belum ada
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Iterasi folder dan file
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            # Baca gambar
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path)
            if img is None:
                print(f"Error reading image: {file_path}")
                continue

            # Pra-pemrosesan gambar
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Deteksi kontur
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            plate_img = None

            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:  # Cari kontur dengan 4 sisi (persegi panjang)
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    if 2 < aspect_ratio < 5:  # Rasio aspek untuk plat nomor
                        plate_img = img[y:y + h, x:x + w]
                        break

            if plate_img is not None:
                # OCR pada plat nomor
                plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                detected_text = pytesseract.image_to_string(plate_thresh, config='--psm 8 --oem 3').strip()

                # Validasi pola dengan regex
                match = re.match(
                    fr"{pattern_asal_kota}\s{pattern_kategori_kendaraan}\s{pattern_jenis_kendaraan}",
                    detected_text
                )

                if match:
                    plate_number = match.group().replace(" ", "")
                    print(f"Detected Plate Number: {plate_number}")

                    # Simpan gambar plat
                    relative_path = os.path.relpath(root, input_dir)
                    output_folder = os.path.join(output_dir, relative_path)
                    ensure_dir(output_folder)

                    output_path = os.path.join(output_folder, f"{plate_number}.png")
                    cv2.imwrite(output_path, plate_img)
                else:
                    print(f"Invalid plate format: {detected_text}")
            else:
                print(f"No plate detected in: {file_path}")
