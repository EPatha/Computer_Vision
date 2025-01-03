import cv2
import os
import glob
import shutil

# Input dan output folder
image_folder = "/home/ep/Documents/Github/Computer_Vision/ESRGAN-master/results/**/*"
output_base_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_plat_numbers"

# Pastikan output folder ada
os.makedirs(output_base_folder, exist_ok=True)

# Inisialisasi detektor Haar Cascade untuk plat nomor
plate_cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
plate_cascade = cv2.CascadeClassifier(plate_cascade_path)

# Fungsi untuk mendeteksi dan memproses plat nomor
def process_image(image_path, output_folder):
    # Baca gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Failed to read image: {image_path}")
        return
    
    # Konversi ke grayscale untuk deteksi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi plat nomor
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 20))
    
    # Ambil nama folder dari path file gambar
    folder_name = os.path.basename(os.path.dirname(image_path))  # Nama folder asal file gambar
    image_output_folder = os.path.join(output_folder, folder_name)
    os.makedirs(image_output_folder, exist_ok=True)

    if len(plates) > 0:
        # Jika plat nomor ditemukan, crop dan simpan
        base_filename = os.path.basename(image_path)
        for (x, y, w, h) in plates:
            cropped_plate = img[y:y+h, x:x+w]
            output_path = os.path.join(image_output_folder, base_filename)
            cv2.imwrite(output_path, cropped_plate)
            print(f"[Cropped] Saved: {output_path}")
        print(f"[Folder Created] {image_output_folder}")
    else:
        # Jika tidak ada plat nomor, tambahkan prefix (not)
        base_filename = os.path.basename(image_path)
        not_output_path = os.path.join(image_output_folder, f"(not){base_filename}")
        shutil.copy(image_path, not_output_path)
        print(f"[No Plate] Saved original with (not): {not_output_path}")
        print(f"[Folder Created] {image_output_folder}")

# Loop melalui semua gambar dalam folder
image_paths = glob.glob(image_folder, recursive=True)

for image_path in image_paths:
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(image_path, output_base_folder)

# Tambahan: Outputkan struktur folder yang diakses
print("\n[INFO] Accessed Folders:")
for root, dirs, files in os.walk(output_base_folder):
    if files or dirs:
        print(f"{root}/*")

print("[INFO] Processing completed.")
