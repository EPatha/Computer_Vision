import os
import keras_ocr
import re
import shutil

# Path ke folder dataset
source_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_rlt"
destination_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/named_plate"

# Pola regex untuk plat nomor
plate_pattern = re.compile(r"^(AA|AD|K|R|G|H|AB|D|F|E|Z|T|A|B|AG|AE|L|M|N|S|W|P|DK|ED|EA|EB|DH|DR|KU|KT|DA|KB|KH|DC|DD|DN|DT|DL|DM|DB|BA|BB|BD|BE|BG|BH|BK|BL|BM|BN|DE|DG|PA|PB)\\s*\\d{1,4}\\s*[A-Z0-9]{0,4}$")

# Membuat pipeline Keras-OCR
pipeline = keras_ocr.pipeline.Pipeline()

# Membaca folder asli
for folder_name in os.listdir(source_folder):
    original_folder_path = os.path.join(source_folder, folder_name)
    if not os.path.isdir(original_folder_path):
        continue

    # Path folder hasil deteksi
    result_folder_path = os.path.join(destination_folder, folder_name)
    os.makedirs(result_folder_path, exist_ok=True)

    # Iterasi setiap gambar di folder asli
    for image_name in os.listdir(original_folder_path):
        image_path = os.path.join(original_folder_path, image_name)
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Deteksi teks dari gambar
        try:
            images = [keras_ocr.tools.read(image_path)]
            predictions = pipeline.recognize(images)
            detected_text = " ".join([word[0] for word in predictions[0]])

            # Validasi teks menggunakan pola plat nomor
            detected_text = detected_text.replace(" ", "")
            if plate_pattern.match(detected_text):
                new_image_name = f"{detected_text}.png"
            else:
                new_image_name = f"invalid_{image_name}"
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            new_image_name = f"error_{image_name}"

        # Menyalin dan mengganti nama gambar hasil deteksi
        result_image_path = os.path.join(result_folder_path, new_image_name)
        shutil.copy(image_path, result_image_path)

print("Proses selesai. Folder hasil deteksi berada di:", destination_folder)
