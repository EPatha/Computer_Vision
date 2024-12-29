import os
import shutil

# Direktori sumber dan tujuan
source_dir = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_rlt"
destination_dir = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset"

# Ekstensi file gambar yang ingin dipindahkan
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

# Fungsi untuk memindahkan file
def move_images(source, destination, extensions):
    if not os.path.exists(destination):
        os.makedirs(destination)

    for root, _, files in os.walk(source):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination, file)
                
                # Pastikan tidak ada file dengan nama yang sama di tujuan
                counter = 1
                while os.path.exists(destination_path):
                    name, ext = os.path.splitext(file)
                    destination_path = os.path.join(destination, f"{name}_{counter}{ext}")
                    counter += 1

                # Pindahkan file
                shutil.move(source_path, destination_path)
                print(f"Moved: {source_path} -> {destination_path}")

# Pindahkan semua file gambar
move_images(source_dir, destination_dir, image_extensions)
