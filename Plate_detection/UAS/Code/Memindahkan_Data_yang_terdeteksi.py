import os
import shutil
from pathlib import Path

# Alamat folder sumber dan tujuan
source_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_plat_numbers"
destination_folder = "/home/ep/Documents/Github/myenv/plate-detection-env/dataset/cropped_rlt"

# Ekstensi file gambar yang valid
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Pola nama file yang harus dikecualikan
excluded_patterns = {f"(not){i}_rlt.png" for i in range(1, 31)}

# Iterasi melalui semua file dalam folder sumber
for file_path in Path(source_folder).rglob("*"):
    if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
        # Cek jika nama file ada dalam daftar yang dikecualikan
        if file_path.name not in excluded_patterns:
            # Ambil nama asli folder dan nama file
            relative_path = file_path.relative_to(source_folder)
            folder_name = relative_path.parent
            file_name = file_path.name

            # Tentukan path folder tujuan
            new_folder_path = Path(destination_folder) / folder_name

            # Buat folder baru jika belum ada
            new_folder_path.mkdir(parents=True, exist_ok=True)

            # Salin file ke folder tujuan
            shutil.copy(file_path, new_folder_path / file_name)

print("Proses selesai!")
