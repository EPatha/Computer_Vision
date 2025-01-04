import os
import shutil

# Daftar kategori folder berdasarkan nama pembukaan
categories = {
    "Sicilian": "Sicilian",
    "Benoni": "Benoni",
    "Birds": "Birds",
    "Caro-Kann": "Caro-Kann",
    "Dutch": "Dutch",
    "English": "English",
    "Indian": "Indian-Game",
    "Kings": "Kings",
    "Modern": "Modern",
    "Gruenfeld": "Gruenfeld",
    "Nimzo": "Nimzo",
    "Owens": "Owens",
    "Queens": "Queens",
    "Reti": "Reti",
    "Slav": "Slav",
    "Torre": "Torre",
    "Van": "Van",
    "Amar": "Amar",
    "Anderssen": "Andersse",
    "Clemenz": "Clemenz",
    "French": "French",
    "Grunfeld": "Grunfeld",
    "Kadas": "Kadas",
    "London": "London",
    "Mieses": "Mieses",
    "Polish": "Polish",
    "Tarrasch": "Tarrasch",
    "Ware": "Ware",
}

def simplify_fen_classification(input_dir, output_dir):
    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # Iterasi melalui semua folder dalam direktori input
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Tentukan kategori berdasarkan nama folder
        category = None
        for keyword, category_name in categories.items():
            if keyword in folder:
                category = category_name
                break

        # Jika tidak ada kategori yang cocok, lewati folder
        if not category:
            print(f"Folder '{folder}' tidak sesuai dengan kategori apa pun. Melewati.")
            continue

        # Buat folder kategori di direktori output jika belum ada
        target_dir = os.path.join(output_dir, category)
        os.makedirs(target_dir, exist_ok=True)

        # Pindahkan folder ke kategori yang sesuai
        new_folder_path = os.path.join(target_dir, folder)
        if os.path.exists(new_folder_path):
            print(f"Folder '{new_folder_path}' sudah ada, menggabungkan isinya.")
            for file in os.listdir(folder_path):
                shutil.move(os.path.join(folder_path, file), new_folder_path)
            os.rmdir(folder_path)
        else:
            shutil.move(folder_path, target_dir)

        print(f"Folder '{folder}' dipindahkan ke kategori '{category}'.")

# Contoh penggunaan
input_dir = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/FEN/"
output_dir = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/FEN_Sederhana/"
simplify_fen_classification(input_dir, output_dir)
