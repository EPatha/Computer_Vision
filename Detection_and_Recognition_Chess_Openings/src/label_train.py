import os
import glob

# Mapping class names to class IDs
class_mapping = {
    "Pawn_White": 0,
    "Pawn_Black": 1,
    "King_White": 2,
    "King_Black": 3,
    "Queen_White": 4,
    "Queen_Black": 5,
    "Bishop_White": 6,
    "Bishop_Black": 7,
    "Knight_White": 8,
    "Knight_Black": 9,
    "Rook_White": 10,
    "Rook_Black": 11
}

# Function to create label text file for each image
def create_labels(image_path):
    # Extract the class name from the image filename (assuming format: "Piece_Color_number.png")
    base_name = os.path.basename(image_path)  # e.g., "Pawn_White_1.png"
    name_parts = base_name.replace(".png", "").split("_")
    
    piece = name_parts[0]  # "Pawn", "King", etc.
    color = name_parts[1]  # "White" or "Black"
    
    # Create the class name key (e.g., "Pawn_White")
    class_name = f"{piece}_{color}"
    
    # Find the class ID from the mapping
    class_id = class_mapping.get(class_name)
    if class_id is None:
        print(f"Warning: Class '{class_name}' not found in mapping.")
        return

    # Define output label file path with the same name as the image file but with .txt extension
    label_file = os.path.splitext(image_path)[0] + ".txt"
    
    # Assuming the bounding box takes the whole image (since it's a single piece image)
    # Format: <class_id> <x_center> <y_center> <width> <height>
    # In this case, the object (piece) takes the entire image, so center is (0.5, 0.5) and size is (1.0, 1.0)
    
    with open(label_file, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    print(f"Label created for {base_name}: {label_file}")

# Directory containing the training images
image_dir = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/detect_catur/data/klasifikasi/images/train"

# Find all PNG images in the directory (recursively)
image_paths = glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)

# Create labels for all images
for image_path in image_paths:
    create_labels(image_path)

print("Label creation complete.")