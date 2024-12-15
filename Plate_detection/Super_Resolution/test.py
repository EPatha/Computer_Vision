import os  # Tambahkan baris ini untuk mengimpor os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = '/home/ep/Documents/Github/Computer_Vision/ESRGAN-master/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # Ensure you're using the CPU

# Path untuk mencari semua gambar JPG di dalam subfolder
test_img_folder = '/home/ep/Documents/Github/myenv/plate-detection-env/dataset/3.TL.Kartini_cropped/**/*.jpg'

# Initialize the model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)

# Fix the model loading with weights_only=True and map_location=device for CPU
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)

# Set the model to evaluation mode
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
# Menggunakan glob untuk mencari file JPG secara rekursif
for path in glob.glob(test_img_folder, recursive=True):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    folder_name = osp.basename(osp.dirname(path))  # Mendapatkan nama folder induk gambar
    print(idx, base, folder_name)
    
    # Membaca gambar
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255  # Normalisasi gambar
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()  # Convert to tensor
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # Melakukan inferensi
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Mengubah kembali ke format BGR untuk menyimpan gambar
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    # Membuat folder output dengan nama folder induk
    output_dir = osp.join('results', folder_name)
    os.makedirs(output_dir, exist_ok=True)  # Membuat folder jika belum ada
    
    # Menyimpan gambar output di folder yang sesuai
    output_path = osp.join(output_dir, f'{base}_rlt.png')
    cv2.imwrite(output_path, output)
    print(f"Saved result to {output_path}")
