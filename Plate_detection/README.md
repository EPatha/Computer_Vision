# Belajar Plate Detection dengan YOLO

## Pendahuluan
Deteksi plat nomor adalah salah satu aplikasi dari visi komputer yang dapat digunakan untuk sistem pengawasan lalu lintas, pengelolaan parkir otomatis, dan banyak lagi. Dalam proyek ini, kita akan mempelajari cara mendeteksi plat nomor menggunakan model YOLO (You Only Look Once).

## Prasyarat
Sebelum memulai, pastikan Anda telah menginstal:
- Python 3.8 atau lebih baru
- PyTorch (untuk YOLOv5 atau YOLOv8)
- OpenCV
- Pandas dan Numpy
- CUDA (opsional, untuk mempercepat inferensi dengan GPU)

## Langkah-langkah

### 1. **Persiapan Dataset**
Untuk pelatihan, Anda memerlukan dataset plat nomor yang sudah diannotasi. Beberapa pilihan:
- **[CCPD Dataset](https://github.com/detectRecog/CCPD)**: Dataset plat nomor kendaraan Cina.
- Kumpulkan gambar sendiri dan anotasi menggunakan alat seperti LabelImg.

**Struktur Folder Dataset**
```
Dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

### 2. **Pilih Versi YOLO**
- **YOLOv4**: Menggunakan framework Darknet.
- **YOLOv5/YOLOv8**: Lebih user-friendly dan cocok untuk Python.

### 3. **Installasi YOLOv5**
Clone repository YOLOv5:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### 4. **Pelatihan Model**
Edit file `data.yaml` untuk mendefinisikan dataset:
```yaml
path: ./Dataset
train: images/train
val: images/val
nc: 1  # Jumlah kelas (plat nomor)
names: ['plate']
```

Jalankan pelatihan:
```bash
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
```

### 5. **Inferensi**
Gunakan model terlatih untuk mendeteksi plat nomor:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source path_to_image_or_video
```

### 6. **Evaluasi Model**
Gunakan metrik seperti mAP (mean Average Precision) untuk mengevaluasi performa model. YOLO menyediakan log evaluasi selama pelatihan.

## Hasil
- **Model**: YOLOv5 atau YOLOv8
- **Dataset**: CCPD atau dataset buatan sendiri
- **Akurasi**: Tergantung pada kualitas dataset dan jumlah epoch

## Sumber Daya
- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [CCPD Dataset](https://github.com/detectRecog/CCPD)
- [LabelImg](https://github.com/heartexlabs/labelImg)

## Lisensi
Proyek ini menggunakan lisensi MIT. Anda bebas menggunakannya untuk tujuan pembelajaran.
