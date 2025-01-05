import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# Direktori data
train_dir = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/train"
val_dir = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/val"

# Data Augmentation dan Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resolusi input gambar
    batch_size=32,
    class_mode="categorical"  # Menggunakan klasifikasi multi-kelas
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Model CNN dengan Transfer Learning (VGG16)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layer awal

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation="softmax")  # Output sesuai jumlah kelas
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Melatih Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Simpan Model
model.save("chess_image_classifier.h5")
