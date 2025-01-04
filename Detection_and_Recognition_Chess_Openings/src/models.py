import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

def load_fen_data(input_dir):
    fen_data = []
    labels = []
    
    # Iterasi melalui semua folder dan subfolder dalam direktori input
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith('.fen'):  # Pastikan hanya file dengan ekstensi .fen yang dibaca
                file_path = os.path.join(root, file_name)
                
                # Baca FEN dari file
                with open(file_path, 'r') as file:
                    fen = file.read().strip()  # Mengambil FEN dari file
                    fen_data.append(fen)  # Menambahkan FEN ke list
                    
                    # Tentukan label hanya berdasarkan nama folder teratas (kategori pembukaan)
                    category = os.path.basename(os.path.basename(root))  # Nama folder teratas sebagai label
                    labels.append(category)  # Menambahkan label kategori
                    
    return fen_data, labels

# Membaca data FEN dan labelnya
input_dir = "/home/ep/Documents/Github/Computer_Vision/Klasifikasi_Chess/data/FEN_Sederhana"
fen_data, labels = load_fen_data(input_dir)

# Vektorisasi FEN dengan TF-IDF
vectorizer = TfidfVectorizer(analyzer='word')
X = vectorizer.fit_transform(fen_data)  # Mengubah FEN menjadi vektor fitur
y = np.array(labels)  # Label kategori

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Coba mengurangi n_neighbors untuk menangani kelas dengan sedikit sampel
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)

# Lakukan resampling pada data train
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Membuat model machine learning (contoh: Naive Bayes)
model = MultinomialNB()
model.fit(X_train_resampled, y_train_resampled)  # Latih model dengan data yang sudah di-resample

# Prediksi pada data test
y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Menyimpan model dan vektorisator untuk penggunaan lebih lanjut
joblib.dump(model, 'chess_opening_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
