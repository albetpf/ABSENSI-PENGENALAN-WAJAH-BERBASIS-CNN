# Import library untuk berbagai kebutuhan seperti manipulasi file, pengolahan gambar, machine learning, dan UI
import os
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import library deep learning dan machine learning
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Konfigurasi parameter untuk model dan data
IMG_SIZE = 225  # Ukuran gambar untuk input model
BATCH_SIZE = 32  # Jumlah batch dalam pelatihan
EPOCHS = 20  # Jumlah epoch untuk pelatihan model
RANDOM_SEED = 42  # Seed acak untuk konsistensi hasil

# Path untuk direktori data dan file model
data_dir = "data"  # Direktori data gambar
model_dir = "models"  # Direktori untuk menyimpan model
attendance_file = "absen.csv"  # File untuk menyimpan data absensi

# Membuat folder jika belum ada
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Inisialisasi Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
label_encoder = LabelEncoder()  # Label encoder untuk konversi nama ke angka

# Fungsi untuk menangkap gambar wajah
def capture_images(name: str, num_images: int):
    st.info("Mengaktifkan kamera...")  # Menampilkan pesan UI
    cap = cv2.VideoCapture(0)  # Membuka kamera
    if not cap.isOpened():
        st.error("Kamera tidak dapat diakses.")  # Notifikasi jika kamera gagal dibuka
        return

    # Membuat folder khusus untuk nama yang dimasukkan
    person_dir = os.path.join(data_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    count = 0  # Counter untuk jumlah gambar yang diambil
    while count < num_images:
        ret, frame = cap.read()  # Membaca frame dari kamera
        if not ret:
            st.error("Gagal membaca dari kamera.")  # Notifikasi jika frame gagal dibaca
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ubah ke grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(255, 255))

        # Simpan gambar yang terdeteksi
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Crop wajah
            resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))  # Resize gambar
            file_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, resized)  # Simpan gambar ke file
            count += 1  # Tambah counter
            st.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), caption=f"Captured: {count}/{num_images}")

            if count >= num_images:
                break

    cap.release()  # Menutup kamera
    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV
    st.success(f"Berhasil mengumpulkan {count} gambar untuk {name}.")

# Fungsi untuk memuat dan memproses gambar dari direktori
def load_and_preprocess_data():
    images, labels = [], []
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):  # Cek jika item adalah direktori
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                try:
                    image = cv2.imread(image_path)  # Baca gambar
                    if image is not None:
                        processed_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0  # Normalisasi
                        images.append(processed_image)
                        labels.append(person_name)  # Tambah label
                except Exception as e:
                    st.warning(f"Gagal memproses {image_path}: {str(e)}")

    images = np.array(images)  # Konversi ke array numpy
    labels = np.array(labels)
    return images, labels

# Fungsi untuk membuat arsitektur CNN
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # Layer konvolusi 1
        MaxPooling2D(pool_size=(2, 2)),  # Layer pooling 1
        Dropout(0.25),  # Dropout 25%

        Conv2D(64, (3, 3), activation='relu'),  # Layer konvolusi 2
        MaxPooling2D(pool_size=(2, 2)),  # Layer pooling 2
        Dropout(0.25),  # Dropout 25%

        Flatten(),  # Meratakan hasil konvolusi
        Dense(128, activation='relu'),  # Dense layer
        Dropout(0.5),  # Dropout 50%
        Dense(num_classes, activation='softmax')  # Output layer dengan softmax
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Kompilasi model
    return model

# Fungsi untuk melatih model menggunakan data yang sudah diproses
def train_model():
    images, labels = load_and_preprocess_data()
    encoded_labels = label_encoder.fit_transform(labels)  # Mengkodekan label menjadi angka

    X_train, X_temp, y_train, y_temp = train_test_split(images, encoded_labels, test_size=0.3, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

    model = create_model(len(label_encoder.classes_))  # Membuat model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    model.save(os.path.join(model_dir, "face_recognition_model.h5"))  # Menyimpan model
    np.save(os.path.join(model_dir, "classes.npy"), label_encoder.classes_)  # Menyimpan kelas
    st.success("Model berhasil dilatih dan disimpan.")

    # Plot grafik akurasi dan loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Akurasi
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    st.pyplot(fig)

    # Evaluasi model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Menampilkan laporan klasifikasi
    report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred_classes)

    # Menampilkan laporan dan confusion matrix
    st.text("Classification Report:")
    st.text(report)

    # Plot confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)
    

# Fungsi mengenali wajah dan mencatat absensi
def recognize_face(model, threshold=0.7):  # Meningkatkan threshold ke 0.7
    # Memuat kelas dari file .npy
    classes = np.load(os.path.join(model_dir, "classes.npy"), allow_pickle=True)

    # Membangun kembali objek LabelEncoder dan fit dengan kelas yang dimuat
    label_encoder = LabelEncoder()
    label_encoder.classes_ = classes

    # Menyiapkan cascade classifier untuk deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)  # Mengakses kamera
    if not cap.isOpened():  # Cek apakah kamera tersedia
        st.error("Kamera tidak dapat diakses.")
        return
    
    # Streamlit untuk update gambar
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()  # Membaca frame dari kamera
        if not ret:
            st.error("Gagal membaca dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah gambar ke grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))  # Ukuran minimal wajah

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0  # Normalisasi dan resize
            face = np.expand_dims(resized, axis=0)

            prediction = model.predict(face)  # Prediksi dengan model
            pred_label = np.argmax(prediction)
            accuracy = np.max(prediction)

            if accuracy >= threshold:  # Hanya jika confidence di atas threshold
                name = label_encoder.inverse_transform([pred_label])[0]
                log_attendance(name, accuracy * 100)  # Mencatat absensi dengan confidence
            else:
                name = "Unknown"  # Jika confidence di bawah threshold atau tidak dikenali

            # Menampilkan hasil prediksi pada frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} - {accuracy*100:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Menampilkan hasil deteksi di Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()  # Melepaskan kamera
    cv2.destroyAllWindows()  # Menutup jendela OpenCV


# Fungsi untuk mencatat kehadiran
def log_attendance(name, confidence):
    if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
        df = pd.read_csv(attendance_file)  # Membaca CSV jika ada
    else:
        df = pd.DataFrame(columns=["Name", "Tanggal", "Waktu", "Kemiripan"])  # Membuat DataFrame baru
    
    # Menambahkan atau memperbarui data kehadiran
    time_now = datetime.now()
    current_time = time_now.strftime("%H:%M:%S")
    current_date = time_now.strftime("%Y-%m-%d")
    if name in df["Name"].values:
        df.loc[df["Name"] == name, ["Tanggal", "Waktu", "Kemiripan"]] = [current_date, current_time, confidence]
    else:
        new_entry = pd.DataFrame([{"Name": name, "Tanggal": current_date, "Waktu": current_time, "Kemiripan": confidence}])
        df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)  # Simpan kembali ke CSV

# Streamlit UI
st.title("Aplikasi Pengenalan Wajah Menggunakan Streamlit Untuk Absensi")
activity = st.sidebar.selectbox("Pilih Aktivitas", [
    "Pengumpulan Gambar",
    "Pelatihan Model",
    "Pengujian Model"
])

# Pengumpulan gambar
if activity == "Pengumpulan Gambar":
    name = st.text_input("Masukkan Nama Anda")
    if st.button("Mulai Pengambilan Gambar"):
        if name:
            capture_images(name, 100)
        else:
            st.warning("Nama harus diisi!")

# Pelatihan model
elif activity == "Pelatihan Model":
    if st.button("Mulai Pelatihan Model"):
        train_model()

# Pengujian model dan pencatatan kehadiran
elif activity == "Pengujian Model":
    model = load_model(os.path.join(model_dir, "face_recognition_model.h5"))  # Memuat model
    if st.button("Mulai Pengujian Wajah"):
        recognize_face(model)