# 🛣️ Road Pothole Detection System

Sistem deteksi lubang jalan berbasis **YOLOv8** dengan antarmuka web modern. Dapat mengakses kamera HP/laptop langsung dari browser.

## 🎯 Demo Live

**URL:** [https://pothole-detection-xxxx.onrender.com](https://pothole-detection-xxxx.onrender.com)
*(Link akan aktif setelah deploy)*

## ✨ Fitur

- 📷 **Real-time Camera** - Akses kamera HP/laptop dari browser
- 🤖 **YOLOv8 Model** - Deteksi lubang jalan otomatis
- 📊 **Live Statistics** - Jumlah deteksi, total, durasi sesi
- 📄 **PDF Report** - Generate laporan lengkap dengan gambar
- 📱 **Responsive** - Optimal di HP dan desktop

## 🚀 Deploy ke Render

### Langkah 1: Fork/Push ke GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/pothole-detection.git
git push -u origin main
```

### Langkah 2: Deploy di Render
1. Buka [render.com](https://render.com) → Sign Up (gratis)
2. Klik **New** → **Web Service**
3. Connect GitHub repository
4. Konfigurasi:
   - **Name:** pothole-detection
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Klik **Create Web Service**
6. Tunggu deploy selesai (5-10 menit)

### Langkah 3: Akses Aplikasi
Setelah deploy, akses via: `https://pothole-detection-xxxx.onrender.com`

## 📱 Cara Penggunaan

1. Buka link aplikasi di browser HP
2. Pilih kamera (depan/belakang)
3. Klik **"Mulai Deteksi"**
4. Arahkan ke jalan
5. Klik **"Stop & Laporan"** untuk download PDF

## 📁 Struktur File

```
├── app.py              # Flask backend
├── best (1).pt         # YOLOv8 model
├── requirements.txt    # Dependencies
├── render.yaml         # Render config
├── templates/
│   └── index.html      # Frontend
└── reports/            # Generated reports
```

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

Buka http://localhost:5000

## 📊 Contoh Output PDF

Laporan PDF berisi:
- Informasi sesi (waktu, durasi)
- Statistik deteksi
- Screenshot lubang terdeteksi

## 🤖 Model

Model YOLOv8 dilatih untuk mendeteksi lubang jalan (pothole).

## 📝 License

MIT License
