# Gas Meter Reader System

Sistem otomatis untuk membaca meteran gas menggunakan ESP32-CAM dan OCR (Optical Character Recognition).

##  Fitur

-  **Auto Capture**: ESP32-CAM menangkap foto meteran secara otomatis
-  **OCR Engine**: Multi-method voting OCR untuk akurasi maksimal
-  **Web Dashboard**: Dashboard real-time dengan grafik dan history
-  **Secure**: API key authentication & PIN login
-  **MySQL Database**: Penyimpanan data terstruktur
-  **Responsive UI**: Antarmuka modern dengan dark mode

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MySQL Server
- Tesseract OCR
- ESP32-CAM Module

### Installation

1. Clone repository:
```bash
git clone https://github.com/Dimsz90/Metering-IOT
cd Metering-IOT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup environment variables:
```bash
cp .env.example .env
# Edit .env dengan konfigurasi Anda
```

4. Create database:
```sql
CREATE DATABASE meter_readings;
```

5. Run the server:
```bash
python flask_server.py
```

Server akan berjalan di `http://localhost:5000`

## 📋 Setup Device Baru

1. Jalankan ROI setup untuk konfigurasi area pembacaan:
```bash
python setup_roi.py
```

2. Upload ESP32 code (`esp32_cam_client.ino`) ke ESP32-CAM
3. Konfigurasi WiFi dan server IP di code ESP32
4. Device akan otomatis muncul di dashboard

## 🔧 Configuration

Edit file `.env`:

```env
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=localhost
DB_NAME=meter_readings

API_KEY=your_secret_api_key
ADMIN_PIN=your_pin
FLASK_SECRET_KEY=your_flask_secret
```

## 🛡️ Security

- API endpoint dilindungi dengan API Key
- Dashboard memerlukan PIN login
- File `.env` tidak di-commit ke repository


## 📝 License

MIT License
