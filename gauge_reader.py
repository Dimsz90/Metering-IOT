import cv2, numpy as np, math, time
from collections import deque
from datetime import datetime
import gspread

# === Konfigurasi CCTV ===
RTSP_URL = "rtsp://admin:P4rama6887@kampuang.ddns.net:58554/Streaming/Channels/101"
CAMERA_NAME = "CCTV_Panel1"

# === Google Sheets ===
SERVICE_ACCOUNT_JSON = "service_account.json"
SPREADSHEET_ID = "18jbl4wal1aUkw5i91UXzBsKYZuxBfBUXuPkUsQuN1Vk"
WORKSHEET_NAME = "Sheet1"

gc = gspread.service_account(filename=SERVICE_ACCOUNT_JSON)
sh = gc.open_by_key(SPREADSHEET_ID)
try:
    ws = sh.worksheet(WORKSHEET_NAME)
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=1000, cols=10)
    ws.append_row(["Timestamp", "Camera", "Mode", "PSI", "Bar"])

def append_row(camera, mode, psi, bar):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ws.append_row([ts, camera, mode, float(f"{psi:.1f}"), float(f"{bar:.2f}")],
                  value_input_option="USER_ENTERED")

# === Parameter Gauge SACHIO (350 PSI / 25 bar) ===
PSI_MAX = 350
BAR_MAX = 25
ZERO_OFFSET = 225  # Bisa diatur lagi
SWEEP_DEG = 270

# === ROI Manual (untuk crop monitor dari CCTV) ===
# Set None untuk auto-detect, atau set manual: (x, y, w, h)
MONITOR_ROI = None  # Contoh: (200, 100, 800, 600)

angle_hist = deque(maxlen=15)
psi_hist = deque(maxlen=10)
stable_angle = None
locked_circle = None
monitor_roi = MONITOR_ROI

# === RTSP Connection ===
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("❌ Gagal buka stream RTSP")
    exit()

print("🎥 Stream aktif. Tekan 'q'=keluar | 'r'=reset ROI | 's'=set ROI manual")

last_sent = 0
SEND_INTERVAL = 10

def enhance_gauge_image(img):
    """Perbaiki kualitas gambar untuk deteksi lebih baik"""
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Contrast
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l,a,b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def detect_monitor_area(frame):
    """Auto-detect area monitor putih di frame CCTV"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold untuk area terang (monitor)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Morphology untuk clean noise
    kernel = np.ones((15,15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find largest contour (monitor)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Filter: area harus cukup besar (monitor)
        if area > frame.shape[0] * frame.shape[1] * 0.1:  # min 10% frame
            x, y, w, h = cv2.boundingRect(largest)
            return (x, y, w, h)
    
    return None

def detect_angle(frame):
    global locked_circle
    
    # Enhance dulu sebelum proses
    frame = enhance_gauge_image(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)  # Lebih blur untuk noise reduction
    
    if locked_circle is None:
        # Deteksi lingkaran dengan parameter lebih toleran
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, 1, 100,
            param1=80,    # Turunkan untuk deteksi lebih sensitif
            param2=25,    # Turunkan threshold
            minRadius=60, # Lebih kecil
            maxRadius=350
        )
        
        if circles is not None:
            # Pilih lingkaran paling bulat (circularity check)
            best_circle = None
            best_score = 0
            
            for circle in circles[0]:
                cx, cy, r = map(int, circle)
                # Score berdasarkan posisi (tengah frame lebih baik)
                h, w = gray.shape
                center_dist = math.hypot(cx - w/2, cy - h/2)
                score = r / (center_dist + 1)  # Lebih besar & tengah = score tinggi
                
                if score > best_score:
                    best_score = score
                    best_circle = (cx, cy, r)
            
            if best_circle:
                locked_circle = best_circle
                print(f"🔒 Lingkaran locked: {locked_circle}, score: {best_score:.2f}")
    else:
        cx, cy, r = locked_circle
    
    if locked_circle is None:
        return frame, None
    
    h, w = gray.shape
    x1, x2 = max(0, cx - r), min(w, cx + r)
    y1, y2 = max(0, cy - r), min(h, cy + r)
    
    # Draw circle untuk debugging
    cv2.circle(frame, (cx, cy), r, (0,255,0), 2)
    cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
    
    roi_g = gray[y1:y2, x1:x2]
    roi_c = frame[y1:y2, x1:x2]
    if roi_g.size == 0:
        return frame, None
    
    # Mask circular area
    center = (roi_g.shape[1]//2, roi_g.shape[0]//2)
    mask = np.zeros_like(roi_g)
    cv2.circle(mask, center, int(min(roi_g.shape)//2 * 0.80), 255, -1)
    roi_masked = cv2.bitwise_and(roi_g, mask)
    
    # Adaptive threshold lebih agresif
    thr = cv2.adaptiveThreshold(roi_masked, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 45, 10)
    
    # Morphology untuk perjelas jarum
    k = np.ones((2,2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k)
    thr = cv2.dilate(thr, k, iterations=2)
    
    # Edge detection
    edges = cv2.Canny(thr, 30, 150)
    
    # Line detection dengan threshold lebih rendah
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                            threshold=40,        # Turunkan
                            minLineLength=r//3,  # Lebih pendek ok
                            maxLineGap=20)
    
    if lines is None:
        return frame, None
    
    # Filter: line harus lewat dekat center
    valid_lines = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        
        # Cek jarak line ke center
        dist_to_center = cv2.pointPolygonTest(
            np.array([[[center[0], center[1]]]]), 
            ((x1+x2)//2, (y1+y2)//2), 
            True
        )
        
        if abs(dist_to_center) < r * 0.4:  # Line dekat center
            length = math.hypot(x2-x1, y2-y1)
            valid_lines.append((ln[0], length))
    
    if not valid_lines:
        return frame, None
    
    # Pilih line terpanjang
    best_line, _ = max(valid_lines, key=lambda x: x[1])
    x1, y1, x2, y2 = best_line
    
    # Draw jarum detection
    cv2.line(roi_c, (x1,y1), (x2,y2), (0,0,255), 3)
    
    # Hitung angle
    dx, dy = x2 - x1, y1 - y2
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0: 
        ang += 360
    
    return frame, ang


frame_count = 0
while True:
    # Buang buffer untuk realtime
    for _ in range(3): 
        cap.grab()
    
    ok, frame = cap.read()
    if not ok:
        print("⚠️ Stream terputus, reconnect...")
        cap.release()
        time.sleep(3)
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        locked_circle = None
        monitor_roi = None
        continue
    
    frame = cv2.resize(frame, (1280, 960))  # Resolusi lebih tinggi
    frame_orig = frame.copy()
    
    # === STEP 1: Detect/Crop Monitor Area ===
    if monitor_roi is None and frame_count % 30 == 0:  # Coba detect tiap 1 detik
        detected_roi = detect_monitor_area(frame)
        if detected_roi:
            monitor_roi = detected_roi
            print(f"📺 Monitor area detected: {monitor_roi}")
    
    # Crop ke area monitor jika sudah detected
    if monitor_roi:
        x, y, w, h = monitor_roi
        cv2.rectangle(frame_orig, (x, y), (x+w, y+h), (255,0,255), 2)
        cv2.putText(frame_orig, "MONITOR ROI", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
        frame_work = frame[y:y+h, x:x+w].copy()
    else:
        frame_work = frame.copy()
        cv2.putText(frame_orig, "Detecting monitor area...", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
    
    # === STEP 2: Detect Gauge ===
    frame_work, ang = detect_angle(frame_work)
    
    if ang is not None:
        angle_hist.append(ang)
        avg_ang = sum(angle_hist) / len(angle_hist)
        
        if stable_angle is None:
            stable_angle = avg_ang
        
        # Smoothing dengan threshold
        if abs(avg_ang - stable_angle) > 3:  # Lebih sensitif
            stable_angle = 0.80 * stable_angle + 0.20 * avg_ang
        
        ang_use = stable_angle
        
        # Hitung PSI/Bar untuk 350 PSI gauge
        adj = (ZERO_OFFSET - ang_use) % 360
        if adj > SWEEP_DEG: 
            adj -= 360
        
        psi = (adj / SWEEP_DEG) * PSI_MAX
        bar = (adj / SWEEP_DEG) * BAR_MAX
        
        psi = max(0, min(PSI_MAX, psi))
        bar = max(0, min(BAR_MAX, bar))
        
        psi_hist.append(psi)
        avg_psi = round(sum(psi_hist) / len(psi_hist), 1)
        avg_bar = round(avg_psi / 14.5038, 2)
        
        # Overlay hasil
        cv2.putText(frame_orig, f"{avg_psi} PSI", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.putText(frame_orig, f"{avg_bar} bar", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        cv2.putText(frame_orig, f"Angle: {ang_use:.1f}°", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Kirim ke Sheets
        now = time.time()
        if now - last_sent >= SEND_INTERVAL:
            try:
                append_row(CAMERA_NAME, "AUTO", avg_psi, avg_bar)
                print(f"✅ {datetime.now()} | {avg_psi} PSI, {avg_bar} bar → Sheets")
            except Exception as e:
                print(f"❌ Sheets error: {e}")
            last_sent = now
    else:
        cv2.putText(frame_orig, "Detecting gauge needle...", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
    
    # Show processed area (kecil di pojok)
    if monitor_roi and frame_work.shape[0] > 0:
        small = cv2.resize(frame_work, (320, 240))
        frame_orig[10:250, frame_orig.shape[1]-330:frame_orig.shape[1]-10] = small
    
    cv2.imshow("CCTV Gauge Reader - Optimized", frame_orig)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset detection
        locked_circle = None
        monitor_roi = None
        print("🔄 Detection reset")
    elif key == ord('s'):  # Set ROI manual (klik drag)
        print("⚠️ Paused. Gunakan mouse untuk drag area monitor, lalu tekan SPACE")
        monitor_roi = cv2.selectROI("Select Monitor Area", frame_orig, False)
        cv2.destroyWindow("Select Monitor Area")
        print(f"📺 Manual ROI set: {monitor_roi}")
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()