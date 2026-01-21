from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import json
import os
import glob
from datetime import datetime, timedelta
import easyocr
from PIL import Image
import pytesseract
from collections import Counter
import mysql.connector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ============ SETUP TESSDATA DIRECTORY ============
TESSDATA_DIR = os.path.join(os.path.dirname(__file__), 'tessdata')
if os.path.exists(TESSDATA_DIR):
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    print(f"✅ Custom tessdata directory: {TESSDATA_DIR}")
    
    # Check if custom model exists
    custom_model = os.path.join(TESSDATA_DIR, 'gas_meter.traineddata')
    if os.path.exists(custom_model):
        print(f"✅ Custom model found: gas_meter.traineddata")
    else:
        print(f"⚠️  Custom model NOT found, using default")
else:
    print(f"⚠️  tessdata directory not found")

# ============ CONFIGURE TESSERACT PATH ============
possible_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'/usr/bin/tesseract',
    r'/usr/local/bin/tesseract'
]

for path in possible_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"✅ Tesseract found at: {path}")
        break

# Initialize OCR engines
easyocr_reader = easyocr.Reader(['en'], gpu=False)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
DEBUG_FOLDER = 'debug'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# Database Helpers
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'meter_readings')
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def save_result_to_db(device_id, reading, confidence, image_filename, timestamp_dt):
    """Save result to MySQL database"""
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cursor = conn.cursor()
        
        # 1. Update/Insert Device Status
        cursor.execute("""
            INSERT INTO devices (device_id, last_seen, last_reading, confidence, is_active)
            VALUES (%s, %s, %s, %s, TRUE)
            ON DUPLICATE KEY UPDATE
            last_seen = %s,
            last_reading = %s,
            confidence = %s,
            is_active = TRUE
        """, (
            device_id, timestamp_dt, reading, confidence,
            timestamp_dt, reading, confidence
        ))
        
        # 2. Insert Reading Log
        cursor.execute("""
            INSERT INTO readings (device_id, reading, confidence, image_filename, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (device_id, reading, confidence, image_filename, timestamp_dt))
        
        conn.commit()
        print(f"✅ Saved to Database: {device_id} -> {reading}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        if conn: conn.close()
        return False

class AdvancedOCR:
    """Advanced OCR with multiple preprocessing strategies"""
    
    def __init__(self):
        self.easyocr_reader = easyocr_reader
        
    def preprocess_method_1_adaptive(self, roi_image):
        """Method 1: Adaptive thresholding with bilateral filter"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    def preprocess_method_2_otsu(self, roi_image):
        """Method 2: Otsu's thresholding"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def preprocess_method_3_morphology(self, roi_image):
        """Method 3: Morphological operations"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Binary threshold
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Closing to connect components
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)
        
        return binary
    
    def preprocess_method_4_clahe(self, roi_image):
        """Method 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def preprocess_method_5_sauvola(self, roi_image):
        """Method 5: Sauvola local thresholding (approximation)"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Approximation of Sauvola using local statistics
        window_size = 15
        k = 0.5
        R = 128
        
        # Calculate local mean and std
        mean = cv2.blur(gray.astype(np.float32), (window_size, window_size))
        sqmean = cv2.blur((gray.astype(np.float32))**2, (window_size, window_size))
        std = np.sqrt(sqmean - mean**2)
        
        # Sauvola threshold
        threshold = mean * (1 + k * ((std / R) - 1))
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        
        return binary
    
    def preprocess_method_6_enhanced(self, roi_image):
        """Method 6: Enhanced contrast with sharpening"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        alpha = 1.5  # Contrast control
        beta = 0     # Brightness control
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
        
        # Threshold
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def upscale_image(self, image, scale=3):
        """Upscale image for better OCR"""
        height, width = image.shape[:2]
        return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    
    def recognize_with_easyocr(self, image):
        """Recognize digit using EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(
                image,
                allowlist='0123456789',
                detail=1,
                paragraph=False,
                min_size=10
            )
            
            if results and len(results) > 0:
                # Get the result with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1]
                confidence = best_result[2]
                
                # Extract first digit
                digit = ''.join(filter(str.isdigit, text))
                if digit:
                    return digit[0], confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return None, 0.0
    
    def recognize_with_tesseract(self, image, use_custom_model=True):
        """Recognize digit using Tesseract (with custom gas_meter model)"""
        try:
            # Pilih model yang digunakan
            if use_custom_model:
                # Gunakan model custom gas_meter
                # OEM 1 = LSTM only, lebih stabil untuk custom model yang hanya punya LSTM
                config = '--psm 10 --oem 1 -l gas_meter -c tessedit_char_whitelist=0123456789'
            else:
                # Gunakan model default
                config = '--psm 10 --oem 1 -l eng -c tessedit_char_whitelist=0123456789'
            
            # Convert to PIL Image
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # OCR
            text = pytesseract.image_to_string(pil_image, config=config).strip()
            
            # Get confidence
            data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            confidence = max(confidences) / 100.0 if confidences else 0.0
            
            # Extract digit
            digit = ''.join(filter(str.isdigit, text))
            if digit:
                return digit[0], confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"Tesseract error: {e}")
            # Jika custom model error, coba fallback ke default
            if use_custom_model:
                print("⚠️  Custom model failed, trying default model...")
                return self.recognize_with_tesseract(image, use_custom_model=False)
            return None, 0.0
    
    def recognize_digit(self, roi_image, digit_idx, save_debug=False):
        """
        Recognize single digit using ensemble of methods
        """
        # Upscale for better recognition
        upscaled = self.upscale_image(roi_image, scale=3)
        
        # Apply all preprocessing methods
        preprocessed_images = {
            'adaptive': self.preprocess_method_1_adaptive(upscaled),
            'otsu': self.preprocess_method_2_otsu(upscaled),
            'morphology': self.preprocess_method_3_morphology(upscaled),
            'clahe': self.preprocess_method_4_clahe(upscaled),
            'sauvola': self.preprocess_method_5_sauvola(upscaled),
            'enhanced': self.preprocess_method_6_enhanced(upscaled)
        }
        
        # Save debug images if requested
        if save_debug:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for method_name, img in preprocessed_images.items():
                debug_path = os.path.join(DEBUG_FOLDER, f"digit_{digit_idx}_{method_name}_{timestamp}.jpg")
                cv2.imwrite(debug_path, img)
        
        # Collect all predictions
        all_predictions = []
        
        for method_name, processed_img in preprocessed_images.items():
            # Try EasyOCR
            digit_easy, conf_easy = self.recognize_with_easyocr(processed_img)
            if digit_easy and conf_easy > 0.2:
                all_predictions.append({
                    'digit': digit_easy,
                    'confidence': conf_easy,
                    'method': f'EasyOCR_{method_name}'
                })
            
            # Try Tesseract (with custom model)
            digit_tess, conf_tess = self.recognize_with_tesseract(processed_img, use_custom_model=True)
            if digit_tess and conf_tess > 0.2:
                all_predictions.append({
                    'digit': digit_tess,
                    'confidence': conf_tess,
                    'method': f'Tesseract_GasMeter_{method_name}'
                })
        
        if not all_predictions:
            # Fallback: Check for vertical line (Digit 1)
            check_img = preprocessed_images.get('enhanced', None)
            if check_img is not None:
                h, w = check_img.shape
                vertical_proj = np.sum(check_img == 255, axis=0)
                center_region = vertical_proj[int(w*0.3):int(w*0.7)]
                if len(center_region) > 0:
                    max_val = np.max(center_region)
                    if max_val > h * 0.5:
                        return '1', 0.6, [{'digit': '1', 'confidence': 0.6, 'method': 'Geometric_Fallback'}]

            return '?', 0.0, []
        
        # Voting mechanism - Weighted System
        vote_scores = {}
        
        for p in all_predictions:
            digit = p['digit']
            confidence = p['confidence']
            method = p['method']
            
            # Base weight
            weight = 1.0
            
            # 1. Trust custom Tesseract model MORE
            if 'Tesseract_GasMeter' in method:
                weight += 1.0  # Boost custom model karena sudah ditraining khusus
            elif 'Tesseract' in method:
                weight += 0.5
            
            # 2. Reward high confidence
            if confidence > 0.90:
                weight += 0.5
            elif confidence > 0.70:
                weight += 0.2
                
            # 3. Specific Heuristic for 3 vs 5 confusion
            if digit == '3':
                weight += 0.3
                
            vote_scores[digit] = vote_scores.get(digit, 0) + weight
            
        if not vote_scores:
            return '?', 0.0, []

        # Get winner (highest score)
        most_common_digit = max(vote_scores, key=vote_scores.get)
        
        # Calculate average confidence for the winning digit
        winning_predictions = [p for p in all_predictions if p['digit'] == most_common_digit]
        avg_confidence = sum(p['confidence'] for p in winning_predictions) / len(winning_predictions)
        
        # --- AUTO HARVEST TRAINING DATA ---
        try:
            best_pred = max(winning_predictions, key=lambda x: x['confidence'])
            best_method_full = best_pred['method']
            
            if 'EasyOCR_' in best_method_full:
                method_key = best_method_full.replace('EasyOCR_', '')
            elif 'Tesseract_GasMeter_' in best_method_full:
                method_key = best_method_full.replace('Tesseract_GasMeter_', '')
            elif 'Tesseract_' in best_method_full:
                method_key = best_method_full.replace('Tesseract_', '')
            else:
                method_key = None
            
            if method_key and method_key in preprocessed_images:
                train_dir = os.path.join('training_data', most_common_digit)
                os.makedirs(train_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"{timestamp}_{method_key}.jpg"
                filepath = os.path.join(train_dir, filename)
                
                cv2.imwrite(filepath, preprocessed_images[method_key])
        except Exception as e:
            print(f"Error saving training data: {e}")
        # ----------------------------------
        
        return most_common_digit, avg_confidence, all_predictions

# Rest of the code remains the same...
def extract_digits_from_image(image, roi_config, save_debug=False):
    """Extract and recognize digits from image using ROI config"""
    ocr_engine = AdvancedOCR()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_digits': roi_config.get('numDigits', 0),
        'digits': [],
        'reading': '',
        'confidence': [],
        'processing_details': []
    }
    
    rois = roi_config.get('rois', [])
    
    for i, roi in enumerate(rois):
        if not roi or 'x' not in roi:
            continue
        
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Validate ROI bounds
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            print(f"Invalid ROI bounds for digit {i+1}")
            continue
        
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        roi_image = image[y1:y2, x1:x2]
        
        if roi_image.size == 0:
            continue
        
        # Recognize digit
        digit, confidence, all_predictions = ocr_engine.recognize_digit(
            roi_image, i + 1, save_debug=save_debug
        )
        
        results['digits'].append({
            'position': i + 1,
            'value': digit,
            'confidence': round(confidence * 100, 2),
            'roi': {'x': x, 'y': y, 'w': w, 'h': h},
            'votes': len([p for p in all_predictions if p['digit'] == digit]),
            'total_predictions': len(all_predictions)
        })
        
        results['processing_details'].append({
            'position': i + 1,
            'all_predictions': all_predictions
        })
        
        results['reading'] += digit
        results['confidence'].append(round(confidence * 100, 2))
    
    # Calculate statistics
    if results['confidence']:
        results['avg_confidence'] = round(sum(results['confidence']) / len(results['confidence']), 2)
        results['min_confidence'] = round(min(results['confidence']), 2)
        results['max_confidence'] = round(max(results['confidence']), 2)
    else:
        results['avg_confidence'] = 0.0
        results['min_confidence'] = 0.0
        results['max_confidence'] = 0.0
    
    # Quality assessment
    if results['avg_confidence'] >= 80:
        results['quality'] = 'EXCELLENT'
    elif results['avg_confidence'] >= 60:
        results['quality'] = 'GOOD'
    elif results['avg_confidence'] >= 40:
        results['quality'] = 'FAIR'
    else:
        results['quality'] = 'POOR'
    
    return results

def draw_results_on_image(image, results):
    """Draw ROI boxes and detected digits on image"""
    output = image.copy()
    
    colors = [
        (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255),
        (0, 128, 255), (255, 128, 0), (128, 255, 0), (255, 0, 128),
        (0, 255, 128), (128, 0, 255)
    ]
    
    for digit_info in results['digits']:
        roi = digit_info['roi']
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        pos = digit_info['position'] - 1
        color = colors[pos % len(colors)]
        
        # Determine box thickness based on confidence
        confidence = digit_info['confidence']
        if confidence >= 80:
            thickness = 4
        elif confidence >= 60:
            thickness = 3
        else:
            thickness = 2
        
        # Draw box
        cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)
        
        # Draw digit, confidence, and votes
        label = f"{digit_info['value']} ({digit_info['confidence']:.1f}%) [{digit_info['votes']}/{digit_info['total_predictions']}]"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(output, (x, y-text_h-10), (x+text_w+10, y), color, -1)
        cv2.putText(output, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
    # Draw final reading at top
    reading_text = f"Reading: {results['reading']}"
    stats_text = f"Avg: {results['avg_confidence']:.1f}% | Quality: {results['quality']}"
    
    cv2.rectangle(output, (10, 10), (700, 80), (0, 0, 0), -1)
    cv2.putText(output, reading_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(output, stats_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return output

@app.route('/upload/<camera_name>', methods=['POST'])
def upload_image(camera_name):
    """Handle image upload with ROI config in header"""
    print(f"\n--- INCOMING REQUEST: {camera_name} ---", flush=True)
    
    # Check API key
    api_key = request.headers.get('X-API-Key')
    if api_key != 'example':
        print(f"❌ Unauthorized access attempt", flush=True)
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get ROI config
    roi_config = None
    
    # 1. Try header
    roi_config_str = request.headers.get('X-ROI-Config')
    if roi_config_str:
        try:
            roi_config = json.loads(roi_config_str)
            print(f"Loaded ROI config from header", flush=True)
        except:
            print(f"Invalid ROI config in header", flush=True)
            
    # 2. Try server-side file if header failed
    if not roi_config:
        config_path = os.path.join('configs', f'{camera_name}.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    roi_config = json.load(f)
                print(f"Loaded ROI config from server file: {config_path}", flush=True)
            except Exception as e:
                print(f"Error loading server config: {e}", flush=True)
    
    if not roi_config:
        print("❌ Error: No ROI configuration found", flush=True)
        print("--- END REQUEST (ERROR) ---\n", flush=True)
        return jsonify({'error': 'No ROI configuration provided (Header or Server File)'}), 400
    
    # Debug: Print request files
    try:
        print(f"Accessing request.files...", flush=True)
        files_keys = list(request.files.keys())
        print(f"Files in written request: {files_keys}", flush=True)
    except Exception as e:
        print(f"❌ Error accessing request.files: {e}", flush=True)
        return jsonify({'error': 'Internal Request Error'}), 500
    
    # Get image file
    if 'image' not in request.files:
        print("❌ Error: 'image' not in request.files", flush=True)
        print("--- END REQUEST (ERROR) ---\n", flush=True)
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['image']
    
    # Save original image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create camera-specific upload folder
    camera_upload_folder = os.path.join(UPLOAD_FOLDER, camera_name)
    os.makedirs(camera_upload_folder, exist_ok=True)
    
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(camera_upload_folder, filename)
    
    print(f"Saving image to {filepath}...", flush=True)
    try:
        file.save(filepath)
        print(f"✅ Image saved successfully", flush=True)
    except Exception as e:
        print(f"❌ Error saving file: {e}", flush=True)
        return jsonify({'error': 'File save error'}), 500
    
    # Read image
    print(f"Reading image with OpenCV...", flush=True)
    image = cv2.imread(filepath)
    
    if image is None:
        print(f"❌ Error: Failed to read image using cv2.imread({filepath})", flush=True)
        print("--- END REQUEST (ERROR) ---\n", flush=True)
        return jsonify({'error': 'Failed to read image'}), 500
    
    print(f"✅ Image loaded into memory. Size: {image.shape}", flush=True)
    
    # Process with advanced OCR (enable debug for first few images)
    save_debug = request.args.get('debug', 'false').lower() == 'true'
    print("Starting OCR processing...", flush=True)
    
    try:
        results = extract_digits_from_image(image, roi_config, save_debug=save_debug)
        print("✅ OCR Processing complete", flush=True)
    except Exception as e:
        print(f"❌ Error within OCR engine: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'OCR Engine Error'}), 500
    
    # Draw results on image
    print("Drawing results...", flush=True)
    result_image = draw_results_on_image(image, results)
    
    # Save result image
    # Create camera-specific result folder
    camera_result_folder = os.path.join(RESULTS_FOLDER, camera_name)
    os.makedirs(camera_result_folder, exist_ok=True)
    
    result_filename = f"{timestamp}_result.jpg"
    result_filepath = os.path.join(camera_result_folder, result_filename)
    cv2.imwrite(result_filepath, result_image)
    print(f"✅ Result image saved to {result_filepath}", flush=True)
    
    # Save JSON result
    json_filename = f"{timestamp}_result.json"
    json_filepath = os.path.join(camera_result_folder, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ JSON result saved to {json_filepath}", flush=True)
    
    print(f"\n{'='*70}", flush=True)
    print(f"📸 Camera: {camera_name}", flush=True)
    print(f"🕒 Time: {timestamp}", flush=True)
    print(f"🔢 Reading: {results['reading']}", flush=True)
    print(f"📊 Confidence: {results['avg_confidence']:.2f}% (Min: {results['min_confidence']:.1f}%, Max: {results['max_confidence']:.1f}%)", flush=True)
    print(f"⭐ Quality: {results['quality']}", flush=True)
    print(f"💾 Saved: {result_filename}", flush=True)
    
    # Print individual digits
    for digit in results['digits']:
        print(f"   D{digit['position']}: {digit['value']} ({digit['confidence']:.1f}%) - {digit['votes']}/{digit['total_predictions']} votes", flush=True)
    
    print(f"{'='*70}\n", flush=True)
    
    # Save to Database
    print("Saving to database...", flush=True)
    reading_val = results['reading']
    # Format reading if needed (e.g. 0011439 -> 00114.39)
    if reading_val.isdigit() and len(reading_val) > 2:
        formatted_reading = f"{reading_val[:-2]}.{reading_val[-2:]}"
    else:
        formatted_reading = reading_val

    save_result_to_db(
        device_id=camera_name,
        reading=formatted_reading,
        confidence=results['avg_confidence'],
        image_filename=result_filename,
        timestamp_dt=datetime.fromisoformat(results['timestamp'])
    )
    
    print("--- END REQUEST (SUCCESS) ---\n", flush=True)
    
    return jsonify({
        'status': 'success',
        'reading': results['reading'],
        'confidence': results['avg_confidence'],
        'result_image': f"/uploads/{camera_name}/{result_filename}"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Advanced ESP32-CAM OCR Server',
        'timestamp': datetime.now().isoformat(),
        'ocr_engines': ['EasyOCR', 'Tesseract'],
        'preprocessing_methods': 6
    })

@app.template_filter('format_reading')
def format_reading(value):
    """Format reading with units (last 2 digits are decimal)"""
    try:
        if not value: return "--"
        s_val = str(value).strip()
        
        # If string is just digits (e.g. 0011439)
        if s_val.isdigit():
            if len(s_val) > 2:
                # Insert decimal before last 2 digits
                return f"{s_val[:-2]}.{s_val[-2:]} m³"
            elif len(s_val) == 2:
                return f"0.{s_val} m³"
            else:
                return f"0.0{s_val} m³"
                
        # If already has decimal or other chars
        return f"{s_val} m³"
    except:
        return f"{value} m³"

@app.route('/')
def dashboard():
    """Main dashboard showing all devices (DB Version)"""
    devices = []
    
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM devices")
            rows = cursor.fetchall()
            
            for row in rows:
                # Calculate if offline (> 1 hour)
                last_seen_dt = row['last_seen']
                if not last_seen_dt:
                    is_late = True
                    display_time = "Never"
                else:
                    is_late = (datetime.now() - last_seen_dt) > timedelta(hours=1)
                    display_time = last_seen_dt.strftime('%H:%M %d/%m')

                # Format reading
                reading = row['last_reading'] if row['last_reading'] else '--'
                
                # Format confidence
                conf = row['confidence'] / 100.0 if row['confidence'] else 0.0

                devices.append({
                    'id': row['device_id'],
                    'reading': reading,
                    'conf': conf,
                    'temp': row.get('temperature', None),
                    'last_seen': display_time,
                    'is_late': is_late
                })
            
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"❌ DB Error in dashboard: {e}")
            if conn: conn.close()
    
    return render_template('index.html', devices=devices)

@app.route('/report/<device_id>')
def device_report(device_id):
    """Detailed report for a device (DB Version)"""
    # Get date range from query params
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    s_param = request.args.get('start')
    e_param = request.args.get('end')
    
    if s_param: start_date = datetime.strptime(s_param, '%Y-%m-%d')
    if e_param: end_date = datetime.strptime(e_param, '%Y-%m-%d')
    
    # Adjust range for query (inclusive)
    query_start = start_date.replace(hour=0, minute=0, second=0)
    query_end = end_date.replace(hour=23, minute=59, second=59)

    logs = []
    chart_data = []
    latest_val = "0"
    
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Fetch logs
            cursor.execute("""
                SELECT * FROM readings 
                WHERE device_id = %s 
                AND timestamp BETWEEN %s AND %s 
                ORDER BY timestamp DESC
            """, (device_id, query_start, query_end))
            
            rows = cursor.fetchall()
            
            for row in rows:
                entry_dt = row['timestamp']
                reading = row['reading']
                
                log_entry = {
                    'timestamp': entry_dt,
                    'reading': reading,
                    'confidence': (row['confidence'] or 0) / 100.0,
                    'device_id': device_id,
                    'image_filename': row['image_filename']
                }
                logs.append(log_entry)
                
                # Chart data (reverse order for chart later)
                try:
                    # Clean reading string to float
                    clean_reading = reading.replace(' m³', '').strip()
                    val = float(clean_reading)
                    chart_data.append({
                        'label': entry_dt.strftime('%d/%m %H:%M'),
                        'value': val
                    })
                except:
                    pass
            
            if logs:
                latest_val = logs[0]['reading']
                
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ DB Error in report: {e}")
            if conn: conn.close()
            
    return render_template('report.html', 
                           device=device_id,
                           stats={'lbl': 'Latest Reading', 'val': latest_val},
                           logs=logs,
                           data=chart_data[::-1], # Chronological for chart
                           type='line',
                           start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'))

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve uploaded images"""
    # Check if file is in RESULTS_FOLDER first (for result images)
    if os.path.exists(os.path.join(RESULTS_FOLDER, filename)):
         return send_from_directory(RESULTS_FOLDER, filename)
    
    # Fallback to general uploads (might be in subfolders or direct)
    # Since logical structure isn't perfect, we try both folders
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ADVANCED ESP32-CAM OCR PROCESSING SERVER")
    print("="*70)
    print("Features:")
    print("  ✓ 6 Preprocessing Methods")
    print("  ✓ 2 OCR Engines (EasyOCR + Tesseract)")
    print("  ✓ Ensemble Voting")
    print("  ✓ 3x Image Upscaling")
    print("  ✓ Confidence Scoring")
    print("  ✓ Quality Assessment")
    print("-"*70)
    print("Server: http://0.0.0.0:5000")
    print("Endpoints:")
    print("  POST /upload/<camera_name>  - Upload with ROI")
    print("  GET  /health                - Health check")
    print("  GET  /test                  - Info page")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
