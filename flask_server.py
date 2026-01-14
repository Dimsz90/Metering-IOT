from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import mysql.connector
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from meter_reader import read_meter_digits
except ImportError:
    print("WARNING: meter_reader module not found. OCR features will be disabled.")
    def read_meter_digits(*args, **kwargs): return "000000", [], [], []

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key')

BASE_DIR = Path(__file__).parent.absolute()
CONFIGS_DIR = BASE_DIR / "configs"
UPLOAD_FOLDER = BASE_DIR / "uploads"
ADMIN_PIN = os.getenv('ADMIN_PIN')
API_KEY = os.getenv('API_KEY')

DB_CONFIG = {
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'meter_readings')
}

CONFIGS_DIR.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)

def get_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        if err.errno == 1049:
            print("Database 'meter_readings' does not exist. Please create it.")
        raise

def init_db():
    """Initialize database tables"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id VARCHAR(50) PRIMARY KEY,
                last_reading VARCHAR(50),
                last_seen DATETIME,
                confidence FLOAT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                device_id VARCHAR(50),
                timestamp DATETIME,
                reading VARCHAR(50),
                confidence FLOAT,
                image_filename VARCHAR(255),
                INDEX (device_id),
                INDEX (timestamp)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Database initialization failed: {e}")

init_db()

def db_get_analytics(device_id, start_str, end_str):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end = datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1) 
    
    query = """
        SELECT timestamp, reading 
        FROM readings 
        WHERE device_id = %s AND timestamp >= %s AND timestamp < %s
        ORDER BY timestamp ASC
    """
    cursor.execute(query, (device_id, start, end))
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    if not rows:
        return [], 'line'
    data_points = []
    for r in rows:
        try:
            val = float(r['reading']) / 100.0
            data_points.append({'label': r['timestamp'].strftime('%Y-%m-%d %H:%M'), 'value': val, 'ts': r['timestamp']})
        except ValueError:
            continue
            
    if not data_points:
        return [], 'line'

    duration = (end - start).days
    
    if duration > 3:
        daily = {}
        for dp in data_points:
            day = dp['ts'].strftime('%Y-%m-%d')
            if day not in daily or dp['value'] > daily[day]['value']:
                daily[day] = dp
        
        chart_data = sorted(daily.values(), key=lambda x: x['ts'])
        final_data = [{'label': d['ts'].strftime('%Y-%m-%d'), 'value': d['value']} for d in chart_data]
        return final_data, 'line'
    else:
        return [{'label': d['label'], 'value': d['value']} for d in data_points], 'line'


@app.route('/')
def dashboard():
    if 'logged_in' not in session: return redirect(url_for('login'))
    
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM devices")
        db_devices = {d['device_id']: d for d in cursor.fetchall()}
        cursor.close()
        conn.close()
    except:
        db_devices = {}
        
    devices_ui = []
    for f in CONFIGS_DIR.glob('*.json'):
        if f.name == 'default.json': continue
        did = f.stem
        
        info = db_devices.get(did, {})
        last_seen = "New"
        is_late = False
        
        if info.get('last_seen'):
            delta = datetime.now() - info['last_seen']
            if delta.total_seconds() > 2100: is_late = True
            
            mins = int(delta.total_seconds() // 60)
            if mins < 60:
                last_seen = f"{mins}m ago"
            else:
                last_seen = f"{mins//60}h {mins%60}m ago"
            
        devices_ui.append({
            'id': did, 
            'reading': info.get('last_reading', 'N/A'),
            'last_seen': last_seen,
            'is_late': is_late,
            'conf': info.get('confidence', 0)
        })
    
    return render_template('index.html', devices=devices_ui)

@app.route('/report/<device_id>')
def router(device_id):
    if 'logged_in' not in session: return redirect(url_for('login'))
    
    start = request.args.get('start', (datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d'))
    end = request.args.get('end', datetime.now().strftime('%Y-%m-%d'))
    
    chart_data, chart_type = db_get_analytics(device_id, start, end)
    val = sum(d['value'] for d in chart_data) if chart_type=='bar' else (chart_data[-1]['value'] if chart_data else 0)
    lbl = "Total Usage" if chart_type=='bar' else "Current Reading"
    
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT * FROM readings 
        WHERE device_id = %s 
        ORDER BY timestamp DESC LIMIT 50
    """, (device_id,))
    logs = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('report.html', 
        device=device_id, data=chart_data, type=chart_type,
        start=start, end=end, stats={'val': val, 'lbl': lbl}, logs=logs)

@app.route('/upload/<device_id>', methods=['POST'])
def upload_reading(device_id):
    token = request.headers.get('X-API-Key') or request.form.get('token')
    print(f"DEBUG: Received token: '{token}'")
    if token != API_KEY:
        print(f"DEBUG: Token mismatch! Expected '{API_KEY}' but got '{token}'")
        return "Unauthorized", 401

    if 'image' not in request.files:
        return "No image uploaded", 400
    
    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400

    device_dir = UPLOAD_FOLDER / device_id
    device_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now()
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = device_dir / filename
    
    file.save(filepath)
    print(f"Received image for {device_id}: {filepath}")

    config_path = CONFIGS_DIR / f"{device_id}.json"
    if not config_path.exists():
        print(f"No config found for {device_id}")
        return "Config not found for device", 400
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    rois = {int(k): tuple(v) for k, v in config.get('rois', {}).items()}
    full_text, _, conf_scores, _ = read_meter_digits(str(filepath), rois)
    avg_conf = sum(conf_scores)/len(conf_scores) if conf_scores else 0
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO readings (device_id, timestamp, reading, confidence, image_filename)
        VALUES (%s, %s, %s, %s, %s)
    """, (device_id, timestamp, full_text, avg_conf, filename))
    
    cursor.execute("""
        INSERT INTO devices (device_id, last_reading, last_seen, confidence)
        VALUES (%s, %s, %s, %s) AS new
        ON DUPLICATE KEY UPDATE
            last_reading = new.last_reading,
            last_seen = new.last_seen,
            confidence = new.confidence
    """, (device_id, full_text, timestamp, avg_conf))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({
        "status": "success",
        "reading": full_text,
        "confidence": avg_conf
    })

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST' and request.form.get('pin')==ADMIN_PIN:
        session['logged_in']=True; return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.template_filter('format_reading')
def format_reading(value):
    """Format reading as 00014.39"""
    if value is None or value == 'N/A': return 'N/A'
    
    # Handle floats (e.g. from stats)
    if isinstance(value, (float, int)):
        return f"{float(value):08.2f}"
        
    s = str(value).strip()
    if not s or not s.replace('.','').isdigit(): return value
    
    if '.' in s:
        try:
            return f"{float(s):08.2f}"
        except:
            return s
            
    s = s.zfill(7)
    return f"{s[:-2]}.{s[-2:]}"

@app.route('/uploads/<d>/<f>')
def file(d,f): 
    return send_from_directory(UPLOAD_FOLDER/d, f)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)