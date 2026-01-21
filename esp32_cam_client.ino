#include "esp_camera.h"
#include "soc/rtc_cntl_reg.h"
#include "soc/soc.h"
#include <ArduinoJson.h>
#include <DNSServer.h>
#include <Preferences.h>
#include <WebServer.h>
#include <WiFi.h>


// ======================= CONFIGURATION =======================
const char *ap_ssid = "ESP32-CAM-Config";
const char *ap_password = "12345678";
const char *serverName = "192.168.1.182";
const int serverPort = 5000;
const char *serverPath = "/upload/Test_Cam";

#define TIME_TO_SLEEP 5
#define CONFIG_MODE_TIMEOUT 600000
#define uS_TO_S_FACTOR 1000000ULL
#define MAX_DIGITS 10
// =============================================================

// Pin definitions
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define LED_FLASH_GPIO 4
#define LED_BUILTIN_GPIO 33

struct CameraSettings {
  int brightness = 0;
  int contrast = 0;
  int saturation = 0;
  int sharpness = 0;
  int denoise = 0;
  int quality = 12;
  int hmirror = 0;
  int vflip = 0;
  int awb = 1;
  int awb_gain = 1;
  int aec = 1;
  int aec_value = 300;
  int ae_level = 0;
  int agc = 1;
  int gainceiling = 0;
  int rotation = 0;
};

struct ROI {
  int x;
  int y;
  int w;
  int h;
};

WiFiClient client;
WebServer server(80);
DNSServer dnsServer;
Preferences preferences;
CameraSettings camSettings;
ROI rois[MAX_DIGITS];
int numDigits = 7;
bool configMode = false;
bool apMode = false;
unsigned long configModeStart = 0;

String saved_ssid = "";
String saved_password = "";

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  pinMode(LED_FLASH_GPIO, OUTPUT);
  digitalWrite(LED_FLASH_GPIO, LOW);
  pinMode(LED_BUILTIN_GPIO, OUTPUT);
  digitalWrite(LED_BUILTIN_GPIO, HIGH);

  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================");
  Serial.println("ESP32-CAM with Integrated ROI Setup");
  Serial.println("========================================");

  loadSettings();
  loadWiFiCredentials();
  loadROIConfig();

  pinMode(0, INPUT_PULLUP);
  delay(100);

  if (digitalRead(0) == LOW) {
    Serial.println("\n🔧 CONFIG MODE ACTIVATED!");
    configMode = true;
  }

  Serial.println("\nSend 'C' within 5 seconds for CONFIG MODE...");
  unsigned long startTime = millis();
  while (millis() - startTime < 5000) {
    if (Serial.available() > 0) {
      char cmd = Serial.read();
      if (cmd == 'C' || cmd == 'c') {
        Serial.println("\n🔧 CONFIG MODE ACTIVATED!");
        configMode = true;
        break;
      }
    }
    delay(10);
  }

  if (!initCamera()) {
    Serial.println("❌ Camera init failed!");
    goToDeepSleep();
  }

  applyCameraSettings();

  if (configMode) {
    startAPMode();
  } else {
    if (saved_ssid.length() > 0) {
      if (connectWiFi(saved_ssid.c_str(), saved_password.c_str())) {
        sendPhoto();
        goToDeepSleep();
      } else {
        Serial.println("⚠️ WiFi failed! Starting AP mode...");
        startAPMode();
      }
    } else {
      Serial.println("⚠️ No WiFi! Starting AP mode...");
      startAPMode();
    }
  }
}

void loop() {
  if (configMode || apMode) {
    dnsServer.processNextRequest();
    server.handleClient();

    static unsigned long lastBlink = 0;
    if (millis() - lastBlink > 1000) {
      digitalWrite(LED_BUILTIN_GPIO, !digitalRead(LED_BUILTIN_GPIO));
      lastBlink = millis();
    }

    if (millis() - configModeStart > CONFIG_MODE_TIMEOUT) {
      Serial.println("\n⏰ Timeout - sleeping");
      goToDeepSleep();
    }
  }
}

void startAPMode() {
  apMode = true;
  configMode = true;
  configModeStart = millis();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ap_ssid, ap_password);

  IPAddress IP = WiFi.softAPIP();
  dnsServer.start(53, "*", IP);
  setupWebServer();

  Serial.println("\n========================================");
  Serial.println("📡 ACCESS POINT MODE");
  Serial.println("SSID: " + String(ap_ssid));
  Serial.println("Password: " + String(ap_password));
  Serial.println("IP: " + IP.toString());
  Serial.println("========================================\n");
}

bool initCamera() {
  Serial.println("\n[CAMERA INIT]");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 15;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  Serial.println("✅ Camera initialized");
  return true;
}

void loadSettings() {
  preferences.begin("camera", false);
  camSettings.brightness = preferences.getInt("brightness", 0);
  camSettings.contrast = preferences.getInt("contrast", 0);
  camSettings.saturation = preferences.getInt("saturation", 0);
  camSettings.sharpness = preferences.getInt("sharpness", 0);
  camSettings.denoise = preferences.getInt("denoise", 0);
  camSettings.quality = preferences.getInt("quality", 12);
  camSettings.hmirror = preferences.getInt("hmirror", 0);
  camSettings.vflip = preferences.getInt("vflip", 0);
  camSettings.awb = preferences.getInt("awb", 1);
  camSettings.awb_gain = preferences.getInt("awb_gain", 1);
  camSettings.aec = preferences.getInt("aec", 1);
  camSettings.aec_value = preferences.getInt("aec_value", 300);
  camSettings.ae_level = preferences.getInt("ae_level", 0);
  camSettings.agc = preferences.getInt("agc", 1);
  camSettings.gainceiling = preferences.getInt("gainceiling", 0);
  camSettings.rotation = preferences.getInt("rotation", 0);
  preferences.end();
  Serial.println("📂 Camera settings loaded");
}

void loadWiFiCredentials() {
  preferences.begin("wifi", false);
  saved_ssid = preferences.getString("ssid", "");
  saved_password = preferences.getString("password", "");
  preferences.end();

  if (saved_ssid.length() > 0) {
    Serial.println("📡 WiFi: " + saved_ssid);
  }
}

void loadROIConfig() {
  preferences.begin("roi", false);
  numDigits = preferences.getInt("numDigits", 7);

  for (int i = 0; i < numDigits && i < MAX_DIGITS; i++) {
    String key = "roi" + String(i);
    String roiData = preferences.getString(key.c_str(), "");

    if (roiData.length() > 0) {
      sscanf(roiData.c_str(), "%d,%d,%d,%d", &rois[i].x, &rois[i].y, &rois[i].w,
             &rois[i].h);
      Serial.printf("ROI[%d]: (%d,%d,%d,%d)\n", i, rois[i].x, rois[i].y,
                    rois[i].w, rois[i].h);
    }
  }
  preferences.end();
  Serial.println("📐 ROI config loaded");
}

void saveROIConfig() {
  preferences.begin("roi", false);
  preferences.putInt("numDigits", numDigits);

  for (int i = 0; i < numDigits && i < MAX_DIGITS; i++) {
    String key = "roi" + String(i);
    String roiData = String(rois[i].x) + "," + String(rois[i].y) + "," +
                     String(rois[i].w) + "," + String(rois[i].h);
    preferences.putString(key.c_str(), roiData);
  }
  preferences.end();
  Serial.println("💾 ROI config saved");
}

void saveWiFiCredentials(String ssid, String password) {
  preferences.begin("wifi", false);
  preferences.putString("ssid", ssid);
  preferences.putString("password", password);
  preferences.end();
  saved_ssid = ssid;
  saved_password = password;
  Serial.println("💾 WiFi saved");
}

void saveSettings() {
  preferences.begin("camera", false);
  preferences.putInt("brightness", camSettings.brightness);
  preferences.putInt("contrast", camSettings.contrast);
  preferences.putInt("saturation", camSettings.saturation);
  preferences.putInt("sharpness", camSettings.sharpness);
  preferences.putInt("denoise", camSettings.denoise);
  preferences.putInt("quality", camSettings.quality);
  preferences.putInt("hmirror", camSettings.hmirror);
  preferences.putInt("vflip", camSettings.vflip);
  preferences.putInt("awb", camSettings.awb);
  preferences.putInt("awb_gain", camSettings.awb_gain);
  preferences.putInt("aec", camSettings.aec);
  preferences.putInt("aec_value", camSettings.aec_value);
  preferences.putInt("ae_level", camSettings.ae_level);
  preferences.putInt("agc", camSettings.agc);
  preferences.putInt("gainceiling", camSettings.gainceiling);
  preferences.putInt("rotation", camSettings.rotation);
  preferences.end();
  Serial.println("💾 Camera settings saved");
}

void applyCameraSettings() {
  sensor_t *s = esp_camera_sensor_get();
  if (s == NULL)
    return;

  s->set_brightness(s, camSettings.brightness);
  s->set_contrast(s, camSettings.contrast);
  s->set_saturation(s, camSettings.saturation);
  s->set_sharpness(s, camSettings.sharpness);
  s->set_denoise(s, camSettings.denoise);
  s->set_quality(s, camSettings.quality);
  s->set_hmirror(s, camSettings.hmirror);
  s->set_vflip(s, camSettings.vflip);
  s->set_whitebal(s, camSettings.awb);
  s->set_awb_gain(s, camSettings.awb_gain);
  s->set_exposure_ctrl(s, camSettings.aec);
  s->set_aec_value(s, camSettings.aec_value);
  s->set_ae_level(s, camSettings.ae_level);
  s->set_gain_ctrl(s, camSettings.agc);
  s->set_gainceiling(s, (gainceiling_t)camSettings.gainceiling);
}

void setupWebServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/stream", HTTP_GET, handleStream);
  server.on("/settings", HTTP_GET, handleGetSettings);
  server.on("/settings", HTTP_POST, handleSetSettings);
  server.on("/scan", HTTP_GET, handleWiFiScan);
  server.on("/connect", HTTP_POST, handleWiFiConnect);
  server.on("/roi", HTTP_GET, handleGetROI);
  server.on("/roi", HTTP_POST, handleSetROI);
  server.on("/save", HTTP_POST, handleSave);
  server.on("/sleep", HTTP_POST, handleSleep);
  server.on("/restart", HTTP_POST, handleRestart);
  server.begin();
}

void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ESP32-CAM Complete Setup</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, system-ui, sans-serif; background: #0a0a0a; color: #fff; padding: 15px; }
    .container { max-width: 1400px; margin: auto; }
    h1 { text-align: center; margin-bottom: 20px; color: #00ff88; font-size: 24px; }
    
    .tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
    .tab { padding: 12px 20px; background: #1a1a1a; border: 2px solid #333; border-radius: 8px; cursor: pointer; transition: all 0.3s; font-size: 14px; }
    .tab.active { background: #00ff88; color: #000; border-color: #00ff88; font-weight: bold; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    /* Camera Tab - Two Column Layout */
    .camera-tab { display: grid; grid-template-columns: 1fr 400px; gap: 20px; }
    @media (max-width: 1000px) {
      .camera-tab { grid-template-columns: 1fr; }
    }
    
    /* Live Preview Mini */
    .live-preview-mini { background: #000; border-radius: 10px; overflow: hidden; margin-bottom: 20px; }
    .live-preview-mini h3 { padding: 15px; margin: 0; background: #1a1a1a; color: #00ff88; }
    .preview-container { position: relative; }
    #cameraPreview { width: 100%; max-height: 300px; object-fit: contain; display: block; }
    .preview-refresh { position: absolute; bottom: 10px; right: 10px; background: rgba(0, 0, 0, 0.7); color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; }
    
    /* ROI Setup Styles */
    .roi-setup { background: #1a1a1a; padding: 20px; border-radius: 10px; }
    .roi-canvas-container { position: relative; background: #000; border-radius: 8px; overflow: hidden; margin-bottom: 15px; }
    #roiCanvas { max-width: 100%; display: block; cursor: crosshair; }
    .roi-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-bottom: 15px; }
    .roi-info { background: #2a2a2a; padding: 15px; border-radius: 8px; }
    .roi-info h4 { color: #00ff88; margin-bottom: 10px; }
    .digit-status { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 10px; }
    .digit-box { padding: 8px 12px; background: #333; border-radius: 5px; font-size: 12px; }
    .digit-box.done { background: #00ff88; color: #000; font-weight: bold; }
    .digit-box.active { background: #ff8800; color: #fff; font-weight: bold; animation: pulse 1s infinite; }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    /* Main Preview */
    .preview { background: #000; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .stream-container { position: relative; max-width: 800px; margin: auto; }
    .stream-container img { width: 100%; border-radius: 5px; }
    
    .wifi-section { background: #1a1a1a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    .wifi-item { padding: 15px; background: #2a2a2a; border-radius: 5px; margin-bottom: 10px; cursor: pointer; display: flex; justify-content: space-between; }
    .wifi-item:hover { background: #333; }
    .wifi-item.selected { border: 2px solid #00ff88; }
    .wifi-form input { width: 100%; padding: 12px; margin-bottom: 10px; background: #2a2a2a; border: 1px solid #444; border-radius: 5px; color: #fff; font-size: 16px; }
    
    .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
    .control-group { background: #1a1a1a; padding: 20px; border-radius: 10px; }
    .control-group h3 { margin-bottom: 15px; color: #00ff88; font-size: 16px; }
    .control-item { margin-bottom: 15px; }
    .control-item label { display: block; margin-bottom: 5px; color: #bbb; font-size: 14px; }
    .control-item input[type="range"] { width: 100%; }
    .control-item input[type="checkbox"] { width: 20px; height: 20px; }
    .value-display { float: right; color: #00ff88; font-weight: bold; }
    
    .actions { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
    .actions button { flex: 1; min-width: 150px; padding: 15px; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; }
    .btn-primary { background: #00ff88; color: #000; }
    .btn-danger { background: #ff4444; color: white; }
    .btn-info { background: #4444ff; color: white; }
    .btn-warning { background: #ff8800; color: white; }
    .btn-small { padding: 8px 15px; font-size: 14px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>📸 ESP32-CAM Complete Setup System</h1>
    
    <div class="tabs">
      <div class="tab active" onclick="switchTab('preview')">📹 Live Preview</div>
      <div class="tab" onclick="switchTab('roi')">📐 ROI Setup</div>
      <div class="tab" onclick="switchTab('wifi')">📡 WiFi</div>
      <div class="tab" onclick="switchTab('camera')">⚙️ Camera</div>
    </div>

    <!-- LIVE PREVIEW -->
    <div id="preview" class="tab-content active">
      <div class="preview">
        <div class="stream-container">
          <img id="stream" src="/stream" alt="Loading...">
        </div>
      </div>
    </div>

    <!-- ROI SETUP -->
    <div id="roi" class="tab-content">
      <div class="roi-setup">
        <div class="roi-info">
          <h4>🎯 ROI Setup - Select Digit Regions</h4>
          <p style="color: #bbb; margin-bottom: 10px;">Click and drag on the image to select each digit region (left to right)</p>
          <div style="margin-bottom: 10px;">
            <label>Number of Digits: <input type="number" id="numDigits" value="7" min="1" max="10" style="width: 80px; padding: 5px; background: #333; color: #fff; border: 1px solid #555; border-radius: 3px;"></label>
            <button onclick="updateDigitCount()" style="padding: 5px 15px; margin-left: 10px; background: #00ff88; color: #000; border: none; border-radius: 5px; cursor: pointer;">Update</button>
          </div>
          <div class="digit-status" id="digitStatus"></div>
          <div style="margin-top: 15px;">
            <strong>Current: <span id="currentDigit" style="color: #ff8800;">-</span></strong> | 
            <strong>Progress: <span id="roiProgress">0/7</span></strong>
          </div>
        </div>
        
        <div class="roi-canvas-container">
          <canvas id="roiCanvas"></canvas>
        </div>
        
        <div class="roi-controls">
          <button class="btn-warning" onclick="redoROI()" style="width: 100%;">↺ Redo Current</button>
          <button class="btn-info" onclick="clearAllROI()" style="width: 100%;">✗ Clear All</button>
          <button class="btn-primary" onclick="saveROI()" style="width: 100%;">💾 Save ROI Config</button>
        </div>
      </div>
    </div>

    <!-- WIFI SETUP -->
    <div id="wifi" class="tab-content">
      <div class="wifi-section">
        <h3>WiFi Connection</h3>
        <button class="btn-primary" onclick="scanWiFi()" style="width: 100%; margin-bottom: 15px;">🔍 Scan Networks</button>
        <div class="wifi-list" id="wifiList"></div>
        <div class="wifi-form">
          <input type="text" id="ssidInput" placeholder="WiFi SSID">
          <input type="password" id="passInput" placeholder="Password">
          <button class="btn-primary" onclick="connectWiFi()" style="width: 100%;">🔗 Connect & Save</button>
        </div>
      </div>
    </div>

    <!-- CAMERA SETTINGS with Live Preview -->
    <div id="camera" class="tab-content">
      <div class="camera-tab">
        <div>
          <div class="controls">
            <div class="control-group">
              <h3>🎨 Image Quality</h3>
              <div class="control-item">
                <label>Brightness <span class="value-display" id="val_brightness">0</span></label>
                <input type="range" id="brightness" min="-2" max="2" value="0" oninput="updateSetting(this, true)">
              </div>
              <div class="control-item">
                <label>Contrast <span class="value-display" id="val_contrast">0</span></label>
                <input type="range" id="contrast" min="-2" max="2" value="0" oninput="updateSetting(this, true)">
              </div>
              <div class="control-item">
                <label>Saturation <span class="value-display" id="val_saturation">0</span></label>
                <input type="range" id="saturation" min="-2" max="2" value="0" oninput="updateSetting(this, true)">
              </div>
              <div class="control-item">
                <label>Sharpness <span class="value-display" id="val_sharpness">0</span></label>
                <input type="range" id="sharpness" min="-2" max="2" value="0" oninput="updateSetting(this, true)">
              </div>
              <div class="control-item">
                <label>Quality <span class="value-display" id="val_quality">12</span></label>
                <input type="range" id="quality" min="4" max="63" value="12" oninput="updateSetting(this, true)">
              </div>
            </div>
            
            <div class="control-group">
              <h3>⚡ Exposure & Gain</h3>
              <div class="control-item">
                <label><input type="checkbox" id="aec" onchange="updateSetting(this, true)"> Auto Exposure</label>
              </div>
              <div class="control-item">
                <label>Exposure Value <span class="value-display" id="val_aec_value">300</span></label>
                <input type="range" id="aec_value" min="0" max="1200" value="300" oninput="updateSetting(this, true)">
              </div>
              <div class="control-item">
                <label><input type="checkbox" id="agc" onchange="updateSetting(this, true)"> Auto Gain</label>
              </div>
              <div class="control-item">
                <label>Gain Ceiling <span class="value-display" id="val_gainceiling">0</span></label>
                <input type="range" id="gainceiling" min="0" max="6" value="0" oninput="updateSetting(this, true)">
              </div>
            </div>
            
            <div class="control-group">
              <h3>🔄 Orientation</h3>
              <div class="control-item">
                <label>Rotation <span class="value-display" id="val_rotation">0</span>°</label>
                <input type="range" id="rotation" min="0" max="359" value="0" oninput="updateSetting(this, true)">
              </div>
              <div class="control-item">
                <label><input type="checkbox" id="hmirror" onchange="updateSetting(this, true)"> H-Mirror</label>
              </div>
              <div class="control-item">
                <label><input type="checkbox" id="vflip" onchange="updateSetting(this, true)"> V-Flip</label>
              </div>
            </div>
            
            <div class="control-group">
              <h3>🌈 White Balance</h3>
              <div class="control-item">
                <label><input type="checkbox" id="awb" onchange="updateSetting(this, true)"> Auto White Balance</label>
              </div>
              <div class="control-item">
                <label><input type="checkbox" id="awb_gain" onchange="updateSetting(this, true)"> AWB Gain</label>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Live Preview Mini -->
        <div class="live-preview-mini">
          <h3>📸 Live Preview</h3>
          <div class="preview-container">
            <img id="cameraPreview" src="/stream" alt="Camera Preview">
            <button class="preview-refresh" onclick="refreshPreview()">🔄 Refresh</button>
          </div>
          <div style="padding: 15px; background: #1a1a1a; margin-top: 10px; border-radius: 5px;">
            <p style="color: #bbb; font-size: 14px; margin-bottom: 10px;">Adjust settings and see live changes here.</p>
            <button class="btn-small btn-primary" onclick="autoRefreshPreview()" id="autoRefreshBtn">⏯️ Auto-refresh: OFF</button>
            <span style="color: #888; font-size: 12px; margin-left: 10px;" id="lastUpdate">Last update: -</span>
          </div>
        </div>
      </div>
    </div>

    <div class="actions">
      <button class="btn-primary" onclick="saveAll()">💾 Save All</button>
      <button class="btn-info" onclick="restartDevice()">🔄 Restart</button>
      <button class="btn-danger" onclick="goToSleep()">😴 Sleep</button>
    </div>
  </div>

  <script>
    let canvas, ctx, img;
    let selecting = false;
    let startX, startY, endX, endY;
    let roiData = [];
    let currentDigitIndex = 0;
    let totalDigits = 7;
    let imageNaturalWidth, imageNaturalHeight;
    let autoRefreshInterval = null;
    let autoRefreshActive = false;
    let previewRefreshTimer = null;

    function switchTab(tab) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(tab).classList.add('active');
      
      // Stop auto-refresh when switching away from camera tab
      if (tab !== 'camera') {
        stopAutoRefresh();
      }
      
      if (tab === 'roi') {
        setTimeout(initROICanvas, 100);
      } else if (tab === 'camera') {
        // Start auto-refresh for camera tab
        setTimeout(refreshCameraPreview, 100);
      }
    }

    function refreshCameraPreview() {
      const img = document.getElementById('cameraPreview');
      if (img) {
        const timestamp = new Date().getTime();
        img.src = '/stream?' + timestamp;
        document.getElementById('lastUpdate').textContent = 'Last update: ' + new Date().toLocaleTimeString();
      }
    }

    function refreshPreview() {
      refreshCameraPreview();
    }

    function autoRefreshPreview() {
      const btn = document.getElementById('autoRefreshBtn');
      if (autoRefreshActive) {
        stopAutoRefresh();
        btn.textContent = '⏯️ Auto-refresh: OFF';
        btn.style.background = '#00ff88';
      } else {
        startAutoRefresh();
        btn.textContent = '⏯️ Auto-refresh: ON';
        btn.style.background = '#ff8800';
      }
    }

    function startAutoRefresh() {
      autoRefreshActive = true;
      if (autoRefreshInterval) clearInterval(autoRefreshInterval);
      autoRefreshInterval = setInterval(refreshCameraPreview, 500); // Refresh every 500ms
      refreshCameraPreview();
    }

    function stopAutoRefresh() {
      autoRefreshActive = false;
      if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
      }
    }

    // Rest of your existing functions remain the same...
    function initROICanvas() {
      canvas = document.getElementById('roiCanvas');
      ctx = canvas.getContext('2d');
      img = new Image();
      img.crossOrigin = "anonymous";
      
      img.onload = function() {
        imageNaturalWidth = img.naturalWidth;
        imageNaturalHeight = img.naturalHeight;
        
        const maxWidth = Math.min(1200, window.innerWidth - 100);
        const scale = maxWidth / img.naturalWidth;
        
        canvas.width = img.naturalWidth * scale;
        canvas.height = img.naturalHeight * scale;
        
        drawCanvas();
        updateDigitStatus();
      };
      
      img.src = '/stream?' + Date.now();
      
      canvas.addEventListener('mousedown', onCanvasMouseDown);
      canvas.addEventListener('mousemove', onCanvasMouseMove);
      canvas.addEventListener('mouseup', onCanvasMouseUp);
      
      // Touch support
      canvas.addEventListener('touchstart', handleTouch);
      canvas.addEventListener('touchmove', handleTouch);
      canvas.addEventListener('touchend', handleTouch);
      
      loadROI();
    }

    function handleTouch(e) {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const touch = e.touches[0] || e.changedTouches[0];
      const mouseEvent = new MouseEvent(e.type.replace('touch', 'mouse'), {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      canvas.dispatchEvent(mouseEvent);
    }

    function onCanvasMouseDown(e) {
      if (currentDigitIndex >= totalDigits) return;
      
      const rect = canvas.getBoundingClientRect();
      startX = e.clientX - rect.left;
      startY = e.clientY - rect.top;
      selecting = true;
    }

    function onCanvasMouseMove(e) {
      if (!selecting) return;
      
      const rect = canvas.getBoundingClientRect();
      endX = e.clientX - rect.left;
      endY = e.clientY - rect.top;
      drawCanvas();
    }

    function onCanvasMouseUp(e) {
      if (!selecting) return;
      
      const rect = canvas.getBoundingClientRect();
      endX = e.clientX - rect.left;
      endY = e.clientY - rect.top;
      selecting = false;
      
      const scaleX = imageNaturalWidth / canvas.width;
      const scaleY = imageNaturalHeight / canvas.height;
      
      const x = Math.min(startX, endX) * scaleX;
      const y = Math.min(startY, endY) * scaleY;
      const w = Math.abs(endX - startX) * scaleX;
      const h = Math.abs(endY - startY) * scaleY;
      
      if (w > 10 && h > 10) {
        roiData[currentDigitIndex] = {x: Math.floor(x), y: Math.floor(y), w: Math.floor(w), h: Math.floor(h)};
        currentDigitIndex++;
        drawCanvas();
        updateDigitStatus();
      }
    }

    function drawCanvas() {
      if (!img.complete) return;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      const scaleX = canvas.width / imageNaturalWidth;
      const scaleY = canvas.height / imageNaturalHeight;
      const colors = ['#00ff00', '#00ffff', '#ffff00', '#ff00ff', '#ff8800', '#0088ff', '#88ff00', '#ff0088', '#00ff88', '#8800ff'];
      
      // Draw existing ROIs
      roiData.forEach((roi, i) => {
        if (roi) {
          ctx.strokeStyle = colors[i % colors.length];
          ctx.lineWidth = 3;
          ctx.strokeRect(roi.x * scaleX, roi.y * scaleY, roi.w * scaleX, roi.h * scaleY);
          ctx.fillStyle = colors[i % colors.length];
          ctx.font = 'bold 16px Arial';
          ctx.fillText('D' + (i + 1), roi.x * scaleX + 5, roi.y * scaleY - 5);
        }
      });
      
      // Draw current selection
      if (selecting) {
        ctx.strokeStyle = '#ff8800';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(startX, startY, endX - startX, endY - startY);
        ctx.setLineDash([]);
      }
    }

    function updateDigitStatus() {
      const status = document.getElementById('digitStatus');
      status.innerHTML = '';
      
      for (let i = 0; i < totalDigits; i++) {
        const box = document.createElement('div');
        box.className = 'digit-box';
        box.textContent = 'D' + (i + 1);
        
        if (roiData[i]) {
          box.classList.add('done');
        } else if (i === currentDigitIndex) {
          box.classList.add('active');
        }
        
        status.appendChild(box);
      }
      
      document.getElementById('currentDigit').textContent = 
        currentDigitIndex < totalDigits ? 'D' + (currentDigitIndex + 1) : 'Complete!';
      document.getElementById('roiProgress').textContent = roiData.filter(r => r).length + '/' + totalDigits;
    }

    function updateDigitCount() {
      const num = parseInt(document.getElementById('numDigits').value);
      if (num >= 1 && num <= 10) {
        totalDigits = num;
        currentDigitIndex = 0;
        roiData = [];
        updateDigitStatus();
        drawCanvas();
      }
    }

    function redoROI() {
      if (currentDigitIndex > 0) {
        currentDigitIndex--;
        roiData[currentDigitIndex] = null;
        updateDigitStatus();
        drawCanvas();
      }
    }

    function clearAllROI() {
      if (confirm('Clear all ROI selections?')) {
        roiData = [];
        currentDigitIndex = 0;
        updateDigitStatus();
        drawCanvas();
      }
    }

    function loadROI() {
      fetch('/roi')
        .then(r => r.json())
        .then(data => {
          totalDigits = data.numDigits || 7;
          document.getElementById('numDigits').value = totalDigits;
          
          roiData = [];
          for (let i = 0; i < totalDigits; i++) {
            if (data.rois && data.rois[i]) {
              roiData[i] = data.rois[i];
            }
          }
          
          currentDigitIndex = roiData.filter(r => r).length;
          updateDigitStatus();
          drawCanvas();
        })
        .catch(e => console.log('No ROI data yet'));
    }

    function saveROI() {
      const complete = roiData.filter(r => r).length;
      if (complete < totalDigits) {
        if (!confirm(`Only ${complete}/${totalDigits} digits selected. Save anyway?`)) {
          return;
        }
      }
      
      fetch('/roi', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          numDigits: totalDigits,
          rois: roiData
        })
      })
      .then(r => r.text())
      .then(data => alert(data))
      .catch(e => alert('Save failed: ' + e));
    }

    // Camera settings with live preview
    function updateSetting(elem, liveUpdate = false) {
      const id = elem.id;
      let value = elem.type === 'checkbox' ? (elem.checked ? 1 : 0) : elem.value;
      
      if (elem.type !== 'checkbox' && document.getElementById('val_' + id)) {
        document.getElementById('val_' + id).textContent = value;
      }
      
      // If liveUpdate is true, apply setting immediately and refresh preview
      if (liveUpdate) {
        fetch('/settings', {
          method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: id + '=' + value
        })
        .then(() => {
          if (autoRefreshActive) {
            // Clear any pending timer and set new one
            if (previewRefreshTimer) clearTimeout(previewRefreshTimer);
            previewRefreshTimer = setTimeout(refreshCameraPreview, 300); // Wait 300ms before refreshing
          }
        });
      } else {
        fetch('/settings', {
          method: 'POST',
          headers: {'Content-Type': 'application/x-www-form-urlencoded'},
          body: id + '=' + value
        });
      }
    }

    // WiFi functions
    function scanWiFi() {
      document.getElementById('wifiList').innerHTML = '<p style="color: #888;">Scanning...</p>';
      fetch('/scan')
        .then(r => r.json())
        .then(data => {
          let html = '';
          data.networks.forEach(net => {
            html += `<div class="wifi-item" onclick="selectWiFi('${net.ssid}')">
              <div><strong>${net.ssid}</strong></div>
              <div style="color: #00ff88;">${net.rssi} dBm</div>
            </div>`;
          });
          document.getElementById('wifiList').innerHTML = html || '<p>No networks</p>';
        });
    }

    function selectWiFi(ssid) {
      document.getElementById('ssidInput').value = ssid;
      document.querySelectorAll('.wifi-item').forEach(item => {
        item.classList.remove('selected');
        if (item.textContent.includes(ssid)) {
          item.classList.add('selected');
        }
      });
    }

    function connectWiFi() {
      const ssid = document.getElementById('ssidInput').value;
      const pass = document.getElementById('passInput').value;
      
      if (!ssid) {
        alert('Enter WiFi SSID');
        return;
      }
      
      fetch('/connect', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: 'ssid=' + encodeURIComponent(ssid) + '&password=' + encodeURIComponent(pass)
      })
      .then(r => r.text())
      .then(data => alert(data));
    }

    function saveAll() {
      fetch('/save', {method: 'POST'})
        .then(r => r.text())
        .then(data => alert(data));
    }

    function restartDevice() {
      if (confirm('Restart device?')) {
        fetch('/restart', {method: 'POST'})
          .then(() => alert('Restarting...'));
      }
    }

    function goToSleep() {
      if (confirm('Enter sleep mode?')) {
        fetch('/sleep', {method: 'POST'})
          .then(() => alert('Sleeping...'));
      }
    }

    // Load settings on start
    fetch('/settings')
      .then(r => r.json())
      .then(data => {
        for (let key in data) {
          let elem = document.getElementById(key);
          if (elem) {
            if (elem.type === 'checkbox') {
              elem.checked = data[key] === 1;
            } else {
              elem.value = data[key];
              let display = document.getElementById('val_' + key);
              if (display) display.textContent = data[key];
            }
          }
        }
      });

    // Auto-refresh main stream
    setInterval(() => {
      const img = document.getElementById('stream');
      if (img) img.src = '/stream?' + Date.now();
    }, 200);

    // Start auto-refresh when camera tab is active
    window.addEventListener('load', function() {
      refreshCameraPreview();
    });
  </script>
</body>
</html>
)rawliteral";

  server.send(200, "text/html", html);
}

void handleStream() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(503, "text/plain", "Camera busy");
    return;
  }

  server.sendHeader("Cache-Control", "no-cache");
  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handleGetSettings() {
  String json = "{";
  json += "\"brightness\":" + String(camSettings.brightness) + ",";
  json += "\"contrast\":" + String(camSettings.contrast) + ",";
  json += "\"saturation\":" + String(camSettings.saturation) + ",";
  json += "\"sharpness\":" + String(camSettings.sharpness) + ",";
  json += "\"denoise\":" + String(camSettings.denoise) + ",";
  json += "\"quality\":" + String(camSettings.quality) + ",";
  json += "\"rotation\":" + String(camSettings.rotation) + ",";
  json += "\"hmirror\":" + String(camSettings.hmirror) + ",";
  json += "\"vflip\":" + String(camSettings.vflip) + ",";
  json += "\"awb\":" + String(camSettings.awb) + ",";
  json += "\"awb_gain\":" + String(camSettings.awb_gain) + ",";
  json += "\"aec\":" + String(camSettings.aec) + ",";
  json += "\"aec_value\":" + String(camSettings.aec_value) + ",";
  json += "\"ae_level\":" + String(camSettings.ae_level) + ",";
  json += "\"agc\":" + String(camSettings.agc) + ",";
  json += "\"gainceiling\":" + String(camSettings.gainceiling);
  json += "}";
  server.send(200, "application/json", json);
}

void handleSetSettings() {
  for (int i = 0; i < server.args(); i++) {
    String name = server.argName(i);
    int value = server.arg(i).toInt();

    if (name == "brightness")
      camSettings.brightness = value;
    else if (name == "contrast")
      camSettings.contrast = value;
    else if (name == "saturation")
      camSettings.saturation = value;
    else if (name == "sharpness")
      camSettings.sharpness = value;
    else if (name == "denoise")
      camSettings.denoise = value;
    else if (name == "quality")
      camSettings.quality = value;
    else if (name == "rotation")
      camSettings.rotation = value;
    else if (name == "hmirror")
      camSettings.hmirror = value;
    else if (name == "vflip")
      camSettings.vflip = value;
    else if (name == "awb")
      camSettings.awb = value;
    else if (name == "awb_gain")
      camSettings.awb_gain = value;
    else if (name == "aec")
      camSettings.aec = value;
    else if (name == "aec_value")
      camSettings.aec_value = value;
    else if (name == "ae_level")
      camSettings.ae_level = value;
    else if (name == "agc")
      camSettings.agc = value;
    else if (name == "gainceiling")
      camSettings.gainceiling = value;
  }

  applyCameraSettings();
  server.send(200, "text/plain", "OK");
}

void handleGetROI() {
  DynamicJsonDocument doc(2048);
  doc["numDigits"] = numDigits;

  JsonArray roisArray = doc.createNestedArray("rois");
  for (int i = 0; i < numDigits && i < MAX_DIGITS; i++) {
    JsonObject roiObj = roisArray.createNestedObject();
    roiObj["x"] = rois[i].x;
    roiObj["y"] = rois[i].y;
    roiObj["w"] = rois[i].w;
    roiObj["h"] = rois[i].h;
  }

  String output;
  serializeJson(doc, output);
  server.send(200, "application/json", output);
}

void handleSetROI() {
  DynamicJsonDocument doc(2048);
  DeserializationError error = deserializeJson(doc, server.arg("plain"));

  if (error) {
    server.send(400, "text/plain", "Invalid JSON");
    return;
  }

  numDigits = doc["numDigits"] | 7;
  JsonArray roisArray = doc["rois"];

  for (int i = 0; i < numDigits && i < MAX_DIGITS && i < roisArray.size();
       i++) {
    JsonObject roiObj = roisArray[i];
    if (!roiObj.isNull()) {
      rois[i].x = roiObj["x"] | 0;
      rois[i].y = roiObj["y"] | 0;
      rois[i].w = roiObj["w"] | 0;
      rois[i].h = roiObj["h"] | 0;
    }
  }

  saveROIConfig();
  server.send(200, "text/plain", "✅ ROI configuration saved!");
}

void handleWiFiScan() {
  int n = WiFi.scanNetworks();
  String json = "{\"networks\":[";
  for (int i = 0; i < n; i++) {
    if (i > 0)
      json += ",";
    json += "{";
    json += "\"ssid\":\"" + WiFi.SSID(i) + "\",";
    json += "\"rssi\":" + String(WiFi.RSSI(i)) + ",";
    json +=
        "\"encryption\":\"" +
        String(WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "Open" : "Secured") +
        "\"";
    json += "}";
  }
  json += "]}";
  server.send(200, "application/json", json);
}

void handleWiFiConnect() {
  String ssid = server.arg("ssid");
  String password = server.arg("password");

  WiFi.mode(WIFI_AP_STA);
  WiFi.begin(ssid.c_str(), password.c_str());

  int timeout = 20;
  while (WiFi.status() != WL_CONNECTED && timeout > 0) {
    delay(500);
    timeout--;
  }

  if (WiFi.status() == WL_CONNECTED) {
    saveWiFiCredentials(ssid, password);
    server.send(200, "text/plain",
                "✅ Connected! IP: " + WiFi.localIP().toString());
  } else {
    server.send(200, "text/plain", "❌ Connection failed!");
    WiFi.mode(WIFI_AP);
  }
}

void handleSave() {
  saveSettings();
  server.send(200, "text/plain", "✅ Settings saved!");
}

void handleSleep() {
  saveSettings();
  server.send(200, "text/plain", "Sleeping...");
  delay(1000);
  goToDeepSleep();
}

void handleRestart() {
  saveSettings();
  server.send(200, "text/plain", "Restarting...");
  delay(1000);
  ESP.restart();
}

bool connectWiFi(const char *ssid, const char *password) {
  Serial.println("\n[WIFI] " + String(ssid));
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int retries = 20;
  while (WiFi.status() != WL_CONNECTED && retries > 0) {
    delay(500);
    Serial.print(".");
    retries--;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ Connected! IP: " + WiFi.localIP().toString());
    return true;
  }

  Serial.println("\n❌ Failed!");
  return false;
}

String sendPhoto() {
  Serial.println("\n[UPLOAD]");
  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
    Serial.println("Capture failed");
    return "Error";
  }

  Serial.printf("Uploading %d bytes...\n", fb->len);

  if (!client.connect(serverName, serverPort)) {
    Serial.println("Connection failed");
    esp_camera_fb_return(fb);
    return "Error";
  }

  String head = "--Boundary\r\nContent-Disposition: form-data; name=\"image\"; "
                "filename=\"cam.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--Boundary--\r\n";

  // Build ROI JSON
  String roiJson = "{\"numDigits\":" + String(numDigits) + ",\"rois\":[";
  for (int i = 0; i < numDigits && i < MAX_DIGITS; i++) {
    if (i > 0)
      roiJson += ",";
    roiJson += "{\"x\":" + String(rois[i].x) + ",\"y\":" + String(rois[i].y) +
               ",\"w\":" + String(rois[i].w) + ",\"h\":" + String(rois[i].h) +
               "}";
  }
  roiJson += "]}";

  client.println("POST " + String(serverPath) + " HTTP/1.1");
  client.println("Host: " + String(serverName));
  client.println("X-API-Key: example");
  client.println("X-ROI-Config: " + roiJson);
  client.println("Content-Length: " +
                 String(fb->len + head.length() + tail.length()));
  client.println("Content-Type: multipart/form-data; boundary=Boundary");
  client.println();
  client.print(head);

  uint8_t *fbBuf = fb->buf;
  size_t fbLen = fb->len;
  for (size_t n = 0; n < fbLen; n += 1024) {
    client.write(fbBuf + n, min((size_t)1024, fbLen - n));
  }

  client.print(tail);
  client.stop();
  esp_camera_fb_return(fb);

  Serial.println("✅ Uploaded with ROI config");
  return "OK";
}

void goToDeepSleep() {
  Serial.println("Waiting 5s for transfer stability...");
  delay(5000);

  Serial.println("\n💤 Sleeping " + String(TIME_TO_SLEEP) + " min");
  Serial.flush();

  esp_camera_deinit();
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);

  esp_sleep_enable_timer_wakeup(TIME_TO_SLEEP * 60 * uS_TO_S_FACTOR);
  esp_deep_sleep_start();
}
