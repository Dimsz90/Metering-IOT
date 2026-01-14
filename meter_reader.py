import cv2
import numpy as np
from datetime import datetime
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from collections import Counter
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# ==================== OCR ENGINE ====================
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'/usr/bin/tesseract',
        r'/usr/local/bin/tesseract'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            TESSERACT_AVAILABLE = True
            print(f"✓ Tesseract found at: {path}")
            break
    
    if not TESSERACT_AVAILABLE:
        print("⚠ Tesseract not found")
except ImportError:
    print("⚠ pytesseract not installed")

# ==================== CONFIGURATION ====================
CONFIG_FILE = "roi_config.json"
DEBUG_MODE = True

# ==================== ROI SELECTOR GUI ====================

class ROISelector:
    """Interactive ROI selector with GUI"""
    
    def __init__(self, image_path, num_digits=7):
        self.image_path = image_path
        self.num_digits = num_digits
        self.rois = {}
        self.current_digit = 1
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.temp_rect = None
        
        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        self.display_image = self.original_image.copy()
        self.scale_factor = 1.0
        
        # Calculate display size
        max_width = 1200
        max_height = 800
        h, w = self.original_image.shape[:2]
        
        if w > max_width or h > max_height:
            scale_w = max_width / w
            scale_h = max_height / h
            self.scale_factor = min(scale_w, scale_h)
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            self.display_image = cv2.resize(self.original_image, (new_w, new_h))
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup GUI window"""
        self.root = tk.Tk()
        self.root.title("ROI Setup - Water Meter Digit Selector")
        
        # Instructions frame
        inst_frame = tk.Frame(self.root, bg='#2c3e50', pady=10)
        inst_frame.pack(fill=tk.X)
        
        title = tk.Label(inst_frame, text="📐 ROI Setup - Select Digit Regions", 
                        font=('Arial', 14, 'bold'), fg='white', bg='#2c3e50')
        title.pack()
        
        instructions = tk.Label(inst_frame, 
                               text="Click and drag to select each digit region • Left to Right order • Press 'Next' when done",
                               font=('Arial', 10), fg='#ecf0f1', bg='#2c3e50')
        instructions.pack()
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#34495e', pady=10)
        control_frame.pack(fill=tk.X)
        
        self.digit_label = tk.Label(control_frame, 
                                    text=f"Select Digit {self.current_digit} of {self.num_digits}",
                                    font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        self.digit_label.pack(side=tk.LEFT, padx=20)
        
        btn_style = {'font': ('Arial', 10, 'bold'), 'padx': 15, 'pady': 5}
        
        self.next_btn = tk.Button(control_frame, text="✓ Next Digit", 
                                  command=self.next_digit, 
                                  bg='#27ae60', fg='white', **btn_style)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn.config(state=tk.DISABLED)
        
        self.redo_btn = tk.Button(control_frame, text="↺ Redo", 
                                  command=self.redo_current,
                                  bg='#f39c12', fg='white', **btn_style)
        self.redo_btn.pack(side=tk.LEFT, padx=5)
        self.redo_btn.config(state=tk.DISABLED)
        
        self.finish_btn = tk.Button(control_frame, text="✓ Finish Setup", 
                                    command=self.finish_setup,
                                    bg='#3498db', fg='white', **btn_style)
        self.finish_btn.pack(side=tk.LEFT, padx=5)
        self.finish_btn.config(state=tk.DISABLED)
        
        tk.Button(control_frame, text="✗ Cancel", 
                 command=self.cancel,
                 bg='#e74c3c', fg='white', **btn_style).pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(control_frame, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=20)
        self.progress['maximum'] = self.num_digits
        
        # Canvas frame
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Convert image for display
        self.photo_image = self.cv2_to_photo(self.display_image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Update canvas size
        self.root.update()
        h, w = self.display_image.shape[:2]
        self.canvas.config(width=w, height=h)
    
    def cv2_to_photo(self, cv2_image):
        """Convert CV2 image to PhotoImage"""
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return ImageTk.PhotoImage(pil_image)
    
    def on_mouse_down(self, event):
        """Mouse down event"""
        self.selecting = True
        self.start_point = (event.x, event.y)
        self.end_point = (event.x, event.y)
    
    def on_mouse_drag(self, event):
        """Mouse drag event"""
        if self.selecting:
            self.end_point = (event.x, event.y)
            self.update_display()
    
    def on_mouse_up(self, event):
        """Mouse up event"""
        if self.selecting:
            self.end_point = (event.x, event.y)
            self.selecting = False
            
            # Calculate ROI in original image coordinates
            x1 = int(min(self.start_point[0], self.end_point[0]) / self.scale_factor)
            y1 = int(min(self.start_point[1], self.end_point[1]) / self.scale_factor)
            x2 = int(max(self.start_point[0], self.end_point[0]) / self.scale_factor)
            y2 = int(max(self.start_point[1], self.end_point[1]) / self.scale_factor)
            
            w = x2 - x1
            h = y2 - y1
            
            # Validate ROI size
            if w > 5 and h > 5:
                self.rois[self.current_digit] = (x1, y1, w, h)
                self.next_btn.config(state=tk.NORMAL)
                self.redo_btn.config(state=tk.NORMAL)
                print(f"✓ Digit {self.current_digit} ROI: ({x1}, {y1}, {w}, {h})")
            else:
                messagebox.showwarning("Invalid ROI", "ROI too small. Please select a larger area.")
                self.start_point = None
                self.end_point = None
            
            self.update_display()
    
    def update_display(self):
        """Update canvas display"""
        display = self.display_image.copy()
        
        # Draw existing ROIs
        colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255),
                 (0, 128, 255), (255, 128, 0), (128, 255, 0)]
        
        for digit_num in sorted(self.rois.keys()):
            x, y, w, h = self.rois[digit_num]
            sx = int(x * self.scale_factor)
            sy = int(y * self.scale_factor)
            sw = int(w * self.scale_factor)
            sh = int(h * self.scale_factor)
            
            color = colors[(digit_num - 1) % len(colors)]
            cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), color, 2)
            cv2.putText(display, f"D{digit_num}", (sx, sy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw current selection
        if self.start_point and self.end_point:
            color = (0, 255, 255) if self.selecting else (0, 255, 0)
            cv2.rectangle(display, self.start_point, self.end_point, color, 2)
            
            if not self.selecting:
                cv2.putText(display, f"D{self.current_digit}", 
                           (self.start_point[0], self.start_point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Update canvas
        self.photo_image = self.cv2_to_photo(display)
        self.canvas.itemconfig(self.canvas_image, image=self.photo_image)
    
    def next_digit(self):
        """Move to next digit"""
        self.current_digit += 1
        self.progress['value'] = len(self.rois)
        
        if self.current_digit <= self.num_digits:
            self.digit_label.config(text=f"Select Digit {self.current_digit} of {self.num_digits}")
            self.next_btn.config(state=tk.DISABLED)
            self.redo_btn.config(state=tk.DISABLED)
            self.start_point = None
            self.end_point = None
            self.update_display()
        
        if self.current_digit > self.num_digits:
            self.finish_btn.config(state=tk.NORMAL)
            self.digit_label.config(text="All digits selected!")
    
    def redo_current(self):
        """Redo current digit selection"""
        if self.current_digit in self.rois:
            del self.rois[self.current_digit]
        self.start_point = None
        self.end_point = None
        self.next_btn.config(state=tk.DISABLED)
        self.redo_btn.config(state=tk.DISABLED)
        self.update_display()
    
    def finish_setup(self):
        """Finish ROI setup"""
        if len(self.rois) < self.num_digits:
            messagebox.showwarning("Incomplete Setup", 
                                 f"Please select all {self.num_digits} digits.")
            return
        
        # Save configuration
        self.save_config()
        messagebox.showinfo("Success", "ROI configuration saved successfully!")
        self.root.quit()
        self.root.destroy()
    
    def cancel(self):
        """Cancel setup"""
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel?"):
            self.rois = {}
            self.root.quit()
            self.root.destroy()
    
    def save_config(self):
        """Save ROI configuration to file"""
        config = {
            'image_file': os.path.basename(self.image_path),
            'num_digits': self.num_digits,
            'rois': self.rois,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ ROI configuration saved to {CONFIG_FILE}")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()
        return self.rois

# ==================== MULTI-METHOD OCR ANALYZER ====================

class MultiMethodOCRAnalyzer:
    """Multi-method OCR with voting system - Conservative approach"""
    
    def __init__(self):
        self.methods = []
        self.method_names = []
        
        # Tesseract methods
        if TESSERACT_AVAILABLE:
            self.methods.extend([
                self.tesseract_psm10,
                self.tesseract_psm8,
                self.tesseract_psm7,
                self.tesseract_psm13,
            ])
            self.method_names.extend([
                "Tesseract-PSM10",
                "Tesseract-PSM8",
                "Tesseract-PSM7",
                "Tesseract-PSM13",
            ])
        
        # Fallback methods
        self.methods.extend([
            self.feature_analysis,
            self.template_matching,
            self.hole_detection,
        ])
        self.method_names.extend([
            "Feature-Analysis",
            "Template-Match",
            "Hole-Detection",
        ])
        
        print(f"✓ {len(self.methods)} OCR methods initialized")
    
    def analyze_digit(self, digit_img, digit_num):
        """Analyze digit using all available methods - Conservative approach"""
        if digit_img is None or digit_img.size == 0:
            return "?", 0.0, []
        
        results = []
        
        # Try multiple preprocessing strategies
        variants = self.create_preprocessing_variants(digit_img)
        
        for variant_name, processed_img in variants:
            for method_name, method_func in zip(self.method_names, self.methods):
                try:
                    result, confidence = method_func(processed_img)
                    if result and result != "?" and result in "0123456789":
                        results.append({
                            'method': f"{variant_name}+{method_name}",
                            'digit': result,
                            'confidence': confidence
                        })
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"      ⚠ {method_name} error: {str(e)[:50]}")
                    continue
        
        # Get consensus with voting - Conservative: require minimum 2 votes
        final_digit, final_conf, vote_details = self.get_consensus(results, min_votes=2)
        
        return final_digit, final_conf, vote_details
    
    def create_preprocessing_variants(self, digit_img):
        """Create multiple preprocessing variants"""
        if len(digit_img.shape) == 3:
            gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = digit_img.copy()
        
        h, w = gray.shape
        variants = []
        
        # Variant 1: Standard processing
        scaled = cv2.resize(gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary == 0) > np.sum(binary == 255):
            binary = cv2.bitwise_not(binary)
        padded = cv2.copyMakeBorder(binary, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        variants.append(("Standard", padded))
        
        # Variant 2: High contrast
        scaled2 = cv2.resize(gray, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
        clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced2 = clahe2.apply(scaled2)
        binary2 = cv2.adaptiveThreshold(enhanced2, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        if np.sum(binary2 == 0) > np.sum(binary2 == 255):
            binary2 = cv2.bitwise_not(binary2)
        padded2 = cv2.copyMakeBorder(binary2, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        variants.append(("HighContrast", padded2))
        
        # Variant 3: Enhanced processing
        scaled3 = cv2.resize(gray, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
        gamma = 0.4
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(scaled3, table)
        kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gamma_corrected, -1, kernel)
        _, binary3 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary3 == 0) > np.sum(binary3 == 255):
            binary3 = cv2.bitwise_not(binary3)
        padded3 = cv2.copyMakeBorder(binary3, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        variants.append(("Enhanced", padded3))
        
        return variants
    
    def get_consensus(self, results, min_votes=2):
        """Get consensus from voting - Conservative approach"""
        if not results:
            return "?", 0.0, []
        
        # Count votes
        votes = Counter([r['digit'] for r in results])
        
        if not votes:
            return "?", 0.0, []
        
        # Get digit with most votes
        best_digit, vote_count = votes.most_common(1)[0]
        
        # Conservative: Require minimum votes
        if vote_count < min_votes:
            if DEBUG_MODE:
                print(f"      ⚠ Insufficient consensus (only {vote_count} votes, need {min_votes})")
            return "?", 0.0, results
        
        # Calculate average confidence for winning digit
        digit_results = [r for r in results if r['digit'] == best_digit]
        avg_confidence = np.mean([r['confidence'] for r in digit_results])
        
        return best_digit, avg_confidence, results
    
    # ==================== TESSERACT METHODS ====================
    
    def tesseract_psm10(self, processed_img):
        """PSM 10: Single character"""
        try:
            pil_img = Image.fromarray(processed_img)
            config = '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(pil_img, config=config).strip()
            digit = re.sub(r'\D', '', text)
            return digit[0] if digit else "?", 0.90
        except:
            return "?", 0.0
    
    def tesseract_psm8(self, processed_img):
        """PSM 8: Single word"""
        try:
            pil_img = Image.fromarray(processed_img)
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(pil_img, config=config).strip()
            digit = re.sub(r'\D', '', text)
            return digit[0] if digit else "?", 0.85
        except:
            return "?", 0.0
    
    def tesseract_psm7(self, processed_img):
        """PSM 7: Single line"""
        try:
            pil_img = Image.fromarray(processed_img)
            config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(pil_img, config=config).strip()
            digit = re.sub(r'\D', '', text)
            return digit[0] if digit else "?", 0.80
        except:
            return "?", 0.0
    
    def tesseract_psm13(self, processed_img):
        """PSM 13: Raw line"""
        try:
            pil_img = Image.fromarray(processed_img)
            config = '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(pil_img, config=config).strip()
            digit = re.sub(r'\D', '', text)
            return digit[0] if digit else "?", 0.70
        except:
            return "?", 0.0
    
    # ==================== FALLBACK METHODS ====================
    
    def feature_analysis(self, binary_img):
        """Feature-based recognition"""
        try:
            h, w = binary_img.shape
            if h == 0 or w == 0:
                return "?", 0.0
            
            holes = self.count_holes(binary_img)
            aspect_ratio = w / h
            vertical_proj = np.sum(binary_img == 0, axis=0)
            
            if np.max(vertical_proj) > 0:
                vertical_peaks = np.sum(vertical_proj > np.max(vertical_proj) * 0.5)
            else:
                vertical_peaks = 0
            
            # Only strong rules
            if holes == 1:
                return "0", 0.6
            elif holes == 0 and vertical_peaks <= 2 and aspect_ratio < 0.4:
                return "1", 0.7
            
            return "?", 0.3
        except:
            return "?", 0.0
    
    def template_matching(self, binary_img):
        """Template matching for thin digits"""
        try:
            h, w = binary_img.shape
            vertical_proj = np.sum(binary_img < 128, axis=0)
            
            if np.max(vertical_proj) > 0:
                max_width = np.sum(vertical_proj > np.max(vertical_proj) * 0.5)
                if max_width < w * 0.3:
                    return "1", 0.6
            
            return "?", 0.2
        except:
            return "?", 0.0
    
    def hole_detection(self, binary_img):
        """Detect digits by hole count"""
        try:
            holes = self.count_holes(binary_img)
            
            if holes == 0:
                return "1", 0.5
            elif holes == 1:
                return "0", 0.5
            elif holes == 2:
                return "8", 0.5
            
            return "?", 0.3
        except:
            return "?", 0.0
    
    def count_holes(self, binary_img):
        """Count holes in binary image"""
        try:
            inverted = cv2.bitwise_not(binary_img)
            contours, hierarchy = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            hole_count = 0
            if hierarchy is not None:
                for i, hier in enumerate(hierarchy[0]):
                    if hier[3] != -1:
                        area = cv2.contourArea(contours[i])
                        if 10 < area < (binary_img.shape[0] * binary_img.shape[1] * 0.8):
                            hole_count += 1
            
            return hole_count
        except:
            return 0

# ==================== MAIN PROCESSING ====================

def read_meter_digits(image_path, digit_rois):
    """Read meter with multi-method voting - No assumptions"""
    print(f"\n🔍 Reading meter with multi-method OCR")
    print("-" * 50)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image '{image_path}'")
        return None, None, None, None
    
    print(f"✓ Image: {img.shape[1]}x{img.shape[0]}")
    print(f"✓ Processing {len(digit_rois)} digits...")
    
    ocr_analyzer = MultiMethodOCRAnalyzer()
    
    digits_text = []
    digit_images = []
    confidence_scores = []
    all_vote_details = []
    
    for digit_num in sorted(digit_rois.keys()):
        x, y, w, h = digit_rois[digit_num]
        digit_img = img[y:y+h, x:x+w]
        
        if digit_img.size == 0:
            print(f"  ⚠ Digit {digit_num}: Empty ROI")
            digits_text.append("?")
            digit_images.append(None)
            confidence_scores.append(0)
            all_vote_details.append([])
            continue
        
        # Analyze with all methods
        digit_char, confidence, vote_details = ocr_analyzer.analyze_digit(digit_img, digit_num)
        
        # Display voting results
        if DEBUG_MODE and vote_details:
            votes = Counter([r['digit'] for r in vote_details])
            print(f"    Vote results: {dict(votes)}")
            print(f"    → Result: '{digit_char}' (conf: {confidence:.2f})")
        
        digits_text.append(digit_char)
        digit_images.append(digit_img)
        confidence_scores.append(confidence)
        all_vote_details.append(vote_details)
        
        status = "✓" if confidence > 0.7 else "⚠" if confidence > 0.4 else "✗"
        print(f"  {status} Digit {digit_num}: '{digit_char}' (conf: {confidence:.2f})")
    
    # No contextual validation - report as detected
    full_text = "".join(digits_text)
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    print(f"\n✓ Detected:    {digits_text}")
    print(f"✓ Combined:    {full_text}")
    print(f"✓ Confidence:  {avg_confidence:.2f}")
    
    return full_text, digit_images, confidence_scores, all_vote_details

# ==================== VISUALIZATION ====================

def create_visualization(original_img, digit_rois, digits, 
                        confidence_scores, digit_images, final_value):
    """Create visualization"""
    fig = plt.figure(figsize=(18, 10))
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 1. Original image with ROIs
    ax1 = plt.subplot(3, 5, (1, 6))
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image with Digit ROIs', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    for i, (digit_num, (x, y, w, h)) in enumerate(sorted(digit_rois.items())):
        color = '#00FF00' if confidence_scores[i] > 0.7 else '#FFAA00'
        rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                edgecolor=color, facecolor='none', alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(x + w/2, y - 10, f'D{digit_num}', 
                color=color, fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 2. Result display
    ax2 = plt.subplot(3, 5, (2, 7))
    ax2.axis('off')
    
    result_text = f"""
╔{'═'*30}╗
║ WATER METER READING        ║
╠{'═'*30}╣
║                            ║
║    {final_value.center(20)}    ║
║                            ║
╠{'═'*30}╣
║ Multi-Method OCR Analysis  ║
║                            ║
"""
    
    for i in range(len(digits)):
        val = digits[i]
        conf = confidence_scores[i] if i < len(confidence_scores) else 0
        
        conf_emoji = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
        line = f"║  D{i+1}: {val}             {conf_emoji} ║"
        result_text += line + "\n"
    
    avg_conf = np.mean(confidence_scores) if confidence_scores else 0
    status = "✓ Complete" if "?" not in digits else "⚠ Partial"
    result_text += f"""╠{'═'*30}╣
║ Confidence: {avg_conf:.1%}           ║
║ Status: {status.ljust(15)} ║
╚{'═'*30}╝
"""
    
    ax2.text(0.05, 0.95, result_text, fontsize=9, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.3))
    
    # 3-9. Show digit crops
    for i in range(len(digit_images)):
        if i < 3:
            ax_idx = 3 + i
        elif i < 5:
            ax_idx = 8 + (i - 3)
        else:
            ax_idx = 13 + (i - 5)
        
        ax = plt.subplot(3, 5, ax_idx)
        
        if i < len(digit_images) and digit_images[i] is not None:
            if len(digit_images[i].shape) == 3:
                ax.imshow(cv2.cvtColor(digit_images[i], cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(digit_images[i], cmap='gray')
            
            title_text = f"D{i+1}: {digits[i]}"
            ax.set_title(title_text, fontsize=9, fontweight='bold')
            ax.axis('off')
            
            conf = confidence_scores[i] if i < len(confidence_scores) else 0
            conf_color = 'green' if conf > 0.7 else 'orange' if conf > 0.4 else 'red'
            ax.text(0.5, 0.05, f"{conf:.2f}", 
                   transform=ax.transAxes, fontsize=8, 
                   ha='center', color=conf_color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Water Meter OCR - Multi-Method Voting', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"meter_result_{timestamp}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    
    return output_file

# ==================== MAIN ====================

def load_or_setup_roi(image_file, num_digits=7):
    """Load ROI from config or setup new ROI"""
    
    # Check if config exists
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            # Check if config matches current image
            if config.get('image_file') == os.path.basename(image_file):
                print(f"✓ Loading ROI configuration from {CONFIG_FILE}")
                
                response = input("Use existing ROI configuration? (y/n): ").lower()
                if response == 'y':
                    rois = {int(k): tuple(v) for k, v in config['rois'].items()}
                    print(f"✓ Loaded {len(rois)} ROIs")
                    return rois
        except Exception as e:
            print(f"⚠ Error loading config: {e}")
    
    # Setup new ROI
    print("\n" + "="*70)
    print("ROI SETUP REQUIRED")
    print("="*70)
    print("Please select the digit regions in the GUI window.")
    print("Select from left to right in order.")
    print("-"*70)
    
    selector = ROISelector(image_file, num_digits)
    rois = selector.run()
    
    if not rois or len(rois) < num_digits:
        print("\n❌ ROI setup cancelled or incomplete")
        return None
    
    return rois

def main():
    """Run meter reading with GUI ROI setup"""
    print("=" * 70)
    print("WATER METER READER - MULTI-METHOD VOTING OCR")
    print("=" * 70)
    print("Conservative approach - No assumptions, report as detected")
    print("-" * 70)
    
    IMAGE_FILE = "R.jpg"
    NUM_DIGITS = 7
    
    if not os.path.exists(IMAGE_FILE):
        print(f"\n❌ Error: Image file '{IMAGE_FILE}' not found!")
        print("   Please make sure the image is in the same directory.")
        return
    
    # Load or setup ROI
    digit_rois = load_or_setup_roi(IMAGE_FILE, NUM_DIGITS)
    
    if not digit_rois:
        print("\n❌ Cannot proceed without ROI configuration")
        return
    
    # Read meter
    final_text, digit_imgs, conf_scores, vote_details = read_meter_digits(
        IMAGE_FILE, digit_rois
    )
    
    if final_text:
        # Format result (preserve ? for undetected digits)
        digits_list = list(final_text)
        
        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"✅ Raw reading:   {final_text}")
        
        # Check for undetected digits
        undetected_count = final_text.count('?')
        if undetected_count > 0:
            print(f"⚠  Warning:       {undetected_count} digit(s) undetected")
            print(f"   Please review and manually verify the reading")
        
        print(f"✓  Confidence:    {np.mean(conf_scores):.1%}")
        
        # Load original for visualization
        img = cv2.imread(IMAGE_FILE)
        if img is not None:
            create_visualization(
                img, digit_rois,
                digits_list,
                conf_scores,
                digit_imgs,
                final_text
            )
        
        print("\n✅ Processing complete!")
        print(f"   Result: {final_text}")
        
        # Save to file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("meter_reading.txt", "w") as f:
            f.write(f"Water Meter Reading\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Reading: {final_text}\n")
            f.write(f"Confidence: {np.mean(conf_scores):.1%}\n")
            if undetected_count > 0:
                f.write(f"Warning: {undetected_count} digit(s) undetected - manual verification required\n")
        print(f"✓  Saved to: meter_reading.txt")
        
    else:
        print("\n❌ Failed to read meter")

if __name__ == "__main__":
    main()