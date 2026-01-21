import cv2
import json
import os
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
import sys

# Default configuration
DEFAULT_IMAGE = "3.jpg"
DEFAULT_CONFIG = "roi_config.json"

class ROISelector:
    """Interactive ROI selector with GUI"""
    
    def __init__(self, image_path, output_config=DEFAULT_CONFIG, num_digits=7):
        self.image_path = image_path
        self.output_config = output_config
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
        self.root.title(f"ROI Setup - {os.path.basename(self.image_path)}")
        
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
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.output_config)), exist_ok=True)
        
        with open(self.output_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ ROI configuration saved to {self.output_config}")
        messagebox.showinfo("Success", f"ROI configuration saved to:\n{self.output_config}")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()
        return self.rois

def main():
    print("=" * 60)
    print("ROI CONFIGURATION TOOL")
    print("=" * 60)
    
    # Defaults
    image_file = DEFAULT_IMAGE
    config_file = DEFAULT_CONFIG
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    if len(sys.argv) > 2:
        config_file = sys.argv[2]
        
    # If using default but file doesn't exist, ask user
    if image_file == DEFAULT_IMAGE and not os.path.exists(image_file):
        print(f"Default image '{image_file}' not found.")
        
        # Try to find any jpg/png in current dir
        files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            print(f"Found images: {', '.join(files)}")
            choice = input(f"Use '{files[0]}'? (y/n): ")
            if choice.lower() == 'y':
                image_file = files[0]
            else:
                image_file = input("Enter image filename: ")
        else:
             image_file = input("Enter image filename: ")
    
    if not os.path.exists(image_file):
        print(f"❌ Error: Image file '{image_file}' not found!")
        return
        
    print(f"Image: {image_file}")
    print(f"Output: {config_file}")
    print("-" * 60)
    
    try:
        selector = ROISelector(image_file, config_file)
        rois = selector.run()
        
        if rois:
            print("\nSetup completed successfully.")
        else:
            print("\nSetup cancelled.")
            
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
