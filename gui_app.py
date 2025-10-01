import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from animal_classifier import AnimalClassifier
import os

class AnimalClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🐾 Animal Face Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f2f6')
        
        # Variables
        self.classifier = None
        self.current_image_path = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Thiết lập giao diện người dùng"""
        
        # ===== HEADER =====
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🐾 Animal Face Classifier", 
            font=("Arial", 24, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Nhận diện khuôn mặt động vật: Cat 🐱 | Dog 🐶 | Wild Animal 🦁",
            font=("Arial", 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()
        
        # ===== MAIN CONTENT =====
        main_frame = tk.Frame(self.root, bg='#f0f2f6')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Left Panel - Image Display
        left_panel = tk.LabelFrame(
            main_frame, 
            text="📷 Ảnh đầu vào", 
            font=("Arial", 14, "bold"),
            bg='#f0f2f6',
            fg='#2c3e50'
        )
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Image display area
        self.image_frame = tk.Frame(left_panel, bg='white', relief='sunken', bd=2)
        self.image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="📷\n\nClick 'Chọn ảnh' để bắt đầu\nphân loại động vật",
            font=("Arial", 16),
            bg='white',
            fg='#7f8c8d',
            justify='center'
        )
        self.image_label.pack(expand=True)
        
        # Right Panel - Controls and Results
        right_panel = tk.Frame(main_frame, bg='#f0f2f6')
        right_panel.pack(side='right', fill='y', padx=(10, 0))
        
        # ===== CONTROL SECTION =====
        control_frame = tk.LabelFrame(
            right_panel, 
            text="🎛️ Điều khiển", 
            font=("Arial", 14, "bold"),
            bg='#f0f2f6',
            fg='#2c3e50'
        )
        control_frame.pack(fill='x', pady=(0, 20))
        
        # Select image button
        self.select_btn = tk.Button(
            control_frame,
            text="📁 Chọn ảnh từ máy tính",
            command=self.select_image,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.select_btn.pack(fill='x', padx=10, pady=10)
        
        # Sample images section
        sample_frame = tk.LabelFrame(
            control_frame, 
            text="🖼️ Ảnh mẫu", 
            font=("Arial", 10, "bold"),
            bg='#f0f2f6'
        )
        sample_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        sample_buttons = [
            ("🐱 Cat", "animal-faces/afhq/val/cat", '#e67e22'),
            ("🐶 Dog", "animal-faces/afhq/val/dog", '#27ae60'), 
            ("🦁 Wild", "animal-faces/afhq/val/wild", '#8e44ad')
        ]
        
        for text, path, color in sample_buttons:
            btn = tk.Button(
                sample_frame,
                text=text,
                command=lambda p=path: self.load_sample_image(p),
                font=("Arial", 10),
                bg=color,
                fg='white',
                relief='flat',
                padx=10,
                pady=5,
                cursor='hand2'
            )
            btn.pack(fill='x', padx=5, pady=2)
        
        # Predict button
        self.predict_btn = tk.Button(
            control_frame,
            text="🔍 PHÂN LOẠI NGAY",
            command=self.predict_image,
            font=("Arial", 14, "bold"),
            bg='#e74c3c',
            fg='white',
            relief='flat',
            padx=20,
            pady=15,
            state='disabled',
            cursor='hand2'
        )
        self.predict_btn.pack(fill='x', padx=10, pady=10)
        
        # ===== RESULTS SECTION =====
        results_frame = tk.LabelFrame(
            right_panel, 
            text="📊 Kết quả phân loại", 
            font=("Arial", 14, "bold"),
            bg='#f0f2f6',
            fg='#2c3e50'
        )
        results_frame.pack(fill='both', expand=True)
        
        # Main result
        self.result_label = tk.Label(
            results_frame,
            text="Chưa có kết quả",
            font=("Arial", 18, "bold"),
            bg='#ecf0f1',
            fg='#2c3e50',
            relief='groove',
            padx=20,
            pady=20
        )
        self.result_label.pack(fill='x', padx=10, pady=10)
        
        # Confidence bars
        confidence_frame = tk.Frame(results_frame, bg='#f0f2f6')
        confidence_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.confidence_bars = {}
        animals = [('Cat 🐱', 'cat'), ('Dog 🐶', 'dog'), ('Wild 🦁', 'wild')]
        
        for display_name, key in animals:
            # Label
            label = tk.Label(
                confidence_frame,
                text=display_name,
                font=("Arial", 11, "bold"),
                bg='#f0f2f6',
                anchor='w'
            )
            label.pack(fill='x', pady=(5, 0))
            
            # Progress bar frame
            bar_frame = tk.Frame(confidence_frame, bg='#f0f2f6')
            bar_frame.pack(fill='x', pady=(2, 5))
            
            # Progress bar
            progress = ttk.Progressbar(
                bar_frame,
                length=200,
                mode='determinate',
                style='TProgressbar'
            )
            progress.pack(side='left', fill='x', expand=True)
            
            # Percentage label
            percent_label = tk.Label(
                bar_frame,
                text="0%",
                font=("Arial", 10),
                bg='#f0f2f6',
                width=6
            )
            percent_label.pack(side='right', padx=(5, 0))
            
            self.confidence_bars[key] = (progress, percent_label)
        
        # ===== STATUS BAR =====
        self.status_label = tk.Label(
            self.root,
            text="Đang khởi tạo...",
            font=("Arial", 10),
            bg='#34495e',
            fg='white',
            relief='sunken',
            anchor='w',
            padx=10
        )
        self.status_label.pack(side='bottom', fill='x')
    
    def load_model(self):
        """Load model classifier"""
        try:
            self.status_label.config(text="🔄 Đang tải model...")
            self.root.update()
            
            self.classifier = AnimalClassifier()
            
            self.status_label.config(text="✅ Model đã sẵn sàng! Chọn ảnh để bắt đầu.")
            self.select_btn.config(state='normal')
            
        except Exception as e:
            self.status_label.config(text="❌ Lỗi tải model")
            messagebox.showerror("Lỗi", f"Không thể tải model:\n{str(e)}")
    
    def select_image(self):
        """Chọn ảnh từ máy tính"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh động vật",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_sample_image(self, sample_dir):
        """Load ảnh mẫu từ thư mục"""
        try:
            if os.path.exists(sample_dir):
                # Lấy ảnh đầu tiên trong thư mục
                files = [f for f in os.listdir(sample_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    sample_path = os.path.join(sample_dir, files[0])
                    self.load_image(sample_path)
                else:
                    messagebox.showwarning("Cảnh báo", "Không tìm thấy ảnh trong thư mục mẫu")
            else:
                messagebox.showwarning("Cảnh báo", "Thư mục mẫu không tồn tại")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi tải ảnh mẫu: {str(e)}")
    
    def load_image(self, image_path):
        """Load và hiển thị ảnh"""
        try:
            # Load ảnh
            image = Image.open(image_path)
            self.current_image_path = image_path
            
            # Resize ảnh để hiển thị (maintain aspect ratio)
            display_size = (400, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert sang PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Hiển thị ảnh
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Enable predict button
            self.predict_btn.config(state='normal')
            
            # Reset results
            self.clear_results()
            
            filename = os.path.basename(image_path)
            self.status_label.config(text=f"📷 Đã tải ảnh: {filename}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")
    
    def predict_image(self):
        """Thực hiện prediction"""
        if not self.current_image_path or not self.classifier:
            return
        
        try:
            # UI feedback
            self.status_label.config(text="🔄 Đang phân tích ảnh...")
            self.predict_btn.config(text="Đang phân tích...", state='disabled')
            self.root.update()
            
            # Predict
            predicted_class, confidence = self.classifier.predict_image(self.current_image_path)
            all_scores = self.classifier.predict_with_all_scores(self.current_image_path)
            
            # Update main result
            icons = {"cat": "🐱", "dog": "🐶", "wild": "🦁"}
            icon = icons.get(predicted_class.lower(), "🐾")
            result_text = f"{icon} {predicted_class.upper()}"
            
            self.result_label.config(text=result_text)
            
            # Update confidence bars
            for animal, (progress, percent_label) in self.confidence_bars.items():
                score = all_scores[animal] * 100
                progress.config(value=score)
                percent_label.config(text=f"{score:.1f}%")
            
            # Color coding based on confidence
            if confidence > 0.8:
                bg_color = '#27ae60'  # Green
                status_msg = f"✅ Kết quả: {predicted_class.upper()} ({confidence:.1%}) - Độ tin cậy cao!"
            elif confidence > 0.6:
                bg_color = '#f39c12'  # Orange
                status_msg = f"⚠️ Kết quả: {predicted_class.upper()} ({confidence:.1%}) - Độ tin cậy trung bình"
            else:
                bg_color = '#e74c3c'  # Red
                status_msg = f"❗ Kết quả: {predicted_class.upper()} ({confidence:.1%}) - Độ tin cậy thấp"
            
            self.result_label.config(bg=bg_color, fg='white')
            self.status_label.config(text=status_msg)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi phân tích: {str(e)}")
            self.status_label.config(text="❌ Có lỗi xảy ra khi phân tích")
        
        finally:
            self.predict_btn.config(text="🔍 PHÂN LOẠI NGAY", state='normal')
    
    def clear_results(self):
        """Xóa kết quả cũ"""
        self.result_label.config(text="Sẵn sàng phân loại", bg='#ecf0f1', fg='#2c3e50')
        for progress, percent_label in self.confidence_bars.values():
            progress.config(value=0)
            percent_label.config(text="0%")

def main():
    """Chạy ứng dụng"""
    root = tk.Tk()
    app = AnimalClassifierGUI(root)
    
    # Icon và styling
    try:
        root.iconbitmap()  # Có thể thêm icon file nếu có
    except:
        pass
    
    # Center window
    root.eval('tk::PlaceWindow . center')
    
    root.mainloop()

if __name__ == "__main__":
    main()