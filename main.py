import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from pushup_counter import PushupCounter

import pushup_counter as pc


class PushupCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Push-up Counter Fine-Tuning")

        self.video_path = None
        self.cap = None
        self.processing = False


        # Instantiate the counter
        self.pushup_counter = PushupCounter()

        # --- GUI Variables ---
        # Thresholds

        self.elbow_down_thresh = tk.DoubleVar(value=115)
        self.elbow_up_thresh = tk.DoubleVar(value=172)
        self.shoulder_down_thresh = tk.DoubleVar(value=90)
        self.shoulder_up_thresh = tk.DoubleVar(value=17)
        self.confidence_thresh = tk.DoubleVar(value=0.5)
        # Performance
        self.frame_skip = tk.IntVar(value=1) # Process every 1 frame by default


        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas = tk.Canvas(left_frame, width=640, height=480, bg='black')
        self.canvas.pack()

        info_frame = tk.Frame(left_frame)
        info_frame.pack(pady=10, fill=tk.X)

        self.counter_label = tk.Label(info_frame, text="Push-ups: 0", font=("Arial", 24))
        self.counter_label.pack(side=tk.LEFT, padx=20)

        self.status_label = tk.Label(info_frame, text="Status: -", font=("Arial", 18))
        self.status_label.pack(side=tk.LEFT, padx=20)

        right_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN)
        right_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        tk.Label(right_frame, text="Controls", font=("Arial", 16, "bold")).pack(pady=10)

        # General Controls
        control_frame = tk.Frame(right_frame, bd=1, relief=tk.GROOVE)
        control_frame.pack(pady=5, padx=10, fill=tk.X)
        self.load_button = tk.Button(control_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(pady=5, fill=tk.X)
        self.start_button = tk.Button(control_frame, text="Start Processing", command=self.toggle_processing, state=tk.DISABLED)
        self.start_button.pack(pady=5, fill=tk.X)

        # Performance Controls
        performance_frame = tk.Frame(right_frame, bd=1, relief=tk.GROOVE)
        performance_frame.pack(pady=10, padx=10)
        tk.Label(performance_frame, text="Performance", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        tk.Label(performance_frame, text="Process 1 every N frames:").grid(row=1, column=0, padx=5, pady=3, sticky='w')
        tk.Scale(performance_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.frame_skip).grid(row=1, column=1, padx=5, pady=3)

        # Threshold controls
        threshold_frame = tk.Frame(right_frame, bd=1, relief=tk.GROOVE)
        threshold_frame.pack(pady=10, padx=10)
        tk.Label(threshold_frame, text="Thresholds", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        tk.Label(threshold_frame, text="Elbow Down (<):").grid(row=1, column=0, padx=5, pady=3, sticky='w')
        tk.Scale(threshold_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.elbow_down_thresh).grid(row=1, column=1, padx=5, pady=3)
        tk.Label(threshold_frame, text="Elbow Up (>):").grid(row=2, column=0, padx=5, pady=3, sticky='w')
        tk.Scale(threshold_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.elbow_up_thresh).grid(row=2, column=1, padx=5, pady=3)
        tk.Label(threshold_frame, text="Shoulder Down (>):").grid(row=3, column=0, padx=5, pady=3, sticky='w')
        tk.Scale(threshold_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.shoulder_down_thresh).grid(row=3, column=1, padx=5, pady=3)
        tk.Label(threshold_frame, text="Shoulder Up (<):").grid(row=4, column=0, padx=5, pady=3, sticky='w')
        tk.Scale(threshold_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.shoulder_up_thresh).grid(row=4, column=1, padx=5, pady=3)

        tk.Label(threshold_frame, text="Pose Confidence (>):").grid(row=5, column=0, padx=5, pady=3, sticky='w')
        tk.Scale(threshold_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=self.confidence_thresh).grid(row=5, column=1, padx=5, pady=3)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.stop_processing()
            self.pushup_counter.reset()
            self.counter_label.config(text=f"Push-ups: {self.pushup_counter.counter}")
            self.status_label.config(text=f"Status: {self.pushup_counter.status}")


            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file.")
                return

            self.start_button.config(state=tk.NORMAL)
            self.show_first_frame()

    def show_first_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        else:
            self.cap.release()
            self.start_button.config(state=tk.DISABLED)

    def toggle_processing(self):
        if self.processing:
            self.stop_processing()
        else:
            self.start_processing()

    def start_processing(self):
        if self.video_path:
            self.processing = True
            self.start_button.config(text="Stop Processing")
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.video_path)
                self.pushup_counter.reset()
            self.frame_count = 0
            self.last_processed_frame = None

            self.update_frame()

    def stop_processing(self):
        self.processing = False
        self.start_button.config(text="Start Processing")
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        if not self.processing:
            return

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                frame = cv2.resize(frame, (640, 480))

                # Only process every Nth frame, where N is from the slider
                if self.frame_count % self.frame_skip.get() == 0:
                    processed_frame, count, status = self.pushup_counter.process_frame(
                        frame,
                        self.elbow_down_thresh.get(),
                        self.elbow_up_thresh.get(),
                        self.shoulder_down_thresh.get(),
                        self.shoulder_up_thresh.get(),
                        self.confidence_thresh.get()
                    )
                    self.last_processed_frame = processed_frame.copy()
                    self.counter_label.config(text=f"Push-ups: {count}")
                    self.status_label.config(text=f"Status: {status}")

                # Display the last processed frame to avoid flickering, or the current raw frame if none exists
                display_frame = self.last_processed_frame if self.last_processed_frame is not None else frame
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                self.root.after(33, self.update_frame) # Target ~30 FPS
            else:
                self.stop_processing()
                self.show_first_frame()


if __name__ == "__main__":
    root = tk.Tk()
    app = PushupCounterApp(root)
    root.mainloop()
