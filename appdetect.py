import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
import cv2
import numpy as np
from model import Meso4
from tkinter import messagebox

# Install required modules
def install_packages():
    required_packages = ["opencv-python", "numpy", "tensorflow"]
    for package in required_packages:
        try:
            __import__(package.split('-')[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

# Load the model
model = Meso4()
model.load_weights("Meso4_DF.h5")

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frame = cv2.resize(frame, (256, 256))
    frame = np.array(frame, dtype=np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_score = 0
    real_score = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)

            if prediction > 0.48725:
                fake_score += 1
            else:
                real_score += 1

        frame_count += 1

    cap.release()

    total_frames = fake_score + real_score
    if total_frames == 0:
        return "No frames processed"

    fake_probability = (fake_score / total_frames) * 100

    if fake_probability <= 10:
        return "Real Video"
    elif 10 < fake_probability < 25:
        return "Real Video or Unidentifiable Deepfake (False Positive)"
    elif 25 <= fake_probability < 50:
        return "Animation or Advanced Deepfake Video"
    else:
        return "Deepfake Video"

def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if file_path:
        result_label.config(text="Processing...", fg="blue")
        root.update_idletasks()
        result = detect_deepfake(file_path)
        result_label.config(text=f"Result: {result}", fg="green" if "Real" in result else "red")
        messagebox.showinfo("Detection Result", result)

def on_hover(event):
    event.widget.config(bg="#5a9bd4", fg="white")

def on_leave(event):
    event.widget.config(bg="#4682b4", fg="white")

# Create the GUI application
root = tk.Tk()
root.title("Deepfake Detection App")
root.geometry("600x500")
root.configure(bg="#f0f8ff")

# Add a canvas for graphics
canvas = Canvas(root, width=600, height=150, bg="#f0f8ff", highlightthickness=0)
canvas.pack()

# Draw decorative borders on the canvas
canvas.create_rectangle(10, 10, 590, 140, outline="#4682b4", width=4)
canvas.create_line(10, 75, 590, 75, fill="#5f9ea0", width=2)

# Add header text to the canvas
canvas.create_text(300, 40, text="Deepfake Detection App", font=("Verdana", 24, "bold"), fill="#4682b4")
canvas.create_text(300, 100, text="Select a video file to analyze deepfakes", font=("Verdana", 14), fill="#5f9ea0")

# Add widgets
select_button = Button(root, text="Select Video", font=("Verdana", 14, "bold"), bg="#4682b4", fg="white", command=open_file, relief="raised", padx=10, pady=5, bd=3)
select_button.bind("<Enter>", on_hover)
select_button.bind("<Leave>", on_leave)
select_button.pack(pady=20)

result_frame = tk.Frame(root, bg="#f0f8ff", highlightbackground="#4682b4", highlightthickness=2, padx=10, pady=10)
result_frame.pack(pady=20)

result_label = Label(result_frame, text="", font=("Verdana", 14, "italic"), bg="#f0f8ff", fg="#ff4500")
result_label.pack()

# Run the app
root.mainloop()
