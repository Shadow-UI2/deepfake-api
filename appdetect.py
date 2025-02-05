# Install required modules
def install_packages():
    required_packages = ["opencv-python", "numpy", "tensorflow"]
    for package in required_packages:
        try:
            __import__(package.split('-')[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install_packages()
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
import cv2
import numpy as np
from model import Meso4
from tkinter import messagebox


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
        result_label.config(text="Processing...", fg="cyan")
        root.update_idletasks()
        result = detect_deepfake(file_path)
        result_label.config(text=f"Result: {result}", fg="lime" if "Real" in result else "red")
        messagebox.showinfo("Detection Result", result)

def on_hover(event):
    event.widget.config(bg="#f39c12", fg="white")

def on_leave(event):
    event.widget.config(bg="#2980b9", fg="white")


root = tk.Tk()
root.title("Deepfake Detection App")
root.geometry("700x600")
root.configure(bg="#2c3e50") 


canvas = Canvas(root, width=700, height=250, bg="#2c3e50", highlightthickness=0)
canvas.pack()


canvas.create_rectangle(10, 10, 690, 240, outline="#16a085", width=6, fill="#34495e")
canvas.create_text(350, 70, text="Deepfake Detection App", font=("Helvetica", 32, "bold"), fill="#ecf0f1")
canvas.create_text(350, 140, text="Select a video file to analyze deepfakes", font=("Helvetica", 16), fill="#1abc9c")


select_button = Button(root, text="Select Video", font=("Helvetica", 18, "bold"), bg="#2980b9", fg="white", command=open_file, relief="raised", padx=15, pady=10, bd=3)
select_button.bind("<Enter>", on_hover)
select_button.bind("<Leave>", on_leave)
select_button.pack(pady=40)

result_frame = tk.Frame(root, bg="#34495e", highlightbackground="#2980b9", highlightthickness=2, padx=10, pady=10, relief="solid", bd=5, borderwidth=3)
result_frame.pack(pady=40)

result_label = Label(result_frame, text="Awaiting Video...", font=("Helvetica", 18, "italic"), bg="#34495e", fg="#e74c3c")
result_label.pack()


footer_label = Label(root, text="Developed by Soul Society", font=("Helvetica", 14), bg="#2c3e50", fg="#2980b9")
footer_label.pack(side="bottom", pady=15)


root.mainloop()
