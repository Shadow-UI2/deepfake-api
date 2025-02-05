import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from model import Meso4

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 


model = Meso4()
model.load_weights(r"C:\Users\Shadow\Desktop\Final\Meso4_DF.h5")


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
            if prediction > 0.49: #threshold value
                fake_score += 1
            else:
                real_score += 1

        frame_count += 1
    cap.release()

    total_frames = fake_score + real_score
    if total_frames == 0:
        return "No frames processed"

    fake_probability = (fake_score / total_frames) * 100
    if fake_probability <= 5:
        return "Real Video"
    elif 5 < fake_probability <= 15:
        return "Real Video or Unidentifiable Deepfake (False Positive)"
    elif 15 < fake_probability <= 30:
        return "Animation or Advanced Deepfake Video"
    else:
        return "Deepfake Video"

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# File upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('result.html', result_message="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', result_message="No selected file.")


    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    try:
        file.save(file_path)
    except PermissionError:
        return render_template('result.html', result_message="Permission error: Unable to save file.")


    result = detect_deepfake(file_path)
    return render_template('result.html', result_message=result)


if __name__ == '__main__':
    app.run(debug=True)
