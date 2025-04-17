# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
CAMERA_FOLDER = 'static/camera'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAMERA_FOLDER, exist_ok=True)

model = load_model("model.h5")

# Preprocessing
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img):
    return cv2.equalizeHist(img)
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vechiles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vechiles',
        'Vechiles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
        'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
        'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vechiles over 3.5 metric tons'
    ]
    return classes[classNo] if classNo < len(classes) else "Unknown"

def predict_img(img):
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    confidence = np.max(predictions)
    classIndex = np.argmax(predictions)

    if confidence > 0.50:
        return getClassName(classIndex), confidence
    else:
        return "Unknown", confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    f = request.files['file']
    filename = secure_filename(f.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(file_path)

    img = cv2.imread(file_path)
    prediction, confidence = predict_img(img)

    return render_template('result.html', image_path=file_path, prediction=prediction, confidence=round(confidence*100, 2))

@app.route('/camera')
def camera():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    
    if not ret:
        return "Failed to grab frame from webcam"

    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(CAMERA_FOLDER, filename)
    cv2.imwrite(file_path, frame)

    prediction, confidence = predict_img(frame)

    return render_template('result.html', image_path=file_path, prediction=prediction, confidence=round(confidence*100, 2))

if __name__ == '__main__':
    app.run(debug=True)