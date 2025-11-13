from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import io
import base64
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Constants for audio feature shapes
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Constants for image shapes
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3

# Load models
# audio_model = load_model('models/bird_audio_model.h5')  # Uncomment when available
image_model = tf.keras.models.load_model('models/bird_image_model.keras')

# Load label encoder classes
label_encoder_audio = LabelEncoder()
label_encoder_audio.classes_ = np.load('data/classes_audio.npy')
label_encoder_image = LabelEncoder()
label_encoder_image.classes_ = np.load('data/classes_image.npy')

# Pad or truncate feature to a fixed shape
def pad_or_truncate(feature, target_length):
    if len(feature) < target_length:
        feature = np.pad(feature, (0, target_length - len(feature)), 'constant')
    else:
        feature = feature[:target_length]
    return feature

# Feature extraction for audio
def preprocess_audio_file(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < SAMPLES_PER_TRACK:
            audio = pad_or_truncate(audio, SAMPLES_PER_TRACK)
        return audio.reshape(1, -1, 1)  # Shape for Conv1D
    except Exception as e:
        app.logger.error(f"Error preprocessing audio file: {e}")
        raise

# Preprocess the image file
def preprocess_image_file(image):
    try:
        image = image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
        image = np.array(image) / 255.0
        return image.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    except Exception as e:
        app.logger.error(f"Error preprocessing image file: {e}")
        raise

# Fetch bird details using Wikipedia API
def get_bird_details_from_wikipedia(bird_name):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{bird_name}"
        response = requests.get(url)
        app.logger.info(f"Wikipedia API response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            details = {
                'description': data.get('extract', 'No description available.'),
                'image': data.get('thumbnail', {}).get('source', '')
            }
            return details
        return None
    except Exception as e:
        app.logger.error(f"Error fetching bird details from Wikipedia: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image_route():
    if 'image_file' not in request.files or request.files['image_file'].filename == '':
        flash('Please upload an image file.')
        return redirect(url_for('index'))

    image_file = request.files['image_file']

    try:
        image = Image.open(image_file)
        preprocessed_image = preprocess_image_file(image)
        predictions = image_model.predict(preprocessed_image)
        predicted_index = np.argmax(predictions[0])

        if predicted_index >= len(label_encoder_image.classes_):
            raise ValueError(f"Predicted index {predicted_index} is out of bounds for label encoder classes")

        predicted_class = label_encoder_image.inverse_transform([predicted_index])[0]
        accuracy = predictions[0][predicted_index] * 100
        top_bird_details = get_bird_details_from_wikipedia(predicted_class)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return render_template('result_image.html', top_bird=(predicted_class, accuracy), top_bird_details=top_bird_details, image_data=img_str)
    except Exception as e:
        app.logger.error(f'Error processing image file: {e}')
        flash(f'Error processing image file: {e}')
        return redirect(url_for('index'))

@app.route('/predict_audio', methods=['POST','GET'])
def predict_audio_route():
    if 'audio_file' not in request.files or request.files['audio_file'].filename == '':
        flash('Please upload an audio file.')
        return redirect(url_for('index'))

    audio_file = request.files['audio_file']

    try:
        # Save the audio file temporarily
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join('static', 'temp_audio', filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        audio_file.save(temp_path)

        # Preprocess the audio
        features = preprocess_audio_file(temp_path)  # Define this function in your code
        predictions = audio_model.predict(features)
        predicted_index = np.argmax(predictions[0])

        if predicted_index >= len(label_encoder_audio.classes_):
            raise ValueError(f"Predicted index {predicted_index} is out of bounds for label encoder classes")

        predicted_class = label_encoder_audio.inverse_transform([predicted_index])[0]
        accuracy = predictions[0][predicted_index] * 100
        top_bird_details = get_bird_details_from_wikipedia(predicted_class)

        return render_template('result_audio.html', top_bird=(predicted_class, accuracy), top_bird_details=top_bird_details)

    except Exception as e:
        app.logger.error(f'Error processing audio file: {e}')
        flash(f'Error processing audio file: {e}')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/report')
def report():
    return render_template('report.html')  # Ensure this HTML file exists in templates

@app.route('/audio_page')
def audio_page():
    return render_template('audio_page.html')  # Make sure you have a template named audio.html

@app.route('/test')
def test():
    return 'This is working'


if __name__ == '__main__':
    app.run(debug=True)
