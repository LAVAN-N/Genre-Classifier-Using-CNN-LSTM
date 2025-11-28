import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import requests
import os

# Flask App Initialization
app = Flask(__name__)

# Load your trained model
model_path = "saved_models/genre_classification.keras"
model = load_model(model_path)

# Preprocessing function
def preprocess_audio(file):
    data, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=70)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    input_data = np.expand_dims(mfccs_scaled_features, axis=0)
    return input_data

@app.route("/classify", methods=["POST"])
def classify_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    temp_path = "temp_audio.wav"
    audio_file.save(temp_path)  # Save the audio file to a temporary path

    input_data = preprocess_audio(temp_path)

    # Clean up the temp file
    os.remove(temp_path)

    # Perform classification
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)[0]
    names=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    predicted_label = names[predicted_class]
    return jsonify({"result": f"{predicted_label}"})

# Start Flask in a separate thread
def start_flask():
    app.run(port=5000, debug=False, use_reloader=False)

flask_thread = Thread(target=start_flask)
flask_thread.daemon = True
flask_thread.start()

def main():
    st.title("Genre Classification")

    st.subheader("Upload a 30-second .wav audio file for classification")
    audio_file = st.file_uploader("Choose an audio file", type=["wav"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        if st.button("Classify"):
            # Send the file to the backend for processing
            files = {"audio": audio_file.getvalue()}
            response = requests.post("http://127.0.0.1:5000/classify", files=files)
            if response.status_code == 200:
                st.success(f"Classification Result: {response.json()['result']}")
            else:
                st.error("Error during classification")

if __name__ == "__main__":
    main()
