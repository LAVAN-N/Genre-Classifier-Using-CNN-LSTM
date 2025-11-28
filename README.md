# 🎵 Music Genre Classifier

### *Deep Learning–Based CNN + LSTM Hybrid Model for Audio Genre Recognition*

This repository contains a complete deep-learning pipeline for **automatic music genre classification**, leveraging a hybrid **CNN + LSTM** architecture designed to capture both spatial and temporal features of audio signals.

Trained on the widely-used **GTZAN Dataset**, the system accurately recognizes 10 music genres and includes a full **Streamlit web application** for real-time prediction.

---

## 🚀 Features

### 🎧 **Hybrid CNN + LSTM Architecture**

* CNN extracts **frequency & timbre patterns** from Mel-Spectrograms
* LSTM processes **temporal sequences** to model rhythm and progression
* Combines the strengths of both for superior genre accuracy

### 📊 **End-to-End Audio Processing Pipeline**

* Mel-Spectrogram generation
* MFCC extraction
* Audio trimming & normalization
* Input shaping for deep learning models

### 🌐 **Streamlit Web App for Real-Time Inference**

* Upload an audio file
* Visualize spectrogram
* Receive genre prediction with **confidence scores**

### 📦 **Pretrained Models Included**

* Load models directly from `saved_models/`
* No training required to use the app

### 🧩 **Modular Codebase**

* Clean, scalable folder structure
* Training, preprocessing, and inference separated for clarity

---

## 🎼 Dataset — GTZAN

The **GTZAN Dataset** is the benchmark dataset for music genre classification.

* **10 Genres**
  *Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock*
* **100 audio clips per genre**
* **30-second WAV format**
* Balanced class distribution
* High-quality examples across genres

Dataset Link:
[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## 🧠 Model Architecture

### 🟦 **CNN Module**

* Extracts spatial audio features from spectrograms
* Learns patterns such as:

  * Timbre
  * Harmonic content
  * Frequency bands

### 🟩 **LSTM Module**

* Models long-term temporal behavior
* Captures:

  * Rhythm
  * Groove
  * Repetitive structures
  * Progressions

### 🔧 **Training Details**

* **200 epochs**
* **Loss:** Categorical Crossentropy
* **Optimizer:** Adam
* Train/Val/Test split handled automatically
* Checkpoints & final models saved in `saved_models/`

---

## 📁 Repository Structure

```
Genre-Classifier-Using-CNN-LSTM/
│── app.py                # Streamlit web app
│── models/               # Model architectures (CNN, LSTM)
│── preprocessing/        # MFCC, spectrogram generation
│── saved_models/         # Pretrained .keras models
│── utils/                # Helper utilities
│── requirements.txt
│── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/LAVAN-N/Genre-Classifier-Using-CNN-LSTM.git
cd Genre-Classifier-Using-CNN-LSTM
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv env
source env/bin/activate      # macOS/Linux
env\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 🎛️ Making Predictions

### 🖥️ **Using the Streamlit Web App**

1. Open the UI
2. Upload any 30-second WAV file
3. View:

   * Predicted genre
   * Confidence scores
   * Spectrogram visualization

### 🐍 **Using Python**

```python
from tensorflow.keras.models import load_model

model = load_model("saved_models/genre_classification.keras")
pred = model.predict(audio_data)
```

---

## 📦 Pretrained Models

All pretrained weights are stored inside:

```
saved_models/
```

You can load them instantly for inference — no training required.

---

## 🔮 Future Enhancements

* 🎙️ Real-time audio recording in browser
* 🔊 Transformer-based audio embeddings
* 🎼 Multi-genre tagging
* 🕒 Temporal CNNs for improved rhythm modeling
* 📈 Improved visualization dashboard

---

## 🧰 Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* Librosa
* NumPy & Pandas
* GTZAN Dataset

---

## 📚 References

* GTZAN Dataset
  [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
* TensorFlow Documentation
  [https://www.tensorflow.org/](https://www.tensorflow.org/)
* Streamlit Documentation
  [https://streamlit.io/](https://streamlit.io/)

---


## 🤝 Contributing

Pull requests are welcome!

---

## 🛡️ License

MIT License

---

## ⭐ If you like this project

Please **star the repository** — it helps a lot!