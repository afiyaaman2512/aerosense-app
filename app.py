# AeroSense - Streamlit app for cough-based respiratory sound classification
# Run with:  streamlit run app.py
import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="AeroSense", page_icon="🫁", layout="centered")

MODEL_PATH = "aerosense_cnn.keras"      # produced by the training notebook
CONFIG_PATH = "model_config.json"       # produced by the training notebook (cell 5.4)


# ---------------------------------------------------------------------------
# Load the trained model once (cached, so it is not reloaded on every click)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_config():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, config


if not (os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH)):
    st.error(f"Could not find '{MODEL_PATH}' and '{CONFIG_PATH}'. "
             "Place both files in the same folder as app.py.")
    st.stop()

model, cfg = load_model_and_config()
class_names = cfg["class_names"]
sr = cfg["sample_rate"]
n_samples = int(sr * cfg["duration"])


# ---------------------------------------------------------------------------
# Preprocessing: must be IDENTICAL to the training notebook
# ---------------------------------------------------------------------------
def preprocess(audio_file):
    """Return (audio, log_mel, model_input) for an uploaded/recorded audio file."""
    audio_file.seek(0)
    audio, _ = librosa.load(audio_file, sr=sr, mono=True)      # load + resample to 16 kHz
    audio, _ = librosa.effects.trim(audio, top_db=30)          # remove silence
    if len(audio) == 0:
        return None, None, None
    audio = audio / (np.max(np.abs(audio)) + 1e-9)             # normalise volume
    if len(audio) < n_samples:                                 # pad / crop to fixed length
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=cfg["n_fft"],
                                         hop_length=cfg["hop_length"], n_mels=cfg["n_mels"])
    log_mel = librosa.power_to_db(mel, ref=np.max)

    x = (log_mel - cfg["norm_mean"]) / cfg["norm_std"]         # same normalisation as training
    x = x[np.newaxis, ..., np.newaxis].astype(np.float32)      # shape (1, mels, frames, 1)
    return audio, log_mel, x


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
st.title("🫁 AeroSense")
st.write("Upload or record a short cough sound. The model estimates which group it most resembles: "
         + ", ".join(f"**{c}**" for c in class_names) + ".")
st.warning("Educational prototype only. This is **not** a medical device and must not be used "
           "to diagnose or rule out any disease.")

audio_file = None
tab_upload, tab_record = st.tabs(["Upload a file", "Record with microphone"])

with tab_upload:
    uploaded = st.file_uploader("Cough recording (wav / flac / ogg)", type=["wav", "flac", "ogg"])
    if uploaded is not None:
        audio_file = uploaded

with tab_record:
    if hasattr(st, "audio_input"):                      # available in recent Streamlit versions
        recorded = st.audio_input("Record a single cough")
        if recorded is not None:
            audio_file = recorded
    else:
        st.info("Microphone recording needs a newer Streamlit version (pip install -U streamlit).")

if audio_file is not None:
    st.audio(audio_file)
    try:
        audio, log_mel, x = preprocess(audio_file)
    except Exception as e:
        st.error(f"Could not read this audio file: {e}")
        st.stop()

    if audio is None:
        st.error("The recording seems to be silent. Please try again.")
        st.stop()

    probs = model.predict(x, verbose=0)[0]
    best = int(np.argmax(probs))

    st.subheader(f"Prediction: {class_names[best]}")
    st.metric("Model confidence", f"{probs[best] * 100:.1f}%")
    if probs[best] < 0.5:
        st.caption("Confidence is low - the model is unsure about this recording.")

    st.bar_chart(pd.DataFrame({"probability": probs}, index=class_names))

    # Show what the model "sees"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].plot(np.arange(len(audio)) / sr, audio, linewidth=0.7)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=cfg["hop_length"], x_axis="time",
                                   y_axis="mel", fmax=sr // 2, ax=axes[1], cmap="magma")
    axes[1].set_title("Log-Mel spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    plt.tight_layout()
    st.pyplot(fig)
