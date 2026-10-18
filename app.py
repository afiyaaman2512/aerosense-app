# AeroSense - Streamlit app (scikit-learn model on audio features, no TensorFlow needed)
# Run with:  streamlit run app.py
import os

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import features

st.set_page_config(page_title="AeroSense", page_icon="🫁", layout="centered")

MODEL_PATH = "aerosense_classic.joblib"      # produced by notebook cell 5B.6


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(f"Could not find '{MODEL_PATH}'. Put it in the same folder as app.py.")
    st.stop()

bundle = load_bundle()
model, scaler, class_names = bundle["model"], bundle["scaler"], bundle["class_names"]

# Safety check: the app's audio settings must match the ones used in training
if bundle["settings"] != features.SETTINGS:
    st.error("Audio settings in the saved model do not match features.py. Please retrain or fix features.py.")
    st.stop()

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
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Record a single cough")
        if recorded is not None:
            audio_file = recorded
    else:
        st.info("Microphone recording needs a newer Streamlit version.")

if audio_file is not None:
    st.audio(audio_file)

    audio_file.seek(0)
    vector = features.extract_handcrafted(audio_file)
    if vector is None:
        st.error("Could not read this recording, or it is silent. Please try another file.")
        st.stop()

    probs = model.predict_proba(scaler.transform(vector.reshape(1, -1)))[0]
    full = np.zeros(len(class_names))
    full[model.classes_] = probs                      # keep the class order of class_names
    best = int(np.argmax(full))

    st.subheader(f"Prediction: {class_names[best]}")
    st.metric("Model confidence", f"{full[best] * 100:.1f}%")
    if full[best] < 0.5:
        st.caption("Confidence is low - the model is unsure about this recording.")
    st.bar_chart(pd.DataFrame({"probability": full}, index=class_names))

    # Show what the recording looks like
    audio_file.seek(0)
    audio = features.load_for_display(audio_file)
    mel = librosa.feature.melspectrogram(y=audio, sr=features.SR, n_fft=features.N_FFT,
                                         hop_length=features.HOP, n_mels=64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].plot(np.arange(len(audio)) / features.SR, audio, linewidth=0.7)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    img = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=features.SR,
                                   hop_length=features.HOP, x_axis="time", y_axis="mel",
                                   ax=axes[1], cmap="magma")
    axes[1].set_title("Log-Mel spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    plt.tight_layout()
    st.pyplot(fig)
