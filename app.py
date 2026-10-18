# AeroSense - Streamlit app (scikit-learn model on audio features, no TensorFlow needed)
# Run with:  streamlit run app.py
import hashlib
import io
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


def run_analysis(audio_bytes, signature):
    """Extract features from the audio, predict, and return everything needed to show the result."""
    vector = features.extract_handcrafted(io.BytesIO(audio_bytes))
    if vector is None:
        return {"sig": signature, "error": "Could not read this recording, or it is silent. Please try another file."}

    probs = model.predict_proba(scaler.transform(vector.reshape(1, -1)))[0]
    full = np.zeros(len(class_names))
    full[model.classes_] = probs                       # keep the class order of class_names

    audio = features.load_for_display(io.BytesIO(audio_bytes))
    mel = librosa.feature.melspectrogram(y=audio, sr=features.SR, n_fft=features.N_FFT,
                                         hop_length=features.HOP, n_mels=64)
    return {"sig": signature, "probs": full, "audio": audio,
            "log_mel": librosa.power_to_db(mel, ref=np.max)}


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
st.title("🫁 AeroSense")
st.write("Upload or record a short cough sound, then press **Analyze audio**. The model estimates which group "
         "it most resembles: " + ", ".join(f"**{c}**" for c in class_names) + ".")
st.warning("Educational prototype only. This is **not** a medical device and must not be used "
           "to diagnose or rule out any disease.")

# Only ONE input is active at a time, so an old recording can never override a new upload.
source = st.radio("Input method", ["Upload a file", "Record with microphone"], horizontal=True)

audio_bytes = None
if source == "Upload a file":
    uploaded = st.file_uploader("Cough recording (wav / flac / ogg)", type=["wav", "flac", "ogg"], key="uploader")
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
else:
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Record a single cough", key="recorder")
        if recorded is not None:
            audio_bytes = recorded.getvalue()
    else:
        st.info("Microphone recording needs a newer Streamlit version (pip install -U streamlit).")

if audio_bytes is None:
    st.session_state.pop("result", None)               # nothing selected -> forget old result
else:
    signature = hashlib.md5(audio_bytes).hexdigest()   # identifies THIS exact audio
    st.audio(audio_bytes)

    if st.button("Analyze audio", type="primary"):
        with st.spinner("Analyzing..."):
            st.session_state["result"] = run_analysis(audio_bytes, signature)

    result = st.session_state.get("result")
    if result is not None and result["sig"] != signature:
        st.session_state.pop("result", None)           # result belongs to a different audio -> drop it
        result = None
        st.info("New audio selected - press **Analyze audio** to get its result.")
    elif result is None:
        st.info("Press **Analyze audio** to get a prediction.")

    if result is not None:
        if "error" in result:
            st.error(result["error"])
        else:
            probs = result["probs"]
            best = int(np.argmax(probs))
            st.subheader(f"Prediction: {class_names[best]}")
            st.metric("Model confidence", f"{probs[best] * 100:.1f}%")
            if probs[best] < 0.5:
                st.caption("Confidence is low - the model is unsure about this recording.")
            st.bar_chart(pd.DataFrame({"probability": probs}, index=class_names))

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
            audio = result["audio"]
            axes[0].plot(np.arange(len(audio)) / features.SR, audio, linewidth=0.7)
            axes[0].set_title("Waveform")
            axes[0].set_xlabel("Time (s)")
            img = librosa.display.specshow(result["log_mel"], sr=features.SR, hop_length=features.HOP,
                                           x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
            axes[1].set_title("Log-Mel spectrogram")
            fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)