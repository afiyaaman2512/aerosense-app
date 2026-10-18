# Audio feature extraction for AeroSense.
# This is the SAME code as cell 5B.1 of the training notebook. Do not change it
# unless you also change the notebook and retrain, otherwise predictions will be wrong.
import librosa
import numpy as np

SETTINGS = {"sr": 16000, "duration": 3.0, "n_fft": 1024, "hop": 256}
SR, DURATION, N_FFT, HOP = SETTINGS["sr"], SETTINGS["duration"], SETTINGS["n_fft"], SETTINGS["hop"]
N_SAMPLES = int(SR * DURATION)


def extract_handcrafted(source):
    """source: a file path or a file-like object (wav/flac/ogg). Returns a 1-D feature vector, or None."""
    try:
        audio, _ = librosa.load(source, sr=SR, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio) == 0:
            return None
        cough_seconds = min(len(audio) / SR, DURATION)
        audio = audio[:N_SAMPLES]
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        if len(audio) < 4000:
            audio = np.pad(audio, (0, 4000 - len(audio)))

        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20, n_fft=N_FFT, hop_length=HOP)
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        extra = [
            librosa.feature.spectral_centroid(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP),
            librosa.feature.spectral_bandwidth(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP),
            librosa.feature.spectral_rolloff(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP),
            librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP),
            librosa.feature.zero_crossing_rate(audio, frame_length=N_FFT, hop_length=HOP),
            librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP),
            librosa.feature.spectral_contrast(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_bands=5),
        ]
        rows = np.vstack([mfcc, delta1, delta2] + extra)
        vector = np.concatenate([rows.mean(axis=1), rows.std(axis=1), [cough_seconds]])
        return np.nan_to_num(vector.astype(np.float32))
    except Exception:
        return None


def load_for_display(source):
    """Load the cough the same way (trim, normalise, fix length) for plotting only."""
    audio, _ = librosa.load(source, sr=SR, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=30)
    audio = audio[:N_SAMPLES]
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    return audio
