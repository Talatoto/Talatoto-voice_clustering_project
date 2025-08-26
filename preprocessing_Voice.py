from pathlib import Path
import librosa, soundfile as sf
import numpy as np

TARGET_SR = 16000

def rms_normalize(y, target_db=-20.0):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    gain = 10 ** (target_db / 20) / (rms + 1e-12)
    y = y * gain
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak
    return y

in_file = Path(r"C:\Users\Tala\Downloads\output_recordings\session.wav")
out_file = Path(r"C:\Users\Tala\Downloads\output_recordings\session_16k.wav")

y, sr = librosa.load(str(in_file), sr=TARGET_SR, mono=True)
y = rms_normalize(y, target_db=-20.0)
sf.write(out_file, y, TARGET_SR, subtype="PCM_16")

print("wrote:", out_file)

