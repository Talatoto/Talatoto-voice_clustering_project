# Voice Clustering (Single-Mic, FSDD → K-Means)

Unsupervised **speaker clustering** pipeline built from the Free Spoken Digit Dataset (FSDD).  
Flow: **Preprocess → VAD → Embeddings (ECAPA-TDNN) → K-Means → Streamlit check**.

> Works with your current structure under `output_recordings\` and uses **K-Means** (set k).

---

## Repo Structure (as committed)

```
.
├─ enroll/
│  ├─ george/
│  ├─ jackson/
│  └─ lucas/
├─ output_recordings/
│  ├─ app_check.py                 # Streamlit UI to listen & review clusters
│  ├─ clusters_k.csv               # K-Means results (generated)
│  ├─ session.wav                  # stitched one-mic session (input)
│  ├─ session_16k.wav              # preprocessed audio (generated)
│  ├─ clean/                       # VAD segments (generated)
│  └─ embeddings/                  # embeddings + manifest.csv (generated)
├─ recordings/                      # raw FSDD wavs you downloaded
├─ preprocessing_Voice.py           # step 1: resample mono 16k + normalize
├─ session.py                       # (optional) stitch multiple wavs into session.wav
├─ vad_split.py                     # step 2: WebRTC VAD
├─ embeddings.py                    # step 3: ECAPA embeddings
└─ cluster_kmeans.py                # step 4: K-Means clustering
```

---

## Requirements

- **Python** 3.9–3.12  
- **ffmpeg** on PATH  
  - Windows (Admin PowerShell): `choco install ffmpeg`  
  - macOS: `brew install ffmpeg`  
  - Ubuntu: `sudo apt-get install ffmpeg`

### Python packages

```bash
pip install --upgrade pip
pip install numpy scipy pandas librosa soundfile pydub webrtcvad scikit-learn torch torchaudio speechbrain faster-whisper matplotlib streamlit
```

> On first run, `speechbrain` will download the pretrained ECAPA-TDNN model.

---

## Quick Start (Windows paths shown)

> Open PowerShell in the repo root.

### 0) (Optional) Virtual env
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

### 1) Preprocess → `session_16k.wav`
```powershell
python preprocessing_Voice.py
# Input : output_recordings\session.wav
# Output: output_recordings\session_16k.wav
```

> If your `session.wav` is already clean, this is still required to ensure **mono 16 kHz + RMS normalize**.

### 2) VAD → speech segments in `clean\`
```powershell
python vad_split.py
# Reads : output_recordings\session_16k.wav
# Writes: output_recordings\clean\session_seg0000.wav ...
```

**Tuning inside `vad_split.py`:**
- `AGGRESSIVENESS = 1..3` (higher = stricter speech)  
- `FRAME_MS = 20` (finer) or `30` (stable)  
- `PADDING_MS = 120..400` (hangover; lower splits more)  
- `MIN_DUR = 0.5..1.0` sec (avoid tiny blips)

### 3) Embeddings → `.npz` + `manifest.csv`
```powershell
python embeddings.py
# Reads : output_recordings\clean\*.wav
# Writes: output_recordings\embeddings\*.npz  + manifest.csv
```

### 4) K-Means clustering → `clusters_k.csv`
```powershell
python cluster_kmeans.py
# Writes: output_recordings\clusters_k.csv
```

- Set **k** inside `cluster_kmeans.py` (e.g., 2, 3, 4).  
- The script prints a **silhouette (cosine)** score (higher ≈ better).  
- (Optional) Add L2-norm before K-Means for cosine-like behavior:
  ```python
  X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
  ```

### 5) Review clusters in the browser (listen to clips)
```powershell
cd output_recordings
streamlit run .\app_check.py
```
- In the text box, paste: `C:\Users\<you>\...\output_recordings\clusters_k.csv`  
- You’ll see:
  - **Counts per cluster**
  - **PCA** scatter of embeddings
  - Audio players to listen inside each cluster

---

## What Each Script Does

- **`preprocessing_Voice.py`**  
  Loads `session.wav` → **mono 16 kHz** → **RMS normalize** → saves `session_16k.wav`.

- **`vad_split.py`**  
  WebRTC VAD over 20–30 ms frames + hysteresis → writes **speech-only** segments into `clean\`.

- **`embeddings.py`**  
  Uses **SpeechBrain ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) to create a **speaker embedding** per segment → saves `.npz` and `manifest.csv`.

- **`cluster_kmeans.py`**  
  Loads all embeddings from `manifest.csv`, runs **K-Means** with your **k**, writes `clusters_k.csv`.

- **`app_check.py`**  
  Small **Streamlit** app to visualize and listen per cluster.

---

## Tips & Troubleshooting

- **Only 1 segment from VAD**  
  Lower `PADDING_MS` (e.g., 120), set `FRAME_MS=20`, raise `AGGRESSIVENESS=3`, or rebuild the session with longer gaps.

- **K-Means error `n_samples=1 >= n_clusters`**  
  You don’t have enough segments. Adjust VAD as above or reduce `k`.

- **`manifest.csv` Permission denied**  
  Close it in Excel/Notepad, delete it, re-run `embeddings.py`.

- **ffmpeg not found**  
  Install ffmpeg and restart the terminal session.

- **Clusters mixing speakers**  
  Try L2-normalizing embeddings (see above) and/or test nearby k values (2–4) and pick the best **silhouette + ear check**.

---

