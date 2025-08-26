# embeddings.py
from pathlib import Path
import numpy as np, pandas as pd
import torch, torchaudio
from speechbrain.pretrained import EncoderClassifier

in_dir  = Path(r"C:\Users\Tala\Downloads\output_recordings\clean")       # VAD output
out_dir = Path(r"C:\Users\Tala\Downloads\output_recordings\embeddings")  # will be created
out_dir.mkdir(parents=True, exist_ok=True)

MODEL = "speechbrain/spkrec-ecapa-voxceleb"

def load_mono16k(path):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:             # to mono
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:                   
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav, sr

def embed(clf, wav):
    with torch.no_grad():
        emb = clf.encode_batch(wav).squeeze().cpu().numpy()
    return emb

def main():
    clf = EncoderClassifier.from_hparams(source=MODEL, run_opts={"device": "cpu"})
    rows = []
    for wav_path in sorted(in_dir.glob("*.wav")):
        wav, sr = load_mono16k(wav_path)
        e = embed(clf, wav)
        npz = out_dir / (wav_path.stem + ".npz")
        np.savez_compressed(npz, embedding=e)
        rows.append({"file": str(wav_path), "embedding": str(npz)})
        print("embedded:", wav_path.name, "->", npz.name)
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print("wrote manifest:", out_dir / "manifest.csv")

if __name__ == "__main__":
    main()

