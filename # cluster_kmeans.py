# cluster_kmeans.py
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

emb_dir = Path(r"C:\Users\Tala\Downloads\output_recordings\embeddings")
out_csv = Path(r"C:\Users\Tala\Downloads\output_recordings\clusters_k.csv")
k = 3 

man = pd.read_csv(emb_dir / "manifest.csv")
X = np.vstack([np.load(p)["embedding"] for p in man["embedding"]])

km = KMeans(n_clusters=k, n_init=10, random_state=0)
labels = km.fit_predict(X)

out = man.copy()
out["cluster"] = labels
out.to_csv(out_csv, index=False)

# quick quality hint (higher ~ better separation)
if len(set(labels)) > 1:
    print("silhouette (cosine):", silhouette_score(X, labels, metric="cosine"))
print("wrote:", out_csv, "| #clusters:", len(set(labels)))

