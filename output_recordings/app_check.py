# app_check.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

st.set_page_config(page_title="Voice Clusters Checker", layout="wide")
st.title("Voice Clusters — Quick Checker")

# --- Paths (edit if needed) ---
default_csv = r"C:\Users\Tala\Downloads\output_recordings\clusters_k.csv"  # or clusters.csv
csv_path = st.text_input("Path to clusters CSV", value=default_csv)

if not Path(csv_path).exists():
    st.warning("CSV not found. Run clustering first or point to the correct file.")
    st.stop()

df = pd.read_csv(csv_path)
if "cluster" not in df.columns or "embedding" not in df.columns or "file" not in df.columns:
    st.error("CSV must have columns: file, embedding, cluster")
    st.stop()

# Sidebar — filters
st.sidebar.header("Filters")
clusters_sorted = sorted(df["cluster"].unique().tolist())
sel_clusters = st.sidebar.multiselect("Select clusters", clusters_sorted, default=clusters_sorted)
df_view = df[df["cluster"].isin(sel_clusters)].copy()

# Cluster counts
st.subheader("Counts per cluster")
counts = df["cluster"].value_counts().sort_index()
st.dataframe(counts.rename("count").to_frame())

# Embedding preview (2D PCA)
with st.expander("Embedding map (PCA)"):
    try:
        # load all embeddings
        E = np.vstack([np.load(p)["embedding"] for p in df["embedding"]])
        pca = PCA(n_components=2, random_state=0).fit_transform(E)
        plot_df = pd.DataFrame({"pc1": pca[:,0], "pc2": pca[:,1], "cluster": df["cluster"]})
        st.scatter_chart(plot_df, x="pc1", y="pc2", color="cluster", height=400)
        st.caption("Points close together should be same speaker. Colors = clusters.")
    except Exception as e:
        st.info(f"Could not load embeddings for PCA: {e}")

# Cluster browser + audio
st.subheader("Listen by cluster")
col1, col2 = st.columns([1,2])

with col1:
    pick = st.selectbox("Cluster", clusters_sorted, index=0)
    sub = df[df["cluster"] == pick].reset_index(drop=True)
    st.write(f"{len(sub)} segments in cluster {pick}")
    start_idx = st.number_input("Start index", 0, max(0, len(sub)-1), 0, step=1)

with col2:
    if len(sub := df[df["cluster"] == pick]) > 0:
        # show a few at a time
        page = sub.iloc[start_idx:start_idx+10]
        for _, row in page.iterrows():
            st.write(Path(row["file"]).name)
            try:
                st.audio(row["file"])  # Streamlit can play WAV from local path
            except Exception as e:
                st.text(f"(Could not play: {e})")
    else:
        st.info("No files in this cluster.")

# File table (optional)
with st.expander("All assignments"):
    st.dataframe(df_view[["file","cluster"]])
