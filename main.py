import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from datetime import datetime
import io

def compute_silhouette(X, labels):
    return silhouette_score(X, labels) if len(set(labels) - {-1}) > 1 else None

def compute_db(X, labels):
    return davies_bouldin_score(X, labels) if len(set(labels) - {-1}) > 1 else None

def compute_ch(X, labels):
    return calinski_harabasz_score(X, labels) if len(set(labels) - {-1}) > 1 else None

def compute_ari(y, labels):
    return adjusted_rand_score(y, labels)

def compute_nmi(y, labels):
    return normalized_mutual_info_score(y, labels)

def compute_hcv(y, labels):
    return homogeneity_completeness_v_measure(y, labels)

def run_acmk(X, k):
    def mk(X, k):
        runs, N = 10, X.shape[0]
        L = np.zeros((runs, N), int)
        for i in range(runs):
            km = KMeans(n_clusters=k, random_state=42 + i)
            km.fit(X)
            L[i] = km.labels_
        return L
    def build(L):
        runs, N = L.shape
        M = np.zeros((N, N))
        for r in range(runs):
            for i in range(N):
                for j in range(i + 1, N):
                    if L[r, i] == L[r, j]:
                        M[i, j] += 1
                        M[j, i] += 1
        M /= runs
        np.fill_diagonal(M, 1)
        return M
    def init_labels(M, k):
        dist = 1 - M
        np.clip(dist, 0, None, out=dist)
        np.fill_diagonal(dist, 0)
        Z = linkage(squareform(dist), method='average')
        return fcluster(Z, k, criterion='maxclust') - 1
    def refine(M, labs):
        N = M.shape[0]
        for _ in range(10):
            old = labs.copy()
            clusters = {c: np.where(labs == c)[0] for c in np.unique(labs)}
            for i in range(N):
                best, score = labs[i], -1
                for c, inds in clusters.items():
                    if len(inds) and np.mean(M[i, inds]) > score:
                        score, best = np.mean(M[i, inds]), c
                labs[i] = best
            if np.mean(old != labs) < 1e-4:
                break
        return labs
    L = mk(X, k)
    M = build(L)
    labs_init = init_labels(M, k)
    return refine(M, labs_init)

if 'history' not in st.session_state:
    st.session_state.history = []

st.set_page_config(layout="wide")
page = st.sidebar.selectbox("Page", ["Clustering", "Descriptions", "History"])
theta = st.sidebar.slider("Theta (θ)", 0.0, 5.0, 1.0, 0.1)
if page == "Clustering":
    algo = st.sidebar.selectbox("Algorithm", ["KMeans", "ACMK (Multi-KMeans)", "Graph Partitioning", "Evidence Accumulation", "USPEC"])
    k    = st.sidebar.slider("Clusters (k)", 2, 10, 4)
    n    = st.sidebar.slider("Points (n)", 100, 1000, 300, step=50)
    std  = st.sidebar.slider("Std (σ)", 0.1, 3.0, 2.0, step=0.1)
    if st.sidebar.button("Run"):
        X, y = make_blobs(n_samples=n, centers=k, cluster_std=std, random_state=42)
        Xs   = StandardScaler().fit_transform(X)
        if algo == "KMeans":
            labs = KMeans(n_clusters=k, random_state=42).fit_predict(Xs)
        elif algo == "ACMK (Multi-KMeans)":
            labs = run_acmk(Xs, k)
        elif algo in ("Evidence Accumulation", "Graph Partitioning"):
            runs = [KMeans(n_clusters=k, random_state=i).fit_predict(Xs) for i in range(5)]
            N = Xs.shape[0]
            M_flat = np.zeros((N, N))
            for lbl in runs:
                for i in range(N):
                    for j in range(i + 1, N):
                        if lbl[i] == lbl[j]:
                            M_flat[i, j] += 1
                            M_flat[j, i] += 1
            M_flat /= len(runs)
            np.fill_diagonal(M_flat, 1)
            eci = M_flat.mean(axis=1)
            if algo == "Evidence Accumulation":
                M = np.zeros((N, N))
                for lbl in runs:
                    for i in range(N):
                        for j in range(i + 1, N):
                            if lbl[i] == lbl[j]:
                                w = 1 + theta * (eci[i] + eci[j])
                                M[i, j] += w
                                M[j, i] += w
                M /= len(runs)
                np.fill_diagonal(M, 1)
                dist = 1 - M
            else:
                A = np.zeros((N, N))
                for lbl in runs:
                    for i in range(N):
                        for j in range(i + 1, N):
                            if lbl[i] == lbl[j]:
                                A[i, j] += theta * (eci[i] * eci[j])
                                A[j, i] += theta * (eci[i] * eci[j])
                A /= len(runs)
                dist = 1 - A
            np.clip(dist, 0, None, out=dist)
            np.fill_diagonal(dist, 0)
            Z = linkage(squareform(dist), method='average')
            labs = fcluster(Z, k, criterion='maxclust') - 1
        else:
            labs = SpectralClustering(n_clusters=k, affinity='rbf', random_state=42).fit_predict(Xs)
        pca = PCA(n_components=2).fit_transform(Xs)
        sil = compute_silhouette(Xs, labs)
        db  = compute_db(Xs, labs)
        ch  = compute_ch(Xs, labs)
        ari = compute_ari(y, labs)
        nmi = compute_nmi(y, labs)
        homo, comp, v = compute_hcv(y, labs)
        st.subheader(f"Results (θ = {theta:.1f})")
        fig, ax = plt.subplots(figsize=(4,4), dpi=100)
        ax.scatter(pca[:,0], pca[:,1], c=labs, s=6, cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        st.image(buf, width=300)
        plt.close(fig)
        metrics = {"Silhouette": f"{sil:.4f}" if sil else "N/A", "Davies–Bouldin": f"{db:.4f}" if db else "N/A", "Calinski–Harabasz": f"{ch:.4f}" if ch else "N/A", "Adjusted Rand": f"{ari:.4f}", "NMI": f"{nmi:.4f}", "Homogeneity": f"{homo:.4f}", "Completeness": f"{comp:.4f}", "V-Measure": f"{v:.4f}"}
        st.subheader("Clustering Metrics")
        st.table(pd.DataFrame(metrics.items(), columns=["Metric","Value"]))
        st.session_state.history.append({"Time": datetime.now().strftime("%H:%M:%S"), "Algorithm": algo, "k": k, "Theta": theta, **metrics})
elif page == "Descriptions":
    tab1, tab2 = st.tabs(["Algorithms", "Metrics"])
    with tab1:
        st.header("Algorithms")
        st.markdown("""
**KMeans**  
Partitions data into k clusters by minimizing within-cluster variance.

**ACMK (Multi-KMeans)**  
Runs KMeans multiple times, builds a co-association matrix, finds a consensus, and refines labels.

**Graph Partitioning**  
Builds a similarity graph from multiple runs and clusters via hierarchical linkage.

**Evidence Accumulation**  
Aggregates base clusterings into a co-association matrix and clusters hierarchically.

**USPEC**  
Applies spectral embedding on a similarity graph followed by KMeans.
""")
    with tab2:
        st.header("Metrics")
        st.markdown("""
**Silhouette Score**  
Tightness and separation of clusters (−1 to 1).

**Davies–Bouldin Index**  
Ratio of within-cluster scatter to between-cluster separation (lower is better).

**Calinski–Harabasz Index**  
Ratio of between-cluster to within-cluster dispersion (higher is better).

**Adjusted Rand Index (ARI)**  
Agreement between true labels and clusters, adjusted for chance (−1 to 1).

**Normalized Mutual Information (NMI)**  
Normalized dependency between assignments (0 to 1).

**Homogeneity**  
Each cluster contains only one class.

**Completeness**  
All members of a class are in the same cluster.

**V-Measure**  
Harmonic mean of homogeneity and completeness.
""")
else:
    st.header("Run History")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.write("No runs yet.")
