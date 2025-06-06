import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from sqlalchemy import create_engine, text
from scipy.spatial.distance import squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import csr_matrix, diags
from datetime import datetime

def pdist2_fast(X, Y, metric='sqeuclidean'):
    if metric in [0, 'sqeuclidean']:
        D = dist_euc_sq(X, Y)
    elif metric == 'euclidean':
        D = np.sqrt(dist_euc_sq(X, Y))
    elif metric == 'L1':
        D = dist_l1(X, Y)
    elif metric == 'cosine':
        D = dist_cosine(X, Y)
    elif metric == 'emd':
        D = dist_emd(X, Y)
    elif metric == 'chisq':
        D = dist_chisq(X, Y)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return np.maximum(0, D)

def dist_l1(X, Y):
    m, n = X.shape[0], Y.shape[0]
    D = np.zeros((m, n))
    for i in range(n):
        yi = np.tile(Y[i, :], (m, 1))
        D[:, i] = np.sum(np.abs(X - yi), axis=1)
    return D

def dist_cosine(X, Y):
    X_norm = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    Y_norm = Y / np.sqrt(np.sum(Y**2, axis=1, keepdims=True))
    return 1 - np.dot(X_norm, Y_norm.T)

def dist_emd(X, Y):
    Xcdf = np.cumsum(X, axis=1)
    Ycdf = np.cumsum(Y, axis=1)
    m, n = X.shape[0], Y.shape[0]
    D = np.zeros((m, n))
    for i in range(n):
        ycdf = np.tile(Ycdf[i, :], (m, 1))
        D[:, i] = np.sum(np.abs(Xcdf - ycdf), axis=1)
    return D

def dist_chisq(X, Y):
    m, n = X.shape[0], Y.shape[0]
    D = np.zeros((m, n))
    for i in range(n):
        yi = np.tile(Y[i, :], (m, 1))
        s = yi + X
        d = yi - X
        D[:, i] = np.sum(d**2 / (s + np.finfo(float).eps), axis=1)
    return D / 2

def dist_euc_sq(X, Y):
    XX = np.sum(X**2, axis=1, keepdims=True)
    YY = np.sum(Y**2, axis=1, keepdims=True).T
    return np.abs(XX + YY - 2 * np.dot(X, Y.T))

def get_representatives_by_hybrid_selection(fea, pSize, distance, cntTimes=10):
    N = fea.shape[0]
    bigPSize = min(cntTimes * pSize, N)
    bigRpFea = fea[np.random.choice(N, bigPSize, replace=False)]
    kmeans = KMeans(n_clusters=pSize, max_iter=10, random_state=0).fit(bigRpFea)
    return kmeans.cluster_centers_

def tcut_for_bipartite_graph(B, Nseg, maxKmIters, cntReps):
    Nx, Ny = B.shape
    if Ny < Nseg:
        raise ValueError("Need more columns!")
    dx = np.array(B.sum(axis=1)).ravel()
    Dx = np.diag(1.0 / np.maximum(dx, 1e-10))
    Wy = B.T @ Dx @ B
    d = np.array(Wy.sum(axis=1)).flatten()
    D = np.diag(1.0 / np.sqrt(d))
    nWy = D @ Wy @ D
    if hasattr(nWy, "toarray"):
        nWy = nWy.toarray()
    eigvals, eigvecs = np.linalg.eigh(nWy)
    idx = np.argsort(-eigvals)[:Nseg]
    Ncut_evec = D @ eigvecs[:, idx]
    evec = Dx @ B @ Ncut_evec
    evec /= np.linalg.norm(evec, axis=1, keepdims=True) + 1e-10
    kmeans = KMeans(n_clusters=Nseg, max_iter=maxKmIters, n_init=cntReps, random_state=0).fit(evec)
    return kmeans.labels_

def uspec(fea, Ks, distance='euclidean', p=1000, Knn=5, maxTcutKmIters=100, cntTcutKmReps=3):
    N = fea.shape[0]
    p = min(p, N)
    RpFea = get_representatives_by_hybrid_selection(fea, p, distance)
    cntRepCls = int(np.sqrt(p))
    kmeans = KMeans(n_clusters=cntRepCls, max_iter=20, random_state=0).fit(RpFea)
    repClsLabel = kmeans.labels_
    repClsCenters = kmeans.cluster_centers_
    centerDist = pdist2_fast(fea, repClsCenters, metric=distance)
    minCenterIdxs = np.argmin(centerDist, axis=1)
    nearestRepInRpFeaIdx = np.zeros(N, dtype=int)
    for i in range(cntRepCls):
        mask = (minCenterIdxs == i)
        repSubset = RpFea[repClsLabel == i]
        nearestIdx = np.argmin(pdist2_fast(fea[mask], repSubset, metric=distance), axis=1)
        nearestRepInRpFeaIdx[mask] = np.flatnonzero(repClsLabel == i)[nearestIdx]
    neighSize = 10 * Knn
    RpFeaW = pdist2_fast(RpFea, RpFea, metric=distance)
    RpFeaKnnIdx = np.argsort(RpFeaW, axis=1)[:, :neighSize + 1]
    RpFeaKnnDist = np.zeros((N, RpFeaKnnIdx.shape[1]))
    for i in range(p):
        mask = (nearestRepInRpFeaIdx == i)
        RpFeaKnnDist[mask] = cdist(fea[mask], RpFea[RpFeaKnnIdx[i]], metric=distance)
    RpFeaKnnIdxFull = RpFeaKnnIdx[nearestRepInRpFeaIdx]
    knnDist = np.zeros((N, Knn))
    knnIdx = np.zeros((N, Knn), dtype=int)
    for i in range(Knn):
        knnDist[:, i] = np.min(RpFeaKnnDist, axis=1)
        minIdx = np.argmin(RpFeaKnnDist, axis=1)
        knnIdx[:, i] = RpFeaKnnIdxFull[np.arange(N), minIdx]
        RpFeaKnnDist[np.arange(N), minIdx] = np.inf
    if distance == 'cosine':
        Gsdx = 1 - knnDist
    else:
        knnMeanDiff = np.mean(knnDist)
        Gsdx = np.exp(-(knnDist ** 2) / (2 * knnMeanDiff ** 2))
    Gsdx[Gsdx == 0] = np.finfo(float).eps
    Gidx = np.tile(np.arange(N), (Knn, 1)).T
    B = csr_matrix((Gsdx.ravel(), (Gidx.ravel(), knnIdx.ravel())), shape=(N, p))
    return tcut_for_bipartite_graph(B, Ks, maxTcutKmIters, cntTcutKmReps)

def usenc(fea, k, M=20, distance='euclidean', p=1000, Knn=5, bcsLowK=20, bcsUpK=60):
    if bcsUpK < bcsLowK:
        bcsUpK = bcsLowK
    base_cls = usenc_ensemble_generation(fea, M, distance, p, Knn, bcsLowK, bcsUpK)
    labels = usenc_consensus_function(base_cls, k)
    return labels

def usenc_ensemble_generation(fea, M, distance='euclidean', p=1000, Knn=5, lowK=5, upK=15):
    N = fea.shape[0]
    if p > N:
        p = N
    members = np.zeros((N, M), dtype=int)
    for i in range(M):
        Ks = np.random.randint(lowK, upK + 1, size=M)
        Ks = len(np.unique(Ks))
        members[:, i] = uspec(fea, Ks, distance, p, Knn)
    return members

def usenc_consensus_function(base_cls, k, max_tcut_km_iters=100, cnt_tcut_km_reps=3):
    N, M = base_cls.shape
    max_cls = np.max(base_cls, axis=0)
    for i in range(1, len(max_cls)):
        max_cls[i] += max_cls[i - 1]
    base_cls[:, 1:] += max_cls[:-1]
    B = csr_matrix((np.ones(N * M), (np.repeat(np.arange(N), M), base_cls.ravel())), shape=(N, max_cls[-1] + 1))
    col_sum = np.array(B.sum(axis=0)).flatten()
    B = B[:, col_sum > 0]
    labels = tcut_for_bipartite_graph(B, k, max_tcut_km_iters, cnt_tcut_km_reps)
    return labels

def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@"
        f"{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    )

def save_history(algorithm, theta, k, silhouette, ari):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO history (algorithm, theta, k, silhouette, ari)
            VALUES (:algorithm, :theta, :k, :silhouette, :ari)
        """), {
            "algorithm": algorithm,
            "theta": theta,
            "k": k,
            "silhouette": float(silhouette),
            "ari": float(ari) if ari is not None else None
        })

def get_history(limit, offset):
    engine = get_engine()
    query = text("""
        SELECT id, timestamp, algorithm, theta, k, silhouette, ari
        FROM history
        ORDER BY timestamp DESC
        LIMIT :limit OFFSET :offset
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"limit": limit, "offset": offset})

def run_multiple_kmeans(X, n_clusters, n_runs=10, max_iter=300, random_state=42):
    np.random.seed(random_state)
    N = X.shape[0]
    labels_array = np.zeros((n_runs, N), dtype=int)
    for i in range(n_runs):
        km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, max_iter=max_iter, random_state=random_state + i)
        km.fit(X)
        labels_array[i] = km.labels_
    return labels_array

def build_coassociation_matrix(labels_array):
    n_runs, N = labels_array.shape
    M = np.zeros((N, N))
    for r in range(n_runs):
        for i in range(N):
            for j in range(i + 1, N):
                if labels_array[r, i] == labels_array[r, j]:
                    M[i, j] += 1
                    M[j, i] += 1
    M /= n_runs
    np.fill_diagonal(M, 1.0)
    return M

def initial_consensus_labels(M, n_clusters):
    dist_vec = squareform(1.0 - M, checks=False)
    Z = linkage(dist_vec, method='average')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    return labels

def refine_consensus_labels(M, labels, max_iter=10, tol=1e-4):
    labels = labels.copy()
    N = M.shape[0]
    for iteration in range(max_iter):
        old_labels = labels.copy()
        unique_clusters = np.unique(labels)
        cluster_indices = {uc: np.where(labels == uc)[0] for uc in unique_clusters}
        for i in range(N):
            current_cluster = labels[i]
            best_cluster = current_cluster
            best_score = -1.0
            for c in unique_clusters:
                inds = cluster_indices[c]
                if len(inds) == 0:
                    continue
                score = np.mean(M[i, inds])
                if score > best_score:
                    best_score = score
                    best_cluster = c
            labels[i] = best_cluster
        changed = np.sum(old_labels != labels)
        changed_ratio = changed / float(N)
        if changed_ratio < tol:
            break
    return labels

def acmk_clustering(X, n_clusters, n_runs=10, max_iter_kmeans=300, random_state=42, refine_max_iter=10, refine_tol=1e-4):
    labels_array = run_multiple_kmeans(X, n_clusters=n_clusters, n_runs=n_runs, max_iter=max_iter_kmeans, random_state=random_state)
    M = build_coassociation_matrix(labels_array)
    init_labels = initial_consensus_labels(M, n_clusters)
    final_labels = refine_consensus_labels(M, init_labels, max_iter=refine_max_iter, tol=refine_tol)
    return final_labels, M

st.set_page_config(layout="wide")
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Page", ["Clustering", "Run History"])

if selected_page == "Clustering":
    st.sidebar.title("Clustering Settings")
    algo_choice = st.sidebar.radio("Algorithm", ["Graph", "Evidence", "ACMK (Multi-KMeans)", "USPEC"])
    algo_name_map = {
        "Graph": "Graph Partitioning",
        "Evidence": "Evidence Accumulation",
        "ACMK (Multi-KMeans)": "ACMK (Multi-KMeans)",
        "USPEC": "USPEC Consensus"
    }
    algorithm = algo_name_map[algo_choice]
    theta = st.sidebar.slider("θ (theta)", 0.0, 1.0, 0.4, 0.05)
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 4)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Generator")
    n_samples = st.sidebar.slider("Number of points", 100, 1000, 300, step=50)
    data_std = st.sidebar.slider("Noise (std)", 0.1, 3.0, 1.0, 0.1)
    run = st.sidebar.button("Run Clustering")

    if run:
        X = np.random.normal(loc=0, scale=data_std, size=(n_samples, 2))
        X_scaled = StandardScaler().fit_transform(X)

        base_clusterings = [KMeans(n_clusters=k, random_state=i).fit_predict(X_scaled) for i in range(5)]
        eci = np.zeros((n_samples, len(base_clusterings)))
        for i, labels in enumerate(base_clusterings):
            for j in range(n_samples):
                mask = (labels == labels[j])
                p = np.bincount(labels[mask], minlength=k).astype(float)
                p /= p.sum()
                if np.any(p > 0):
                    entropy_term = (p[p > 0] * np.log2(p[p > 0])).sum()
                else:
                    entropy_term = 0.0
                eci[j, i] = 1 + entropy_term / np.log2(k)

        if algorithm == "Graph Partitioning":
            lw_matrix = np.zeros((n_samples, n_samples))
            for t, labels in enumerate(base_clusterings):
                for i_ in range(n_samples):
                    for j_ in range(n_samples):
                        if labels[i_] == labels[j_]:
                            lw_matrix[i_, j_] += eci[i_, t] * eci[j_, t]
            lw_matrix /= len(base_clusterings)
            dist = 1.0 - lw_matrix
            dist = (dist + dist.T) / 2.0
            np.fill_diagonal(dist, 0.0)
            Z = linkage(squareform(dist), method='average')
            labels = fcluster(Z, k, criterion='maxclust') - 1

        elif algorithm == "Evidence Accumulation":
            lwca_matrix = np.zeros((n_samples, n_samples))
            for t, labels in enumerate(base_clusterings):
                for i_ in range(n_samples):
                    for j_ in range(n_samples):
                        if labels[i_] == labels[j_]:
                            lwca_matrix[i_, j_] += 1 + theta * (eci[i_, t] + eci[j_, t])
            lwca_matrix /= len(base_clusterings)
            dist = 1.0 - lwca_matrix
            dist = np.clip(dist, 0.0, None)
            dist = (dist + dist.T) / 2.0
            np.fill_diagonal(dist, 0.0)
            Z = linkage(squareform(dist), method='average')
            labels = fcluster(Z, k, criterion='maxclust') - 1

        elif algorithm == "ACMK (Multi-KMeans)":
            labels, _ = acmk_clustering(X_scaled, n_clusters=k)

        elif algorithm == "USPEC Consensus":
            labels = usenc(X_scaled, k)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ari = adjusted_rand_score(np.zeros_like(labels), labels)

        st.title("Locally Weighted Clustering Algorithms")
        st.subheader(algorithm)

        col1, col2 = st.columns([1, 3])
        with col1:
            fig, ax = plt.subplots(figsize=(3, 2.5), dpi=120)
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=6)
            ax.set_title("PCA", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")
            fig.tight_layout(pad=0.5)
            st.pyplot(fig, clear_figure=True)

        metrics = {
            "Number of Clusters (k)": k,
            "Silhouette Score": f"{sil:.4f}",
            "ARI Score": f"{ari:.4f}"
        }
        metrics_table = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        st.markdown("### Clustering Metrics")
        st.table(metrics_table)

        save_history(algorithm, theta, k, sil, ari)

elif selected_page == "Run History":
    st.title("Run History")
    page = st.number_input("Page", min_value=1, step=1, value=1)
    page_size = 5
    offset = (page - 1) * page_size
    history_df = get_history(limit=page_size, offset=offset)
    st.dataframe(history_df)
