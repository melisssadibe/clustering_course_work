import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

# Use real functions extracted from your uploaded file
from typing import List

# Paste your real functions here
def locally_weighted_graph_partitioning(base_clusterings, n_clusters_list, n_objects, theta=0.4, k=4):
    from sklearn.metrics import pairwise_distances
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster

    # Compute uncertainty and ECI
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def cluster_uncertainty(cluster_labels):
        return entropy(cluster_labels)

    def compute_eci(base_clusterings, n_clusters_list, n_objects):
        n_partitions = len(base_clusterings)
        eci = np.zeros((n_objects, n_partitions))
        for i, labels in enumerate(base_clusterings):
            for j in range(n_objects):
                cluster_label = labels[j]
                mask = labels == cluster_label
                eci[j, i] = 1 - cluster_uncertainty(labels[mask]) / np.log2(n_clusters_list[i])
        return eci

    eci_matrix = compute_eci(base_clusterings, n_clusters_list, n_objects)
    lw_matrix = np.zeros((n_objects, n_objects))

    for t in range(len(base_clusterings)):
        labels = base_clusterings[t]
        for i in range(n_objects):
            for j in range(n_objects):
                if labels[i] == labels[j]:
                    lw_matrix[i, j] += eci_matrix[i, t] * eci_matrix[j, t]

    lw_matrix /= len(base_clusterings)

    dist = 1 - lw_matrix
    Z = linkage(pairwise_distances(dist), method='average')
    return fcluster(Z, k, criterion='maxclust') - 1

def locally_weighted_evidence_accumulation(base_clusterings, n_clusters_list, n_objects, theta=0.4, k=4):
    from sklearn.metrics import pairwise_distances
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    def cluster_uncertainty(cluster_labels):
        return entropy(cluster_labels)

    def compute_eci(base_clusterings, n_clusters_list, n_objects):
        n_partitions = len(base_clusterings)
        eci = np.zeros((n_objects, n_partitions))
        for i, labels in enumerate(base_clusterings):
            for j in range(n_objects):
                cluster_label = labels[j]
                mask = labels == cluster_label
                eci[j, i] = 1 - cluster_uncertainty(labels[mask]) / np.log2(n_clusters_list[i])
        return eci

    eci_matrix = compute_eci(base_clusterings, n_clusters_list, n_objects)
    lwca_matrix = np.zeros((n_objects, n_objects))

    for t in range(len(base_clusterings)):
        labels = base_clusterings[t]
        for i in range(n_objects):
            for j in range(n_objects):
                if labels[i] == labels[j]:
                    lwca_matrix[i, j] += 1 + theta * (eci_matrix[i, t] + eci_matrix[j, t])

    lwca_matrix /= len(base_clusterings)
    dist = 1 - lwca_matrix
    Z = linkage(pairwise_distances(dist), method='average')
    return fcluster(Z, k, criterion='maxclust') - 1

# --- Streamlit UI ---
st.title("Locally Weighted Clustering Algorithms")

st.sidebar.header("Data Generator")
n_samples = st.sidebar.slider("Samples", 100, 1000, 300, 50)
n_features = st.sidebar.slider("Features", 2, 10, 4)
centers = st.sidebar.slider("True Clusters", 2, 10, 3)
std_dev = st.sidebar.slider("Cluster Std", 0.5, 5.0, 1.0, 0.1)

tabs = st.tabs(["Graph Partitioning", "Evidence Accumulation"])

def generate_base_clusterings(X, n_runs=5):
    base_clusterings = []
    n_clusters_list = []
    for _ in range(n_runs):
        k = np.random.randint(2, 10)
        labels = KMeans(n_clusters=k, random_state=np.random.randint(10000)).fit_predict(X)
        base_clusterings.append(labels)
        n_clusters_list.append(k)
    return base_clusterings, n_clusters_list

with tabs[0]:
    st.header("Locally Weighted Graph Partitioning")
    theta_g = st.slider("θ (theta)", 0.0, 1.0, 0.4, 0.05)
    k_g = st.slider("Final clusters (k)", 2, 10, 4)

    if st.button("Run Graph Partitioning"):
        X, y_true = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,
                               cluster_std=std_dev, random_state=42)
        base_clusterings, n_clusters_list = generate_base_clusterings(X)
        labels = locally_weighted_graph_partitioning(base_clusterings, n_clusters_list, n_samples, theta=theta_g, k=k_g)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        st.subheader("PCA 2D Plot")
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis")
        ax.set_title("Graph Partitioning - PCA")
        st.pyplot(fig)

        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
        ari = adjusted_rand_score(y_true, labels)

        st.subheader("Metrics")
        st.table(pd.DataFrame([{
            "Algorithm": "Locally Weighted Graph Partitioning",
            "Clusters": len(np.unique(labels)),
            "Silhouette Score": sil,
            "Adjusted Rand Index": ari
        }]))

with tabs[1]:
    st.header("Locally Weighted Evidence Accumulation")
    theta_e = st.slider("θ (theta)", 0.0, 1.0, 0.4, 0.05, key="e_theta")
    k_e = st.slider("Final clusters (k)", 2, 10, 4, key="e_k")

    if st.button("Run Evidence Accumulation"):
        X, y_true = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,
                               cluster_std=std_dev, random_state=42)
        base_clusterings, n_clusters_list = generate_base_clusterings(X)
        labels = locally_weighted_evidence_accumulation(base_clusterings, n_clusters_list, n_samples, theta=theta_e, k=k_e)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        st.subheader("PCA 2D Plot")
        fig2, ax2 = plt.subplots()
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis")
        ax2.set_title("Evidence Accumulation - PCA")
        st.pyplot(fig2)

        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
        ari = adjusted_rand_score(y_true, labels)

        st.subheader("Metrics")
        st.table(pd.DataFrame([{
            "Algorithm": "Locally Weighted Evidence Accumulation",
            "Clusters": len(np.unique(labels)),
            "Silhouette Score": sil,
            "Adjusted Rand Index": ari
        }]))
