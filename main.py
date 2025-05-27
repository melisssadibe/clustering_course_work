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
from scipy.spatial.distance import squareform
from datetime import datetime

def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@"
        f"{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    )

def init_db():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS history (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT now(),
                algorithm TEXT,
                theta DOUBLE PRECISION,
                k INT,
                silhouette DOUBLE PRECISION
            );
        """))
        try:
            conn.execute(text("ALTER TABLE history ADD COLUMN ari DOUBLE PRECISION;"))
        except Exception as e:
            if "already exists" not in str(e):
                raise
        conn.commit()

def generate_base_clusterings(X, n_runs=5):
    base_clusterings = []
    n_clusters_list = []
    for _ in range(n_runs):
        k = np.random.randint(2, 10)
        labels = KMeans(n_clusters=k, random_state=np.random.randint(10000)).fit_predict(X)
        base_clusterings.append(labels)
        n_clusters_list.append(k)
    return base_clusterings, n_clusters_list

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def cluster_uncertainty(cluster_labels):
    return entropy(cluster_labels)

def compute_eci(base_clusterings, n_clusters_list, n_objects):
    eci = np.zeros((n_objects, len(base_clusterings)))
    for i, labels in enumerate(base_clusterings):
        for j in range(n_objects):
            mask = labels == labels[j]
            eci[j, i] = 1 - cluster_uncertainty(labels[mask]) / np.log2(n_clusters_list[i])
    return eci

def locally_weighted_graph_partitioning(base_clusterings, n_clusters_list, n_objects, theta=0.4, k=4):
    from scipy.cluster.hierarchy import linkage, fcluster
    eci_matrix = compute_eci(base_clusterings, n_clusters_list, n_objects)
    lw_matrix = np.zeros((n_objects, n_objects))
    for t, labels in enumerate(base_clusterings):
        for i in range(n_objects):
            for j in range(n_objects):
                if labels[i] == labels[j]:
                    lw_matrix[i, j] += eci_matrix[i, t] * eci_matrix[j, t]
    lw_matrix /= len(base_clusterings)
    dist = 1 - lw_matrix
    Z = linkage(squareform(dist), method='average')
    return fcluster(Z, k, criterion='maxclust') - 1

def locally_weighted_evidence_accumulation(base_clusterings, n_clusters_list, n_objects, theta=0.4, k=4):
    from scipy.cluster.hierarchy import linkage, fcluster
    eci_matrix = compute_eci(base_clusterings, n_clusters_list, n_objects)
    lwca_matrix = np.zeros((n_objects, n_objects))
    for t, labels in enumerate(base_clusterings):
        for i in range(n_objects):
            for j in range(n_objects):
                if labels[i] == labels[j]:
                    lwca_matrix[i, j] += 1 + theta * (eci_matrix[i, t] + eci_matrix[j, t])
    lwca_matrix /= len(base_clusterings)
    dist = 1 - lwca_matrix
    dist = np.clip(dist, 0, None)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist), method='average')
    return fcluster(Z, k, criterion='maxclust') - 1

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

st.set_page_config(layout="wide")
init_db()

st.markdown("""
    <style>
    h1, h2, h3 {
        font-weight: 700;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTable tbody tr td {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Page", ["Clustering", "Run History"])

if selected_page == "Clustering":
    st.sidebar.title("Clustering Settings")

    algo_choice = st.sidebar.radio("Algorithm", ["Graph", "Evidence"])
    algo_name_map = {
        "Graph": "Graph Partitioning",
        "Evidence": "Evidence Accumulation"
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
        base_clusterings, n_clusters_list = generate_base_clusterings(X_scaled)

        if algorithm == "Graph Partitioning":
            labels = locally_weighted_graph_partitioning(base_clusterings, n_clusters_list, n_samples, theta=theta, k=k)
        else:
            labels = locally_weighted_evidence_accumulation(base_clusterings, n_clusters_list, n_samples, theta=theta, k=k)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        ari = adjusted_rand_score(base_clusterings[0], labels) if base_clusterings else None

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
