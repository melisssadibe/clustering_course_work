import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from sqlalchemy import create_engine, text
from sqlalchemy.sql import func
from scipy.spatial.distance import squareform
from datetime import datetime

# ---------- DATABASE SETUP ----------
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
        conn.commit()

# ---------- CORE FUNCTIONS ----------
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

def save_history(algorithm, theta, k, silhouette):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO history (algorithm, theta, k, silhouette)
            VALUES (:algorithm, :theta, :k, :silhouette)
        """), {
            "algorithm": algorithm,
            "theta": theta,
            "k": k,
            "silhouette": float(silhouette)
        })

def get_history(limit, offset):
    engine = get_engine()
    query = text("""
        SELECT id, timestamp, algorithm, theta, k, silhouette
        FROM history
        ORDER BY timestamp DESC
        LIMIT :limit OFFSET :offset
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"limit": limit, "offset": offset})

# ---------- STREAMLIT APP ----------
st.set_page_config(layout="wide")
init_db()

st.sidebar.title("📘 Навигация")
selected_page = st.sidebar.selectbox("Страница", ["Кластеризация", "История запусков"])

if selected_page == "Кластеризация":
    st.title("Locally Weighted Clustering Algorithms")

    st.sidebar.title("⚙️ Настройки кластеризации")
    algorithm = st.sidebar.radio("Алгоритм", ["Graph Partitioning", "Evidence Accumulation"])
    theta = st.sidebar.slider("θ (theta)", 0.0, 1.0, 0.4, 0.05)
    k = st.sidebar.slider("Число кластеров (k)", 2, 10, 4)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Генерация данных")
    n_samples = st.sidebar.slider("Количество точек", 100, 1000, 300, step=50)
    data_std = st.sidebar.slider("Шум (std)", 0.1, 3.0, 1.0, 0.1)

    run = st.sidebar.button("🚀 Запустить кластеризацию")

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

        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis")
        ax.set_title(f"{algorithm} - PCA")
        st.pyplot(fig)

        sil = silhouette_score(X_scaled, labels)
        st.table(pd.DataFrame([{
            "Algorithm": algorithm,
            "Clusters": len(np.unique(labels)),
            "Silhouette Score": sil
        }]))

        save_history(algorithm, theta, k, sil)

elif selected_page == "История запусков":
    st.title("🕓 История запусков")
    page = st.number_input("Страница", min_value=1, step=1, value=1)
    page_size = 5
    offset = (page - 1) * page_size
    history_df = get_history(limit=page_size, offset=offset)
    st.dataframe(history_df)
