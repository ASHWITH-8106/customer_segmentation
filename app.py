# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

# -----------------------------
# UI Title
# -----------------------------
st.title("Customer Segmentation App (K-Means + Elbow Method)")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

st.subheader("Raw Dataset")
st.write(df.head())

# -----------------------------
# Feature Selection
# -----------------------------
st.subheader("Select Features for Clustering")

features = st.multiselect(
    "Choose features",
    df.columns,
    default=["Annual Income (k$)", "Spending Score (1-100)"]
)

if len(features) < 2:
    st.warning("Select at least 2 features")
    st.stop()

X = df[features]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Elbow Method
# -----------------------------
st.subheader("Elbow Method")

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Find optimal K using KneeLocator
kneedle = KneeLocator(K, inertia, curve="convex", direction="decreasing")
optimal_k = kneedle.elbow

# Plot
fig, ax = plt.subplots()
ax.plot(K, inertia, marker='o')
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")

if optimal_k:
    ax.axvline(optimal_k, color='r', linestyle='--', label=f"Optimal K = {optimal_k}")
    ax.legend()

st.pyplot(fig)

st.success(f"Optimal number of clusters: {optimal_k}")

# -----------------------------
# KMeans Model
# -----------------------------
st.subheader("Apply K-Means")

k = st.slider("Select number of clusters", 2, 10, optimal_k if optimal_k else 3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

st.write("Clustered Data")
st.write(df.head())

# -----------------------------
# Visualization
# -----------------------------
st.subheader("Cluster Visualization")

if len(features) == 2:
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=clusters
    )
    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])
    st.pyplot(fig2)
else:
    st.info("Select exactly 2 features for visualization")

# -----------------------------
# Cluster Insights
# -----------------------------
st.subheader("Cluster Insights")

cluster_summary = df.groupby("Cluster")[features].mean()
st.write(cluster_summary)