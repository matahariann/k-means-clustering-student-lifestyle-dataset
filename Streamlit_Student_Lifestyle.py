import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Load pipeline
with open('Model_KMeans_Student_Lifestyle.pkl', 'rb') as f:
    pipeline = pickle.load(f)

scaler = pipeline['scaler']
pca = pipeline['pca']
kmeans = pipeline['kmeans']

# Title for the app
st.title('Student Lifestyle Clustering')

# Input fields for the user
col1, col2 = st.columns(2)

with col1:
    Study_Hours_Per_Day = st.number_input('Study Hours Per Day', step=0.1)
    Extracurricular_Hours_Per_Day = st.number_input('Extracurricular Hours Per Day', step=0.1)
    Sleep_Hours_Per_Day = st.number_input('Sleep Hours Per Day', step=0.1)

with col2:
    Social_Hours_Per_Day = st.number_input('Social Hours Per Day', step=0.1)
    Physical_Activity_Hours_Per_Day = st.number_input('Physical Activity Hours Per Day', step=0.1)
    GPA = st.number_input('GPA', min_value=0.0, step=0.1)

# Predict when the button is pressed
if st.button('Predict Cluster'):
    # Prepare the new data as a DataFrame
    data_baru = pd.DataFrame([{
        'Study_Hours_Per_Day': Study_Hours_Per_Day,
        'Extracurricular_Hours_Per_Day': Extracurricular_Hours_Per_Day,
        'Sleep_Hours_Per_Day': Sleep_Hours_Per_Day,
        'Social_Hours_Per_Day': Social_Hours_Per_Day,
        'Physical_Activity_Hours_Per_Day': Physical_Activity_Hours_Per_Day,
        'GPA': GPA
    }])

    # Normalisasi data baru
    data_baru_scaled = scaler.transform(data_baru)

    # Reduksi dimensi dengan PCA
    data_baru_pca = pca.transform(data_baru_scaled)

    # Prediksi cluster
    cluster_pred = kmeans.predict(data_baru_pca)

    # Map the cluster prediction to a readable label
    if cluster_pred[0] == 0:
        cluster_label = '1'
    elif cluster_pred[0] == 1:
        cluster_label = '2'
    elif cluster_pred[0] == 2:
        cluster_label = '3'
    elif cluster_pred[0] == 3:
        cluster_label = '4'

    # Display the result to the user
    st.success(f'Predicted Cluster: {cluster_label}')

    # Hitung jarak data baru ke setiap centroid
    distances = np.linalg.norm(kmeans.cluster_centers_ - data_baru_pca, axis=1)

    # Menampilkan jarak ke setiap centroid dengan format yang diinginkan
    for i, distance in enumerate(distances):
        st.write(f"Jarak Data Baru Ke Centroid Cluster {i+1} = {distance:.2f}")

    # Visualisasi data baru dengan centroid
    plt.figure(figsize=(8, 6))

    # Plot centroid cluster
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                color='red', label='Centroids', s=100, marker='X')

    # Tambahkan label untuk setiap centroid di bawah titik (termasuk jarak)
    for i, (x, y) in enumerate(kmeans.cluster_centers_):
        plt.text(x, y - 0.05, f'Cluster {i+1}\nDist: {distances[i]:.2f}',
                 color='black', fontsize=10, ha='center', va='top')

    # Plot data baru
    plt.scatter(data_baru_pca[:, 0], data_baru_pca[:, 1],
                color='yellow', label='New Data', s=100, edgecolor='black')

    # Hubungkan data baru ke setiap centroid dengan garis
    for i, centroid in enumerate(kmeans.cluster_centers_):
        plt.plot([data_baru_pca[0, 0], centroid[0]],
                 [data_baru_pca[0, 1], centroid[1]],
                 linestyle='--', color='gray')

    # Atur rentang sumbu dan jarak ticks
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks(np.arange(-1, 1.25, 0.25))
    plt.yticks(np.arange(-1, 1.25, 0.25))

    # Label dan judul
    plt.title("New Data Points and Cluster Centroids with Distances")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(alpha=0.5)

    # Display the plot in Streamlit
    st.pyplot(plt)
