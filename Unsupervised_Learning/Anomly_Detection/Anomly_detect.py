import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def z_score(X, threshold=3):
    mean = np.mean(X)
    std_dev = np.std(X)
    scores = np.abs((X - mean) / std_dev)
    return np.any(scores > threshold, axis=1)

def main():
    # Load Data From csv File
    anomly_data = pd.read_csv("Submission.csv")
    #print(anomly_data.head())
    #print(anomly_data.info())

    #print(anomly_data.isnull().sum()) 
    # No data is missing

    # Data Acquisition
    X = anomly_data.iloc[:, [1]]  # Single feature

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create IsolationForest Model 
    isolate_forest = IsolationForest(contamination=0.2, random_state=42)
    isolate_forest_outliers = isolate_forest.fit_predict(X_scaled) == -1

    # Z-Score method
    Z_score_outliers = z_score(X_scaled, threshold=4)  # Apply on scaled data

    # Plotting 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.scatter(range(len(X_scaled)), X_scaled[:, 0], c=Z_score_outliers.astype(int), cmap='coolwarm')
    ax1.set_title("Z-Score Method for Anomaly Detection")

    ax2.scatter(range(len(X_scaled)), X_scaled[:, 0], c=isolate_forest_outliers.astype(int), cmap='coolwarm')
    ax2.set_title("Isolation Forest Anomaly Detection")

    plt.tight_layout()
    plt.show()


    # Make sure the labels are 0 and 1 (not boolean)
    z_score_labels = Z_score_outliers.astype(int)
    isolation_labels = isolate_forest_outliers.astype(int)

    # Compute silhouette scores
    sil_zscore = silhouette_score(X_scaled, z_score_labels)
    sil_isolation = silhouette_score(X_scaled, isolation_labels)

    print(f"Silhouette Score (Z-Score): {sil_zscore:.4f}\n")
    print(f"Silhouette Score (Isolation Forest): {sil_isolation:.4f}\n")


if __name__ == "__main__":
    main()
