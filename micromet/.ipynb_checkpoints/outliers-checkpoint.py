import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_kmeans(timeseries: np.ndarray, n_clusters: int = 3, threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in a time series using K-means clustering.

    Args:
    timeseries (np.ndarray): 1D numpy array containing the time series data.
    n_clusters (int): Number of clusters to use in K-means. Default is 5.
    threshold (float): Number of standard deviations from cluster center to consider as outlier. Default is 2.0.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D numpy arrays:
        - Boolean array where True indicates an outlier.
        - Array of distances from each point to its nearest cluster center.
    """
    # Reshape and scale the time series
    X = timeseries.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Calculate distances to nearest cluster centers
    distances = np.min(kmeans.transform(X_scaled), axis=1)

    # Calculate mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Identify outliers
    is_outlier = distances > mean_distance + threshold * std_distance

    return is_outlier, distances

# Example usage:
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate sample data
# np.random.seed(42)
# timeseries = np.random.randn(1000)
# timeseries[500:510] += 5  # Add some outliers
#
# # Detect outliers
# is_outlier, distances = detect_outliers_kmeans(timeseries)
#
# # Print results
# print(f"Number of outliers detected: {np.sum(is_outlier)}")
# print(f"Indices of outliers: {np.where(is_outlier)[0]}")
#
# # Plot the results
# plt.figure(figsize=(12, 6))
# plt.plot(timeseries, label='Time Series')
# plt.scatter(np.where(is_outlier)[0], timeseries[is_outlier], color='red', label='Outliers')
# plt.legend()
# plt.title('Time Series with Detected Outliers')
# plt.show()


def detect_outliers_lof(timeseries: np.ndarray, n_neighbors: int = 20, contamination: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in a high-frequency time series using the Local Outlier Factor (LOF) algorithm.

    Args:
    timeseries (np.ndarray): 1D numpy array containing the time series data.
    n_neighbors (int): Number of neighbors to consider for each point. Default is 20.
    contamination (float): The proportion of outliers in the data set. Default is 0.08.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D numpy arrays:
        - Boolean array where True indicates an outlier.
        - Array of outlier scores (negative LOF values).
    """
    # Reshape the time series for sklearn
    X = timeseries.reshape(-1, 1)

    # Initialize and fit the LOF model
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)

    # Get the outlier scores
    outlier_scores = lof.negative_outlier_factor_

    # Create a boolean mask for outliers
    is_outlier = y_pred == -1

    return is_outlier, outlier_scores

# Example usage:
# import numpy as np
#
# # Generate sample data
# np.random.seed(42)
# timeseries = np.random.randn(10000)
# timeseries[5000:5010] += 10  # Add some outliers
#
# # Detect outliers
# is_outlier, outlier_scores = detect_outliers_lof(timeseries)
#
# # Print results
# print(f"Number of outliers detected: {np.sum(is_outlier)}")
# print(f"Indices of outliers: {np.where(is_outlier)[0]}")