# Q1
import pandas as pd

# Load the dataset
df = pd.read_csv('C:/VSProjects/SIT307-6-2C/microclimate-sensors-data.csv')
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Exclude the rows with missing 'SensorLocation' values
df = df[df['SensorLocation'].notnull()]

# Exclude 'Device_id' and 'LatLong' columns
df = df.drop(columns=['Device_id', 'LatLong'])

# Convert the 'Time' column from 2025-02-09T11:54:37+11:00 format to datetime
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%S%z', utc=True)
# Convert the 'Time' column to seconds since epoch
df['Time'] = df['Time'].astype('int64') // 10**9

# Encode 'SensorLocation' with label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['SensorLocation'] = le.fit_transform(df['SensorLocation'])

print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# List columns with null values
null_columns = df.columns[df.isnull().any()].tolist()
# # Plot histograms of the columns with missing values
# for column in null_columns:
#     plt.figure(figsize=(10, 5))
#     # remove null values from the column before plotting
#     sns.histplot(df[column].dropna(), bins=30, kde=True)
#     plt.title(f'Histogram of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()

# Fill null values with the mean or median of the column
for column in null_columns:
    if column == 'AirTemperature':
        # Fill null values with the mean of the column
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        # Fill null values with the median of the column
        df[column].fillna(df[column].median(), inplace=True)

# Apply min-max scaling to the dataset excluding the 'SensorLocation' column
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[df.columns.difference(['SensorLocation'])] = scaler.fit_transform(df[df.columns.difference(['SensorLocation'])])

print(df_scaled.head())
print(df_scaled.info())
print(df_scaled.describe())
print(df_scaled.isnull().sum())

from sklearn.decomposition import PCA
import numpy as np
PCA_COMPONENTS = 6 # set the number of components for PCA
SEED = 439 # set a random seed for reproducibility
# Perform PCA to reduce the dimensionality of the dataset to 4 components
pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
df_scaled_no_sen_loc = df_scaled.drop(columns=['SensorLocation'])
df_pca_no_label = pca.fit_transform(df_scaled_no_sen_loc)
# Convert the PCA result back to a DataFrame
df_pca_no_label = pd.DataFrame(df_pca_no_label, columns=[f'PC{i+1}' for i in range(PCA_COMPONENTS)])
df_pca = df_pca_no_label.copy()
df_pca['SensorLocation'] = df['SensorLocation'].values # add the 'SensorLocation' column back to the DataFrame
print(df_pca.head())
print(df_pca.info())

# Plot the cumulative variance explained by the principal components
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# # determine how many components are needed to retain 90% of the total variance.
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
# plt.title('Cumulative Explained Variance by Principal Components')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.axhline(y=0.9, color='r', linestyle='--')
# plt.axvline(x=5, color='g', linestyle='--')
# plt.xticks(range(1, len(cumulative_variance) + 1))
# plt.grid()
# plt.show()
# 5 components are needed to retain 90% of the total variance.

import os
# Set the number of threads for OpenMP to avoid a runtime error
os.environ["OMP_NUM_THREADS"] = "16"
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

model = KMeans(random_state=SEED, n_init=1, init='k-means++')
visualizer = KElbowVisualizer(model, k=(1,11), metric='distortion', timings=False) # distortion = Euclidean distance
# visualizer.fit(df_pca_no_label) # Fit the data to the visualizer
# visualizer.show()
# The optimal number of clusters is 5, as the elbow point is at k=5.

# Fit the KMeans model with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=1, init='k-means++')
kmeans.fit(df_pca_no_label)

from sklearn import metrics
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score

SIL_SAMPLE_SIZE = 10000 # set the sample size for silhouette score calculation

# Evaluate the clustering results using inertia, silhouette score, purity score, and mutual information score
avg_inertia = kmeans.inertia_
silhouette_kmeans = silhouette_score(df_pca_no_label, kmeans.labels_, metric='euclidean', sample_size=SIL_SAMPLE_SIZE, random_state=SEED)
purity = purity_score(df['SensorLocation'], kmeans.labels_)
mis = normalized_mutual_info_score(df['SensorLocation'], kmeans.labels_)

# Print the evaluation results
print(f'Inertia: {avg_inertia}')
print(f'Silhouette Score: {silhouette_kmeans}')
print(f'Purity Score: {purity}')
print(f'Mutual Information Score: {mis}')

# Q2

# # DBSCAN
# from sklearn.cluster import DBSCAN

# euclidean_results = []
# mahalanobis_results = []
# manhattan_results = []
# cosine_results = []

# cov_matrix = np.cov(df_pca_no_label.T)  # covariance matrix for Mahalanobis distance
# cov_matrix_inv = np.linalg.inv(cov_matrix)  # Inverse of the covariance matrix

# def calculate_DBSCAN_performance(eps, min_samples, df, metric, metric_params=None,
#                                  sil_metric=None, sil_sample_size=None, random_state=None):
#     # apply DBSCAN to the dataset using the specified metric
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
#                     metric_params=metric_params, n_jobs=-1).fit(df)
#     # get the number of clusters and noise points
#     labels = dbscan.labels_
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = list(labels).count(-1)
#     # calculate silhouette score
#     if n_clusters > 1:
#         silhouette = silhouette_score(df, labels, metric=sil_metric, sample_size=sil_sample_size,
#                                       random_state=random_state)
#     else:
#         silhouette = -1
#     return n_clusters, n_noise, silhouette

# for i in range(2, 53, 10): # eps values from 0.0002 to 0.0052 with a step of 0.0001
#     for j in range(3, 11): # min_samples values from 3 to 10
#         eps = i / 10000
#         # calculate DBSCAN performance for each metric
#         euclidean_result = calculate_DBSCAN_performance(eps, j, df_pca_no_label, 'euclidean',
#                                                             sil_metric='euclidean', sil_sample_size=SIL_SAMPLE_SIZE,
#                                                             random_state=SEED)
#         mahalanobis_result = calculate_DBSCAN_performance(eps, j, df_pca_no_label, 'mahalanobis',
#                                                             metric_params={'V':cov_matrix_inv},
#                                                             sil_metric='euclidean', sil_sample_size=SIL_SAMPLE_SIZE,
#                                                             random_state=SEED)
#         manhattan_result = calculate_DBSCAN_performance(eps, j, df_pca_no_label, 'cityblock',
#                                                             sil_metric='euclidean', sil_sample_size=SIL_SAMPLE_SIZE,
#                                                             random_state=SEED)
#         # cosine_result = calculate_DBSCAN_performance(eps, j, df_pca_no_label, 'cosine',
#         #                                                 sil_metric='euclidean', sil_sample_size=SIL_SAMPLE_SIZE,
#         #                                                 random_state=SEED)
        
#         euclidean_results.append({'eps': eps, 'min_samples': j, 'n_clusters': euclidean_result[0],
#                                   'n_noise': euclidean_result[1], 'silhouette': euclidean_result[2]})
#         mahalanobis_results.append({'eps': eps, 'min_samples': j, 'n_clusters': mahalanobis_result[0],
#                                     'n_noise': mahalanobis_result[1], 'silhouette': mahalanobis_result[2]})
#         manhattan_results.append({'eps': eps, 'min_samples': j, 'n_clusters': manhattan_result[0],
#                                     'n_noise': manhattan_result[1], 'silhouette': manhattan_result[2]})
#         # cosine_results.append({'eps': eps, 'min_samples': j, 'n_clusters': cosine_result[0],
#         #                         'n_noise': cosine_result[1], 'silhouette': cosine_result[2]})

# # create dataframes for the results
# euclidean_results_df = pd.DataFrame(euclidean_results)
# mahalanobis_results_df = pd.DataFrame(mahalanobis_results)
# manhattan_results_df = pd.DataFrame(manhattan_results)
# # cosine_results_df = pd.DataFrame(cosine_results)

# # deterine the optimal number of samples and eps for both methods
# optimal_euclidean = euclidean_results_df.loc[euclidean_results_df['silhouette'].idxmax()]
# optimal_mahalanobis = mahalanobis_results_df.loc[mahalanobis_results_df['silhouette'].idxmax()]
# optimal_manhattan = manhattan_results_df.loc[manhattan_results_df['silhouette'].idxmax()]
# # optimal_cosine = cosine_results_df.loc[cosine_results_df['silhouette'].idxmax()]
# print("Optimal parameters for Euclidean distance:")
# print(optimal_euclidean)
# print("Optimal parameters for Mahalanobis distance:")
# print(optimal_mahalanobis) 
# print("Optimal parameters for Manhattan distance:")
# print(optimal_manhattan)
# # print("Optimal parameters for Cosine distance:")
# # print(optimal_cosine)

# # CURE Algorithm
# from sklearn.cluster import AgglomerativeClustering

# # Create a sample of the dataset
# CURE_SAMPLE_SIZE = 50000 # Full tree requires 358GiB of memory, so take a sample
# CURE_SEED = 1818
# df_pca_no_label_sample_cure = df_pca_no_label.sample(n=CURE_SAMPLE_SIZE, random_state=CURE_SEED)

# cure_results = []

# for i in range(2, 11):
#     # Apply CURE clustering to the dataset using the specified metric
#     ac = AgglomerativeClustering(n_clusters=i, metric='euclidean', compute_full_tree=True, linkage='ward', distance_threshold=None)
#     ac.fit(df_pca_no_label_sample_cure)
#     # Get the number of clusters
#     labels = ac.labels_
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     # Calculate silhouette score
#     silhouette_cure = silhouette_score(df_pca_no_label_sample_cure, labels, metric='euclidean', sample_size=SIL_SAMPLE_SIZE, random_state=SEED)
    
#     cure_results.append({'n_clusters': n_clusters, 'silhouette': silhouette_cure})

# # Create a DataFrame for the results
# cure_results_df = pd.DataFrame(cure_results)

# # Plot the silhouette score for different numbers of clusters
# plt.figure(figsize=(10, 6))
# plt.plot(cure_results_df['n_clusters'], cure_results_df['silhouette'], marker='o')
# plt.title('Silhouette Score for Different Numbers of Clusters (CURE)')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')
# plt.xticks(cure_results_df['n_clusters'])
# plt.grid()
# plt.show()

# # Determine the optimal number of clusters for CURE clustering
# optimal_cure = cure_results_df.loc[cure_results_df['silhouette'].idxmax()]
# print(f"Optimal parameters for CURE clustering with sample size of {CURE_SAMPLE_SIZE}:")
# print(optimal_cure)

# Q3

# Dynamic time warping (DTW)
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import cdist_dtw

# Create a sample of the dataset
DTW_SAMPLE_SIZE = 1000 # Much larger and the program crashes after ~30 minutes
DTW_SEED = 1818
df_pca_no_label_sample_dtw = df_pca_no_label.sample(n=DTW_SAMPLE_SIZE, random_state=DTW_SEED)

# Convert the dataset to a time series dataset
df_time_series = to_time_series_dataset(df_pca_no_label_sample_dtw.values)
# Scale the time series data
df_time_series = TimeSeriesScalerMeanVariance().fit_transform(df_time_series)

# Range of cluster numbers to test
DTW_CLUSTER_COUNT_RANGE = range(1, 11)
dtw_distortions = []

for k in DTW_CLUSTER_COUNT_RANGE:
    # Fit the DTW-based KMeans model
    dtw_model = TimeSeriesKMeans(n_clusters=k, metric="dtw", n_init=1, random_state=SEED, n_jobs=-1, init='k-means++')
    dtw_model.fit(df_time_series)
    
    # Compute the distortion (sum of DTW distances to cluster centers)
    distances = cdist_dtw(df_time_series, dtw_model.cluster_centers_)
    distortion = sum([min(dist) for dist in distances])
    dtw_distortions.append(distortion)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(DTW_CLUSTER_COUNT_RANGE, dtw_distortions, marker='o')
plt.title('Elbow Method for DTW Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Sum of DTW Distances)')
plt.xticks(DTW_CLUSTER_COUNT_RANGE)
plt.axvline(x=4, color='g', linestyle='--')
plt.grid()
plt.show()

# The optimal number of clusters is 4, as the elbow point is at k=4.
# # Fit the DTW-based KMeans model with the optimal number of clusters
dtw_model = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=1, random_state=SEED, n_jobs=-1, init='k-means++')
dtw_model.fit(df_time_series)
# Get the cluster labels for each time series
dtw_labels = dtw_model.labels_
# Get the cluster centers for each time series
dtw_centers = dtw_model.cluster_centers_
# Get the number of clusters
n_clusters = len(set(dtw_labels)) - (1 if -1 in dtw_labels else 0)
# Calculate silhouette score
dtw_distance_matrix = cdist_dtw(df_time_series, n_jobs=-1)
silhouette_dtw = silhouette_score(dtw_distance_matrix, dtw_labels, metric='precomputed') # No need to sample, as sampling is done in the DTW clustering step

# Print the DTW clustering results
print(f"CLustering results for DTW clustering with sample size of {DTW_SAMPLE_SIZE}:")
print(f"Number of clusters: {n_clusters}")
print(f"Silhouette score: {silhouette_dtw}")

# KShape
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# from tslearn.utils import to_time_series_dataset
from tslearn.metrics import cdist_dtw
from tslearn.clustering import KShape

# KSHAPE_SAMPLE_SIZE = 1000
# KSHAPE_SEED = 578

# df_pca_no_label_sample_kshape = df_pca_no_label.sample(n=KSHAPE_SAMPLE_SIZE, random_state=KSHAPE_SEED)
# # Convert the dataset to a time series dataset
# df_time_series_kshape = to_time_series_dataset(df_pca_no_label_sample_kshape.values)
# # Scale the time series data
# df_time_series_kshape = TimeSeriesScalerMeanVariance().fit_transform(df_time_series_kshape)

# Use the same sample as DTW for KShape clustering

KSHAPE_CLUSTER_COUNT_RANGE = range(1, 11)
kshape_distortions = []

for k in KSHAPE_CLUSTER_COUNT_RANGE:
    # Fit the KShape-based KMeans model
    kshape_model = KShape(n_clusters=k, n_init=1, random_state=DTW_SEED, init='random')
    kshape_model.fit(df_time_series)
    
    # Compute the distortion (sum of KShape distances to cluster centers)
    distances = cdist_dtw(df_time_series, kshape_model.cluster_centers_)
    distortion = sum([min(dist) for dist in distances])
    kshape_distortions.append(distortion)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(KSHAPE_CLUSTER_COUNT_RANGE, kshape_distortions, marker='o')
plt.title('Elbow Method for KShape Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Sum of KShape Distances)')
plt.xticks(KSHAPE_CLUSTER_COUNT_RANGE)
plt.axvline(x=3, color='g', linestyle='--')
plt.grid()
plt.show()
# Elbow at 3 clusters

# Perform KShape clustering with ideal number of clusters
kshape_model = KShape(n_clusters=3, n_init=1, random_state=KSHAPE_SEED, init='random')
kshape_model.fit(df_time_series)
silhouette_kshape = silhouette_score(df_time_series[:, :, 0], kshape_model.labels_, metric="euclidean")

# Print the KShape clustering results
print(f"Clustering results for KShape clustering with sample size of {DTW_SAMPLE_SIZE}:")
print(f"Number of clusters: {kshape_model.n_clusters}")
print(f"Silhouette score: {silhouette_kshape}")