# Modeling

## Clustering Algorithm
- Selected K-means clustering for its simplicity and effectiveness in identifying groups
- Used the Weka implementation of SimpleKMeans

## Clustering Parameters
- Number of clusters: 3
- Distance function: Euclidean distance

## Anomaly Detection
- Implemented Isolation Forest algorithm for identifying process anomalies
- Selected for its efficiency with high-dimensional data and robustness to outliers

## Anomaly Detection Parameters
- Number of estimators: 100
- Contamination rate: 5% (assumed percentage of anomalies)
- Random state: 42 (for reproducibility)

## Model Building Process
1. Converted data to ARFF format for Weka clustering
2. Applied SimpleKMeans clustering
3. Assigned cluster labels to each data point
4. Standardized numerical features for anomaly detection
5. Applied Isolation Forest to identify process anomalies
6. Calculated anomaly scores for each process
7. Saved the results to 'clustered_anomaly_data.csv'

## Initial Results
- Cluster 1: 1204 instances (39.7%)
- Cluster 0: 1002 instances (33.0%)
- Cluster 2: 826 instances (27.2%)

## Anomaly Detection Results
- Identified 152 anomalous processes (5.0%)

### Anomalies by Cluster
- Cluster 0: 9 anomalies (0.9% of cluster)
- Cluster 1: 33 anomalies (2.7% of cluster)
- Cluster 2: 110 anomalies (13.3% of cluster)

### Process Duration Analysis
- Average duration for normal processes: 7.77 minutes
- Average duration for anomalous processes: 9.45 minutes
- Difference: 1.68 minutes (21.7% longer)
