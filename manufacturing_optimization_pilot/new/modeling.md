# Modeling

## Algorithm Selection
- Selected K-means clustering for its simplicity and effectiveness in identifying groups
- Used the Weka implementation of SimpleKMeans

## Parameters
- Number of clusters: 3
- Distance function: Euclidean distance

## Model Building Process
1. Converted data to ARFF format for Weka
2. Applied SimpleKMeans clustering
3. Assigned cluster labels to each data point
4. Saved the results to 'clustered_transformed_data.csv'

## Initial Results
- Cluster 0: 138 instances (37.5%)
- Cluster 1: 138 instances (37.5%)
- Cluster 2: 92 instances (25.0%)
