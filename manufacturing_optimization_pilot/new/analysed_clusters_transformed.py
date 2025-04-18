import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
from weka.clusterers import Clusterer
import pandas as pd
import os
from datetime import datetime

# Start the JVM
jvm.start()

# First, we need to convert the CSV to ARFF format since your original code uses ARFF
transformed_data = pd.read_csv('transformed_data.csv')

# Save the transformed data to ARFF format
# This is needed because the original code expects an ARFF file
arff_path = "transformed_data.arff"

# Check if any categorical columns need to be handled
categorical_columns = transformed_data.select_dtypes(include=['object']).columns

# Create the ARFF file manually
with open(arff_path, 'w') as arff_file:
    # Write header
    arff_file.write('@relation transformed_data\n\n')
    
    # Write attribute information
    for column in transformed_data.columns:
        if column in categorical_columns:
            # Get unique values for categorical attributes
            unique_values = transformed_data[column].unique()
            arff_file.write(f'@attribute {column} {{')
            arff_file.write(','.join([f'"{val}"' if isinstance(val, str) else str(val) for val in unique_values]))
            arff_file.write('}\n')
        else:
            arff_file.write(f'@attribute {column} numeric\n')
    
    # Write data
    arff_file.write('\n@data\n')
    for _, row in transformed_data.iterrows():
        row_values = []
        for column in transformed_data.columns:
            if column in categorical_columns:
                if pd.isna(row[column]):
                    row_values.append('?')
                else:
                    row_values.append(f'"{row[column]}"')
            else:
                if pd.isna(row[column]):
                    row_values.append('?')
                else:
                    row_values.append(str(row[column]))
        arff_file.write(','.join(row_values) + '\n')

print(f"Created ARFF file: {arff_path}")

# Load the ARFF file
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(arff_path)

# Apply k-means clustering
kmeans = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])  # 3 clusters
kmeans.build_clusterer(data)

# Print cluster assignments and analyze how they got classified
cluster_assignments = []
for index, inst in enumerate(data):
    cluster = kmeans.cluster_instance(inst)
    cluster_assignments.append((index, cluster))
    print(f"Instance {index} is in cluster {cluster}")

# Load the transformed data to analyze the clusters
transformed_data = pd.read_csv('transformed_data.csv')

# Add cluster assignments to the transformed data
transformed_data['cluster'] = [cluster for _, cluster in cluster_assignments]

# Verify that the data with clusters is correct
print("\nSample of transformed data with clusters:")
print(transformed_data.head())

# Analyze the clusters by grouping the data and calculating summary statistics
# Note: We're using the columns available in the transformed data
cluster_analysis = transformed_data.groupby(['cluster', 'process_name']).agg({
    'ordered_height': 'mean',
    'ordered_width': 'mean',
    'num_staff_required': 'mean',
    'process_duration_minutes': 'mean'
}).reset_index()

print("\nCluster Analysis:")
print(cluster_analysis)

# Additional analysis: Count processes per cluster
process_counts = transformed_data.groupby(['cluster', 'process_name']).size().reset_index(name='count')
print("\nProcess counts per cluster:")
print(process_counts)

# Stop the JVM at the very end
jvm.stop()

# Save the updated data with cluster assignments to a new CSV file
transformed_data.to_csv('clustered_transformed_data.csv', index=False)

print("\nTransformed data with cluster assignments has been saved to 'clustered_transformed_data.csv'.")