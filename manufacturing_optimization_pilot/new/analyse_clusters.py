'''
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.clusterers import Clusterer
import pandas as pd

# Start the JVM
jvm.start()

# Load the ARFF file
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("combined_raw_data.arff")

# Apply k-means clustering
kmeans = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])  # 3 clusters
kmeans.build_clusterer(data)

# Print cluster assignments and analyze how they got classified
cluster_assignments = []
for index, inst in enumerate(data):
    cluster = kmeans.cluster_instance(inst)
    cluster_assignments.append((index, cluster))
    print(f"Instance {index} is in cluster {cluster}")

# Load the combined raw data to analyze the clusters
raw_data = pd.read_csv('combined_raw_data.csv')

# Add cluster assignments to the raw data
raw_data['cluster'] = [cluster for _, cluster in cluster_assignments]

# Define the process data
process_data = pd.DataFrame({
    'process_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'process_name': ['cut_2_sheets_to_size', 'cut_sheet_of_larger_size', 'add_attachment_to_sheet', 'apply_indents_to_sheet', 'combine_sheets', 'cut_panel_to_size', 'quality_control_check', 'wrap_on_pallet']
})

# Debug: Print unique process_id values in raw_data and process_data
print("Unique process_id values in raw_data:", raw_data['process_id'].unique())
print("Unique process_id values in process_data:", process_data['process_id'].unique())

# Debug: Check if process_id exists in raw_data
if 'process_id' not in raw_data.columns:
    raise KeyError("The 'process_id' column is missing from raw_data. Ensure it exists before merging.")

# Merge the process data with the raw data to add process names
raw_data = raw_data.merge(process_data, on='process_id', how='left')

# Debug: Check if process_name exists after the merge
if 'process_name' not in raw_data.columns:
    raise KeyError("The 'process_name' column is missing after merging. Check the process_id values in raw_data and process_data.")

# Fill missing process names with a default value
raw_data['process_name'] = raw_data['process_name'].fillna('Unknown')

# Analyze the clusters by grouping the data by cluster and process name and calculating summary statistics
cluster_analysis = raw_data.groupby(['cluster', 'process_name']).agg({
    'process_id': 'count',
    'datetime_started': 'min',
    'datetime_ended': 'max',
    'equipment_required': 'mean',
    'num_staff_required': 'mean'
}).reset_index()

print(cluster_analysis)

# Stop the JVM at the very end
jvm.stop()

# Save the updated raw data with process names to a new CSV file
raw_data.to_csv('updated_combined_raw_data.csv', index=False)

print("Updated combined raw data with process names has been saved to updated_combined_raw_data.csv.")
'''
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.clusterers import Clusterer
import pandas as pd

# Start the JVM
jvm.start()

# Load the ARFF file
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("combined_raw_data.arff")

# Apply k-means clustering
kmeans = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])  # 3 clusters
kmeans.build_clusterer(data)

# Print cluster assignments and analyze how they got classified
cluster_assignments = []
for index, inst in enumerate(data):
    cluster = kmeans.cluster_instance(inst)
    cluster_assignments.append((index, cluster))
    print(f"Instance {index} is in cluster {cluster}")

# Load the combined raw data to analyze the clusters
raw_data = pd.read_csv('combined_raw_data.csv')

# Add cluster assignments to the raw data
raw_data['cluster'] = [cluster for _, cluster in cluster_assignments]

# Define the process data
process_data = pd.DataFrame({
    'process_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'process_name': ['cut_2_sheets_to_size', 'cut_sheet_of_larger_size', 'add_attachment_to_sheet', 'apply_indents_to_sheet', 'combine_sheets', 'cut_panel_to_size', 'quality_control_check', 'wrap_on_pallet']
})

# Merge the process data with the raw data to add process names
raw_data = raw_data.merge(process_data, on='process_id', how='left')

# Verify that the process_name_y column is present
print(raw_data.head())

# Analyze the clusters by grouping the data by cluster and calculating summary statistics
cluster_analysis = raw_data.groupby(['cluster', 'process_name_y']).agg({
    'process_id': 'count',
    'datetime_started': 'min',
    'datetime_ended': 'max',
    'equipment_required': 'mean',
    'num_staff_required': 'mean'
}).reset_index()

print(cluster_analysis)

# Stop the JVM at the very end
jvm.stop()

# Save the updated raw data with process names to a new CSV file
raw_data.to_csv('updated_combined_raw_data.csv', index=False)

print("Updated combined raw data with process names has been saved to updated_combined_raw_data.csv.")

'''
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.clusterers import Clusterer
import pandas as pd

# Start the JVM
jvm.start()

# Load the ARFF file
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("combined_raw_data.arff")

# Apply k-means clustering
kmeans = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])  # 3 clusters
kmeans.build_clusterer(data)

# Print cluster assignments and analyze how they got classified
cluster_assignments = []
for index, inst in enumerate(data):
    cluster = kmeans.cluster_instance(inst)
    cluster_assignments.append((index, cluster))
    print(f"Instance {index} is in cluster {cluster}")

# Load the combined raw data to analyze the clusters
raw_data = pd.read_csv('combined_raw_data.csv')

# Add cluster assignments to the raw data
raw_data['cluster'] = [cluster for _, cluster in cluster_assignments]

# Define the process data
process_data = pd.DataFrame({
    'process_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'process_name': ['cut_2_sheets_to_size', 'cut_sheet_of_larger_size', 'add_attachment_to_sheet', 'apply_indents_to_sheet', 'combine_sheets', 'cut_panel_to_size', 'quality_control_check', 'wrap_on_pallet']
})

# Merge the process data with the raw data to add process names
raw_data = raw_data.merge(process_data, on='process_id', how='left')

# Verify that the process_name column is present
print(raw_data.head())

# Analyze the clusters by grouping the data by cluster and process name and calculating summary statistics
cluster_analysis = raw_data.groupby(['cluster', 'process_name']).agg({
    'process_id': 'count',
    'equipment_required': 'mean',
    'num_staff_required': 'mean'
}).reset_index()

print(cluster_analysis)

# Stop the JVM at the very end
jvm.stop()

# Save the updated raw data with process names to a new CSV file
raw_data.to_csv('updated_combined_raw_data.csv', index=False)

print("Updated combined raw data with process names has been saved to updated_combined_raw_data.csv.")

'''
'''
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.clusterers import Clusterer
import pandas as pd

# Start the JVM
jvm.start()

# Load the ARFF file
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("combined_raw_data.arff")
# Remove the class attribute setting, as clustering does not require it
# data.class_is_last()

# Apply k-means clustering
kmeans = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])  # 3 clusters
kmeans.build_clusterer(data)

# Print cluster assignments and analyze how they got classified
cluster_assignments = []
for index, inst in enumerate(data):
    cluster = kmeans.cluster_instance(inst)
    cluster_assignments.append((index, cluster))
    print(f"Instance {index} is in cluster {cluster}")

# Load the combined raw data to analyze the clusters
raw_data = pd.read_csv('combined_raw_data.csv')

# Add cluster assignments to the raw data
raw_data['cluster'] = [cluster for _, cluster in cluster_assignments]

# Analyze the clusters by grouping the data by cluster and calculating summary statistics
cluster_analysis = raw_data.groupby('cluster').agg({
    'process_id': 'count',
    'datetime_started': 'min',
    'datetime_ended': 'max',
    'equipment_required': 'mean',
    'num_staff_required': 'mean'
}).reset_index()

print(cluster_analysis)

# Stop the JVM at the very end
jvm.stop()'
'''