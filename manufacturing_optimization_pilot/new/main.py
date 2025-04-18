import pandas as pd
import arff  # liac-arff library
from sklearn.preprocessing import LabelEncoder
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.clusterers import Clusterer
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.filters import Filter

# Start the JVM once
jvm.start()

# Load the combined raw data
raw_data = pd.read_csv('combined_raw_data.csv')

# Preprocess the data: Convert string attributes to numeric
label_encoders = {}
for col in raw_data.columns:
    if pd.api.types.is_string_dtype(raw_data[col]):
        le = LabelEncoder()
        raw_data[col] = le.fit_transform(raw_data[col])
        label_encoders[col] = le  # Save the encoder for potential inverse transformation

# Determine attribute types
attributes = []
for col in raw_data.columns:
    if pd.api.types.is_numeric_dtype(raw_data[col]):
        attributes.append((col, 'NUMERIC'))
    elif pd.api.types.is_datetime64_any_dtype(raw_data[col]):
        attributes.append((col, 'DATE "yyyy-MM-dd HH:mm:ss"'))  # Adjust format as needed
    else:
        raise ValueError(f"Unsupported data type in column: {col}")

# Save the data to ARFF format
arff_data = {
    'description': '',
    'relation': 'combined_raw_data',
    'attributes': attributes,
    'data': raw_data.values.tolist(),
}

with open('combined_raw_data.arff', 'w') as f:
    arff.dump(arff_data, f)

# Load the ARFF file
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("combined_raw_data.arff")

# Apply k-means clustering
kmeans = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])  # 3 clusters
kmeans.build_clusterer(data)

# Print cluster assignments
for index, inst in enumerate(data):
    cluster = kmeans.cluster_instance(inst)
    print(f"Instance {index} is in cluster {cluster}")

# Ensure the class attribute is nominal
data.class_is_last()
numeric_to_nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
numeric_to_nominal.inputformat(data)
data = numeric_to_nominal.filter(data)

# Apply J48 decision tree
classifier = Classifier(classname="weka.classifiers.trees.J48")
classifier.build_classifier(data)

# Evaluate the classifier
evaluation = Evaluation(data)
evaluation.crossvalidate_model(classifier, data, 10, Random(1))

# Print evaluation summary
print(evaluation.summary())

# Stop the JVM at the end
jvm.stop()