import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.filters import Filter

# Start the JVM
jvm.start()

# Load the ARFF file
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file("combined_raw_data.arff")
data.class_is_last()

# Convert the class attribute to nominal if it is numeric
numeric_to_nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
numeric_to_nominal.inputformat(data)
data = numeric_to_nominal.filter(data)

# Apply J48 decision tree
classifier = Classifier(classname="weka.classifiers.trees.J48")
classifier.build_classifier(data)

# Evaluate the classifier with cross-validation
evaluation = Evaluation(data)
evaluation.crossvalidate_model(classifier, data, 10, Random(1))

# Print evaluation summary
print(evaluation.summary())

# Stop the JVM
jvm.stop()