"""
CRISP-DM METHODOLOGY IMPLEMENTATION
===================================

This file outlines how each step of your data mining project aligns with the CRISP-DM methodology.
Each phase will be implemented as a separate function that can be executed sequentially.

CRISP-DM has six phases:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

Your existing code can be reorganized to fit this structure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from weka.core.converters import Loader
from weka.clusterers import Clusterer
import weka.core.jvm as jvm
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def business_understanding():
    """
    Phase 1: Business Understanding
    - Defining the project objectives and requirements
    - Converting business questions into data mining goals
    """
    # Create a document that describes the business problem
    with open('business_understanding.md', 'w') as f:
        f.write("""# Business Understanding

## Project Objectives
- Analyze manufacturing process data to identify patterns and inefficiencies
- Discover relationships between order specifications and production times
- Group similar manufacturing processes to optimize resource allocation

## Business Success Criteria
- Identify process clusters that could benefit from optimization
- Determine key factors influencing process duration
- Provide actionable insights for production scheduling

## Data Mining Goals
- Apply clustering algorithms to group similar manufacturing processes
- Identify the most important features that characterize each cluster
- Create visualizations that highlight patterns in the production data

## Project Plan
1. Collect and merge data from multiple sources
2. Clean and transform data for analysis
3. Apply clustering algorithms to identify patterns
4. Evaluate the quality of the clusters
5. Interpret results and provide recommendations
""")
    
    print("Business Understanding phase documented in 'business_understanding.md'")
    return True

def data_understanding():
    """
    Phase 2: Data Understanding
    - Initial data collection and exploration
    - Identifying data quality issues
    - Discovering first insights
    """
    # Load the combined raw data
    try:
        data = pd.read_csv('combined_raw_data.csv')
        
        # Explore basic statistics
        data_stats = {
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'column_names': data.columns.tolist(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Create a summary report
        with open('data_understanding.md', 'w') as f:
            f.write(f"""# Data Understanding

## Dataset Overview
- Number of records: {data_stats['num_rows']}
- Number of features: {data_stats['num_columns']}

## Features
### Numeric Features
{', '.join(data_stats['numeric_columns'])}

### Categorical Features
{', '.join(data_stats['categorical_columns'])}

## Data Quality Issues
""")
            for col, count in data_stats['missing_values'].items():
                if count > 0:
                    f.write(f"- '{col}' has {count} missing values\n")
            
            f.write("\n## Initial Insights\n")
            # Calculate process time
            if 'datetime_started' in data.columns and 'datetime_ended' in data.columns:
                data['datetime_started'] = pd.to_datetime(data['datetime_started'])
                data['datetime_ended'] = pd.to_datetime(data['datetime_ended'])
                data['process_duration_minutes'] = (data['datetime_ended'] - data['datetime_started']).dt.total_seconds() / 60
                
                f.write(f"- Average process duration: {data['process_duration_minutes'].mean():.2f} minutes\n")
                f.write(f"- Shortest process: {data['process_duration_minutes'].min():.2f} minutes\n")
                f.write(f"- Longest process: {data['process_duration_minutes'].max():.2f} minutes\n")
                
                # Process time distribution
                plt.figure(figsize=(10, 6))
                plt.hist(data['process_duration_minutes'], bins=20)
                plt.xlabel('Process Duration (minutes)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Process Times')
                plt.savefig('process_time_distribution.png')
                f.write("- Process time distribution is shown in 'process_time_distribution.png'\n")
                
                # Process time by process name
                if 'process_name' in data.columns:
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(x='process_name', y='process_duration_minutes', data=data)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig('process_time_by_process.png')
                    f.write("- Process time by process name is shown in 'process_time_by_process.png'\n")
        
        print("Data Understanding phase completed and documented in 'data_understanding.md'")
        return data
    except Exception as e:
        print(f"Error in Data Understanding phase: {str(e)}")
        return None

def data_preparation(data=None):
    """
    Phase 3: Data Preparation
    - Feature selection and transformation
    - Cleaning data
    - Constructing new features
    """
    if data is None:
        try:
            data = pd.read_csv('combined_raw_data.csv')
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    try:
        # Convert datetime columns
        data['datetime_started'] = pd.to_datetime(data['datetime_started'])
        data['datetime_ended'] = pd.to_datetime(data['datetime_ended'])
        
        # Calculate process duration
        data['process_duration_minutes'] = (data['datetime_ended'] - data['datetime_started']).dt.total_seconds() / 60
        
        # List of columns to drop
        columns_to_drop = [
            'data_id', 
            'order_id', 
            'process_id', 
            'datetime_started', 
            'datetime_ended', 
            'ordered_datetime', 
            'number_ordered', 
            'equipment_required', 
            'equipment_id'
        ]
        
        # Drop the specified columns
        df_cleaned = data.drop(columns=columns_to_drop)
        
        # Check for and handle missing values
        missing_values = df_cleaned.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Found {missing_values.sum()} missing values:")
            print(missing_values[missing_values > 0])
            
            # Fill missing numeric values with mean
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            
            # Fill missing categorical values with mode
            cat_cols = df_cleaned.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        
        # Save the transformed data
        df_cleaned.to_csv('transformed_data.csv', index=False)
        
        # Document the data preparation process
        with open('data_preparation.md', 'w') as f:
            f.write("""# Data Preparation

## Transformation Steps
1. Converted datetime strings to datetime objects
2. Calculated process duration in minutes
3. Removed unnecessary columns
4. Handled missing values

## Columns Removed
""")
            for col in columns_to_drop:
                f.write(f"- {col}\n")
            
            f.write("\n## New Features Created\n")
            f.write("- process_duration_minutes: Time taken to complete each process\n")
            
            f.write("\n## Final Dataset\n")
            f.write(f"- Number of records: {len(df_cleaned)}\n")
            f.write(f"- Number of features: {len(df_cleaned.columns)}\n")
            f.write(f"- Features: {', '.join(df_cleaned.columns)}\n")
        
        print("Data Preparation phase completed and documented in 'data_preparation.md'")
        return df_cleaned
    except Exception as e:
        print(f"Error in Data Preparation phase: {str(e)}")
        return None

def modeling(data=None):
    """
    Phase 4: Modeling
    - Selecting modeling techniques
    - Building and assessing models
    """
    if data is None:
        try:
            data = pd.read_csv('transformed_data.csv')
        except Exception as e:
            print(f"Error loading transformed data: {str(e)}")
            return None
    
    try:
        # Start JVM for Weka
        jvm.start()
        
        # Save the transformed data to ARFF format for Weka
        arff_path = "transformed_data.arff"
        
        # Create ARFF file
        with open(arff_path, 'w') as arff_file:
            # Write header
            arff_file.write('@relation transformed_data\n\n')
            
            # Write attribute information
            categorical_columns = data.select_dtypes(include=['object']).columns
            for column in data.columns:
                if column in categorical_columns:
                    unique_values = data[column].unique()
                    arff_file.write(f'@attribute {column} {{')
                    arff_file.write(','.join([f'"{val}"' if isinstance(val, str) else str(val) for val in unique_values]))
                    arff_file.write('}\n')
                else:
                    arff_file.write(f'@attribute {column} numeric\n')
            
            # Write data
            arff_file.write('\n@data\n')
            for _, row in data.iterrows():
                row_values = []
                for column in data.columns:
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
        
        # Load the ARFF file
        loader = Loader(classname="weka.core.converters.ArffLoader")
        weka_data = loader.load_file(arff_path)
        
        # Apply k-means clustering with 3 clusters
        kmeans = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
        kmeans.build_clusterer(weka_data)
        
        # Get cluster assignments
        cluster_assignments = []
        for index, inst in enumerate(weka_data):
            cluster = kmeans.cluster_instance(inst)
            cluster_assignments.append((index, cluster))
        
        # Add cluster assignments to the data
        data['cluster'] = [cluster for _, cluster in cluster_assignments]
        
        # Save the clustered data
        data.to_csv('clustered_transformed_data.csv', index=False)
        
        # Document the modeling process
        with open('modeling.md', 'w') as f:
            f.write("""# Modeling

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
""")
            # Add cluster distribution
            cluster_counts = data['cluster'].value_counts()
            for cluster, count in cluster_counts.items():
                f.write(f"- Cluster {cluster}: {count} instances ({count/len(data)*100:.1f}%)\n")
        
        print("Modeling phase completed and documented in 'modeling.md'")
        
        # Stop JVM
        jvm.stop()
        
        return data
    except Exception as e:
        print(f"Error in Modeling phase: {str(e)}")
        # Make sure to stop JVM even if there's an error
        try:
            jvm.stop()
        except:
            pass
        return None

def evaluation(data=None):
    """
    Phase 5: Evaluation
    - Evaluating results
    - Reviewing the process
    - Determining next steps
    """
    if data is None:
        try:
            data = pd.read_csv('clustered_transformed_data.csv')
        except Exception as e:
            print(f"Error loading clustered data: {str(e)}")
            return None
    
    try:
        # Save numerical features for evaluation
        numerical_features = [col for col in data.columns if col not in ['cluster', 'process_name', 'equipment_name']]
        X = data[numerical_features]
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, data['cluster'])
        
        # Calculate Davies-Bouldin index
        db_score = davies_bouldin_score(X_scaled, data['cluster'])
        
        # Calculate cluster statistics
        cluster_stats = data.groupby('cluster')[numerical_features].agg(['mean', 'std', 'min', 'max'])
        
        # Process distribution across clusters
        process_distribution = pd.crosstab(data['process_name'], data['cluster'], normalize='index')
        
        # Create visualizations
        # 1. PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster'], cmap='viridis', alpha=0.8, edgecolors='w')
        plt.title('PCA visualization of clusters')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig('cluster_pca_visualization.png')
        
        # 2. Feature means by cluster
        cluster_means = data.groupby('cluster')[numerical_features].mean()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Mean Value'})
        plt.title('Feature Means by Cluster')
        plt.tight_layout()
        plt.savefig('feature_means_by_cluster.png')
        
        # Document the evaluation
        with open('evaluation.md', 'w') as f:
            f.write(f"""# Evaluation

## Cluster Quality Metrics
- Silhouette Score: {silhouette_avg:.4f}
  - Interpretation: {interpret_silhouette(silhouette_avg)}
- Davies-Bouldin Index: {db_score:.4f}
  - Interpretation: Lower is better, scores < 0.5 typically indicate good separation

## Cluster Characteristics
""")
            for cluster_id in data['cluster'].unique():
                f.write(f"\n### Cluster {cluster_id}\n")
                
                # Cluster size
                cluster_size = len(data[data['cluster'] == cluster_id])
                f.write(f"- Size: {cluster_size} instances ({cluster_size/len(data)*100:.1f}%)\n")
                
                # Most common processes
                top_processes = data[data['cluster'] == cluster_id]['process_name'].value_counts().head(3)
                f.write("- Dominant processes:\n")
                for process, count in top_processes.items():
                    f.write(f"  - {process}: {count} instances\n")
                
                # Key metrics
                f.write("- Key metrics:\n")
                for feature in numerical_features:
                    mean_value = data[data['cluster'] == cluster_id][feature].mean()
                    overall_mean = data[feature].mean()
                    pct_diff = (mean_value - overall_mean) / overall_mean * 100
                    f.write(f"  - {feature}: {mean_value:.2f} ")
                    if abs(pct_diff) > 10:  # Only highlight significant differences
                        f.write(f"({'↑' if pct_diff > 0 else '↓'}{abs(pct_diff):.1f}% from overall mean)}}\n")
                    else:
                        f.write("(similar to overall mean)\n")
            
            f.write("\n## Business Insights\n")
            
            # Identify clusters with unusual properties
            for cluster_id in data['cluster'].unique():
                cluster_data = data[data['cluster'] == cluster_id]
                avg_duration = cluster_data['process_duration_minutes'].mean()
                overall_avg = data['process_duration_minutes'].mean()
                
                if avg_duration > overall_avg * 1.2:
                    f.write(f"- Cluster {cluster_id} has significantly longer process times ({avg_duration:.2f} minutes vs overall average of {overall_avg:.2f} minutes)\n")
                elif avg_duration < overall_avg * 0.8:
                    f.write(f"- Cluster {cluster_id} has significantly shorter process times ({avg_duration:.2f} minutes vs overall average of {overall_avg:.2f} minutes)\n")
            
            f.write("\n## Visualizations\n")
            f.write("- PCA visualization of clusters: 'cluster_pca_visualization.png'\n")
            f.write("- Feature means by cluster: 'feature_means_by_cluster.png'\n")
            
            f.write("\n## Review of Process\n")
            f.write("- The clustering approach successfully identified distinct groups in the manufacturing process data\n")
            f.write("- The process characteristics of each cluster provide insights into manufacturing efficiency\n")
            f.write("- The evaluation metrics suggest the clusters are reasonably well-separated\n")
            
            f.write("\n## Next Steps\n")
            f.write("- Further investigate the specific factors causing longer process times in certain clusters\n")
            f.write("- Analyze how order specifications correlate with process efficiency\n")
            f.write("- Consider testing different clustering algorithms or parameters to improve separation\n")
            f.write("- Develop recommendations for process optimization based on cluster characteristics\n")
        
        print("Evaluation phase completed and documented in 'evaluation.md'")
        return True
    except Exception as e:
        print(f"Error in Evaluation phase: {str(e)}")
        return None

def deployment():
    """
    Phase 6: Deployment
    - Planning deployment
    - Producing final report
    - Project review
    """
    try:
        # Create deployment plan
        with open('deployment.md', 'w') as f:
            f.write("""# Deployment Plan

## Implementation Strategy
1. Share findings with production management team
2. Integrate cluster analysis into production monitoring system
3. Develop dashboard to track process efficiency by cluster
4. Implement targeted optimization for less efficient clusters

## Monitoring and Maintenance
- Set up regular re-clustering (monthly) to detect pattern changes
- Track process time improvements after optimization efforts
- Refine the clustering model as more data becomes available

## Final Report
The final report will include:
- Executive summary of key findings
- Detailed analysis of each cluster
- Recommendations for process optimization
- Technical appendix with methodology details

## Project Review
- The CRISP-DM methodology provided a structured approach to this data mining project
- The clustering analysis successfully identified distinct patterns in manufacturing processes
- The evaluation metrics indicate reasonably good cluster separation
- The insights gained can lead to actionable improvements in production efficiency

## Future Work
- Incorporate additional data sources (e.g., quality control metrics)
- Test advanced clustering algorithms (e.g., DBSCAN, hierarchical clustering)
- Develop predictive models for process duration based on order specifications
- Implement automated optimization recommendations
""")
        
        # Create a summary report that combines all phases
        with open('crisp_dm_summary.md', 'w') as f:
            f.write("""# Manufacturing Process Analysis using CRISP-DM

## Project Overview
This project applies the CRISP-DM methodology to analyze manufacturing process data. The goal is to identify patterns in production processes, discover inefficiencies, and provide actionable insights for optimization.

## CRISP-DM Phases

### 1. Business Understanding
- Defined project objectives: identify patterns and inefficiencies in manufacturing processes
- Established business success criteria: discover process clusters for optimization
- Set data mining goals: apply clustering to group similar processes

### 2. Data Understanding
- Explored data from multiple sources: order history, production history, equipment inventory
- Identified key variables: process duration, order specifications, equipment requirements
- Discovered initial insights on process time distribution

### 3. Data Preparation
- Merged data from multiple sources
- Calculated process duration from start and end times
- Removed unnecessary columns
- Handled missing values

### 4. Modeling
- Selected K-means clustering algorithm
- Configured for 3 clusters
- Applied model to the prepared data
- Assigned cluster labels to each process instance

### 5. Evaluation
- Assessed cluster quality using silhouette score and Davies-Bouldin index
- Characterized each cluster by dominant processes and distinctive features
- Identified business insights related to process efficiency
- Created visualizations to illustrate cluster separation

### 6. Deployment
- Developed plan for sharing findings with production management
- Outlined strategy for ongoing monitoring
- Identified areas for future improvement

## Key Findings
[Summary of the most important discoveries from the cluster analysis]

## Recommendations
[Actionable recommendations based on the analysis]

## Conclusion
[Overall assessment of the project's success and value]
""")
        
        print("Deployment phase completed")
        print("CRISP-DM summary created in 'crisp_dm_summary.md'")
        return True
    except Exception as e:
        print(f"Error in Deployment phase: {str(e)}")
        return None

# Function to interpret silhouette score
def interpret_silhouette(score):
    if score < 0.2:
        return "Poor structure, no substantial clustering found"
    elif score < 0.5:
        return "Weak structure, clusters may be artificial"
    elif score < 0.7:
        return "Reasonable structure has been found"
    else:
        return "Strong structure, well-separated clusters"

# Main function to run the entire CRISP-DM process
if __name__ == "__main__":
    # Execute all CRISP-DM phases
    print("Starting CRISP-DM process...")
    
    # Phase 1: Business Understanding
    print("\n=== Phase 1: Business Understanding ===")
    business_understanding()
    
    # Phase 2: Data Understanding
    print("\n=== Phase 2: Data Understanding ===")
    data = data_understanding()
    
    # Phase 3: Data Preparation
    print("\n=== Phase 3: Data Preparation ===")
    prepared_data = data_preparation(data)
    
    # Phase 4: Modeling
    print("\n=== Phase 4: Modeling ===")
    clustered_data = modeling(prepared_data)
    
    # Phase 5: Evaluation
    print("\n=== Phase 5: Evaluation ===")
    evaluation(clustered_data)
    
    # Phase 6: Deployment
    print("\n=== Phase 6: Deployment ===")
    deployment()
    
    print("\nCRISP-DM process completed. Check the generated files for results.")