import pandas as pd
from datetime import datetime

# Load the CSV file
df = pd.read_csv('combined_raw_data.csv')

# Calculate time difference between datetime_started and datetime_ended
# First convert string datetime to datetime objects
df['datetime_started'] = pd.to_datetime(df['datetime_started'])
df['datetime_ended'] = pd.to_datetime(df['datetime_ended'])

# Calculate the difference in minutes
df['process_duration_minutes'] = (df['datetime_ended'] - df['datetime_started']).dt.total_seconds() / 60

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
df_cleaned = df.drop(columns=columns_to_drop)

# Save the transformed data to a new CSV file
df_cleaned.to_csv('transformed_data.csv', index=False)

print("Transformation complete. Data saved to 'transformed_data.csv'")
print("Columns in the new file:", df_cleaned.columns.tolist())