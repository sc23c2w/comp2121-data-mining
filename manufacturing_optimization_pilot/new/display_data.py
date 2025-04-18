import pandas as pd
import matplotlib.pyplot as plt

# Load the production data
production_data = pd.read_csv('combined_raw_data.csv')

# Display the first few rows of the data
# print(production_data.head())

# Visualize the distribution of process times
production_data['process_time'] = (pd.to_datetime(production_data['datetime_ended']) - pd.to_datetime(production_data['datetime_started'])).dt.seconds / 60
plt.hist(production_data['process_time'], bins=20)
plt.xlabel('Process Time (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Process Times')
#plt.show()
plt.savefig('process_time_distribution.png')

# Calculate average process time for each process_id
average_process_time = production_data.groupby('process_id')['process_time'].mean()
print(average_process_time)

# Identify processes with high variability
process_variability = production_data.groupby('process_id')['process_time'].std()
print(process_variability)

'''# Example of parallel processing'''
'''
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

def process_panel(panel_id, process_id, start_time):
    process_time = randomize_time(process_times[process_id])
    end_time = start_time + timedelta(minutes=process_time)
    return [panel_id, process_id, start_time, end_time]

# Example of parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for panel_id in range(1, 21):  # Assuming 20 panels
        for process_id in range(1, 9):
            futures.append(executor.submit(process_panel, panel_id, process_id, datetime.now()))
    
    results = [future.result() for future in futures]
    print(results)'
'''