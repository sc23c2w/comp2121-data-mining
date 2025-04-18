import pandas as pd

# Load the order history data
order_history = pd.read_csv('order_history.csv')

# Load the production history data
production_history = pd.read_csv('production_data.csv')

# Define the equipment inventory data
equipment_inventory = pd.DataFrame({
    'equipment_id': [1, 2, 3, 4, 5],
    'equipment_name': ['cutting_table', 'CNC_machine', 'heating_element', 'indenting_machine', 'pallet']
})

# Define the process data
process_data = pd.DataFrame({
    'process_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'process_name': ['cut_sheet_to_size', 'cut_sheet_of_larger_size', 'add_attachment_to_sheet', 'apply_indents_to_sheet', 'combine_sheets', 'cut_panel_to_size', 'quality_control_check', 'wrap_on_pallet'],
    'equipment_required': [2, 2, 1, 4, 3, 1, 1, 5],
    'num_staff_required': [1, 1, 1, 2, 2, 1, 1, 2]
})

# Merge the dataframes into one raw data file
raw_data = production_history.merge(order_history, on='order_id')
raw_data = raw_data.merge(process_data, on='process_id')
raw_data = raw_data.merge(equipment_inventory, left_on='equipment_required', right_on='equipment_id')

# Save the combined raw data to a CSV file
raw_data.to_csv('combined_raw_data.csv', index=False)

print("Combined raw data has been saved to combined_raw_data.csv.")