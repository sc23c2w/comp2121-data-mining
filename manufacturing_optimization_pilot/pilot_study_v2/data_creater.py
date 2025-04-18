import csv
import random
import datetime
from datetime import timedelta

# Define the manufacturing processes from the table
processes = [
    {"id": 1, "name": "cut_sheet_to_size", "staff": 1, "equipment": "CNC Machine", "time": 10},
    {"id": 2, "name": "cut_sheet_of_larger_size", "staff": 1, "equipment": "CNC Machine", "time": 5},
    {"id": 3, "name": "add_attachment_to_sheet", "staff": 1, "equipment": "Cutting Table", "time": 10},
    {"id": 4, "name": "apply_indents_to_sheet", "staff": 2, "equipment": "Indenting Machine", "time": 8},
    {"id": 5, "name": "combine_sheets", "staff": 2, "equipment": "Heating Element", "time": 15},
    {"id": 6, "name": "cut_panel_to_size", "staff": 1, "equipment": "Cutting Table", "time": 5},
    {"id": 7, "name": "quality_control_check", "staff": 1, "equipment": "Cutting Table", "time": 5},
    {"id": 8, "name": "wrap_on_pallet", "staff": 2, "equipment": "Pallet", "time": 2}
]

# Define equipment IDs
equipment_ids = {
    "CNC Machine": ["CNC001", "CNC002", "CNC003"],
    "Cutting Table": ["CT001", "CT002", "CT003", "CT004"],
    "Indenting Machine": ["IM001", "IM002"],
    "Heating Element": ["HE001", "HE002", "HE003"],
    "Pallet": ["PAL001", "PAL002", "PAL003", "PAL004", "PAL005"]
}

# Create the CSV file
with open('combined_raw_data.csv', 'w', newline='') as csvfile:
    fieldnames = [
        'data_id', 'order_id', 'process_id', 'datetime_started', 'datetime_ended',
        'ordered_datetime', 'ordered_height', 'ordered_width', 'number_ordered',
        'process_name', 'equipment_required', 'num_staff_required',
        'equipment_id', 'equipment_name'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    data_id = 1
    num_orders = 50  # Adjust to ensure we get at least 350 records
    
    # Generate random start date within the last year
    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    
    # Staff availability tracking
    staff_availability = {}
    
    for order_id in range(1, num_orders + 1):
        # Generate order details
        num_panels = random.randint(1, 15)
        ordered_datetime = start_date + datetime.timedelta(days=random.randint(0, 350))
        # Updated height and width ranges (0.2 to 6.1)
        ordered_height = round(random.uniform(0.2, 6.1), 1)
        ordered_width = round(random.uniform(0.2, 6.1), 1)

        
        # Production starts 1-5 days after ordering
        production_start = ordered_datetime + datetime.timedelta(days=random.randint(1, 5))
        
        # Process each panel
        for panel_num in range(1, num_panels + 1):
            # Initialize the start time for this panel
            current_time = production_start + datetime.timedelta(hours=random.randint(0, 8))
            
            # Go through each process for this panel
            for process in processes:
                # Occasionally vary staff requirements (20% chance)
                if random.random() < 0.2:
                    staff_required = process["staff"] + random.choice([1, 2])
                else:
                    staff_required = process["staff"]
                
                # Occasionally vary process time (30% chance)
                if random.random() < 0.3:
                    process_time = process["time"] * random.uniform(0.8, 1.5)
                else:
                    process_time = process["time"]
                
                # Round process time to nearest minute
                process_time = round(process_time)
                
                # Find a time when enough staff are available
                can_start = False
                while not can_start:
                    # Check if staff are available at this time
                    staff_count = 0
                    for staff_id in range(1, 6):  # Max 5 staff total
                        if staff_id not in staff_availability or staff_availability[staff_id] <= current_time:
                            staff_count += 1
                    
                    if staff_count >= staff_required:
                        # Enough staff are available
                        can_start = True
                        
                        # Allocate staff
                        allocated_staff = 0
                        for staff_id in range(1, 6):
                            if allocated_staff < staff_required:
                                if staff_id not in staff_availability or staff_availability[staff_id] <= current_time:
                                    staff_availability[staff_id] = current_time + datetime.timedelta(minutes=process_time)
                                    allocated_staff += 1
                    else:
                        # Not enough staff, wait a bit
                        current_time += datetime.timedelta(minutes=15)
                
                # Calculate end time
                end_time = current_time + datetime.timedelta(minutes=process_time)
                
                # Select equipment
                equipment_name = process["equipment"]
                equipment_id = random.choice(equipment_ids[equipment_name])
                
                # Write record to CSV
                writer.writerow({
                    'data_id': data_id,
                    'order_id': order_id,
                    'process_id': process["id"],
                    'datetime_started': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'datetime_ended': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'ordered_datetime': ordered_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    'ordered_height': ordered_height,
                    'ordered_width': ordered_width,
                    'number_ordered': num_panels,
                    'process_name': process["name"],
                    'equipment_required': equipment_name,
                    'num_staff_required': staff_required,
                    'equipment_id': equipment_id,
                    'equipment_name': equipment_name
                })
                
                data_id += 1
                
                # Update time for next process
                current_time = end_time

print(f"CSV file created with {data_id-1} records.")