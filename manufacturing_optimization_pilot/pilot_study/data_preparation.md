# Data Preparation

## Transformation Steps
1. Converted datetime strings to datetime objects
2. Calculated process duration in minutes
3. Removed unnecessary columns
4. Handled missing values

## Columns Removed
- data_id
- order_id
- process_id
- datetime_started
- datetime_ended
- ordered_datetime
- number_ordered
- equipment_required
- equipment_id

## New Features Created
- process_duration_minutes: Time taken to complete each process

## Final Dataset
- Number of records: 368
- Number of features: 6
- Features: ordered_height, ordered_width, process_name, num_staff_required, equipment_name, process_duration_minutes
