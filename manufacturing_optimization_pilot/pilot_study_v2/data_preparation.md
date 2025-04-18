# Data Preparation

## Transformation Steps
1. Converted datetime strings to datetime objects
2. Calculated process duration in minutes
3. Extracted time-based features (hour, day of week, weekend indicator)
4. Removed unnecessary columns
5. Handled missing values

## New Temporal Features
- hour_started: Hour when the process started (0-23)
- day_of_week: Day of the week (0=Monday, 6=Sunday)
- is_weekend: Binary indicator for weekend (1) vs weekday (0)

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
- hour_started: Hour when the process started (0-23)
- day_of_week: Day of the week (0=Monday, 6=Sunday)
- is_weekend: Binary indicator for weekend (1) vs weekday (0)

## Final Dataset
- Number of records: 3032
- Number of features: 9
- Features: ordered_height, ordered_width, process_name, num_staff_required, equipment_name, process_duration_minutes, hour_started, day_of_week, is_weekend
