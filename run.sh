#!/bin/bash

# Process all video rows with GPU memory cleanup between each execution
# This replaces the pandas iterrows() loop with process-level memory isolation

set -e  # Exit on any error

PYTHON_SCRIPT="process_single_row.py"
MODE="$1"  # Mode is now required
CSV_TYPE="${2:-long}"  # CSV type (long/short), defaults to long
CSV_FILE="./assets/labels_${CSV_TYPE}.csv"

# Check if CSV exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: CSV file not found: $CSV_FILE"
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if mode is provided and valid
if [[ -z "$MODE" ]]; then
    echo "ERROR: Mode is required"
    echo "Usage: $0 <mode> [csv_type]"
    echo "  mode: 'default' or 'csv'"
    echo "  csv_type: 'long' or 'short' (default: 'long')"
    exit 1
fi

if [[ "$MODE" != "default" && "$MODE" != "csv" ]]; then
    echo "ERROR: Invalid mode: $MODE"
    echo "Usage: $0 <mode> [csv_type]"
    echo "  mode: 'default' or 'csv'"
    echo "  csv_type: 'long' or 'short' (default: 'long')"
    exit 1
fi

if [[ "$CSV_TYPE" != "long" && "$CSV_TYPE" != "short" ]]; then
    echo "ERROR: Invalid csv_type: $CSV_TYPE"
    echo "Usage: $0 <mode> [csv_type]"
    echo "  mode: 'default' or 'csv'"
    echo "  csv_type: 'long' or 'short' (default: 'long')"
    exit 1
fi

# Get total number of rows (subtract 1 for header)
TOTAL_ROWS=$(($(wc -l < "$CSV_FILE") - 1))
echo "Starting processing of $TOTAL_ROWS video rows in '$MODE' mode with '$CSV_TYPE' labels..."
echo "Using CSV file: $CSV_FILE"
echo "Each row will run in isolated process for complete GPU cleanup"

# Counter for progress tracking
COMPLETED=0
START_TIME=$(date +%s)

# Process each row in separate process
for ((row_index=0; row_index<TOTAL_ROWS; row_index++)); do
    echo ""
    echo "=== Processing row $((row_index + 1))/$TOTAL_ROWS ==="
    
    # Track timing for this row
    ROW_START=$(date +%s)
    
    # Run Python script for this specific row with mode and csv_type
    if python3 "$PYTHON_SCRIPT" "$row_index" "$MODE" "$CSV_TYPE"; then
        COMPLETED=$((COMPLETED + 1))
        ROW_END=$(date +%s)
        ROW_TIME=$((ROW_END - ROW_START))
        
        # Calculate ETA
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        if [ $COMPLETED -gt 0 ]; then
            AVG_TIME_PER_ROW=$((ELAPSED / COMPLETED))
            REMAINING_ROWS=$((TOTAL_ROWS - COMPLETED))
            ETA_SECONDS=$((REMAINING_ROWS * AVG_TIME_PER_ROW))
            
            # Format ETA
            ETA_HOURS=$((ETA_SECONDS / 3600))
            ETA_MINUTES=$(((ETA_SECONDS % 3600) / 60))
            ETA_SECS=$((ETA_SECONDS % 60))
            
            printf "SUCCESS: Row $((row_index + 1)) completed in ${ROW_TIME}s\n"
            printf "Progress: $COMPLETED/$TOTAL_ROWS ($(( COMPLETED * 100 / TOTAL_ROWS ))%%)\n"
            printf "ETA: ${ETA_HOURS}h ${ETA_MINUTES}m ${ETA_SECS}s (avg: ${AVG_TIME_PER_ROW}s/row)\n"
        else
            echo "SUCCESS: Row $((row_index + 1)) completed in ${ROW_TIME}s"
            echo "Progress: $COMPLETED/$TOTAL_ROWS ($(( COMPLETED * 100 / TOTAL_ROWS ))%)"
        fi
    else
        echo "ERROR: Row $((row_index + 1)) failed, continuing with next row..."
    fi
    
    # Brief pause to ensure complete cleanup
    sleep 1
done

echo ""
echo "Processing complete!"
echo "Successfully processed: $COMPLETED/$TOTAL_ROWS rows"

if [[ $COMPLETED -eq $TOTAL_ROWS ]]; then
    echo "All rows processed successfully!"
else
    echo "WARNING: Some rows failed. Check logs above for details."
fi

# Aggregate CSV results
echo ""
echo "Aggregating processing results..."
python3 -c "
import pandas as pd
import glob
import os
from datetime import datetime

# Find all log files for this mode and csv_type
log_pattern = './logs/processing_results_${MODE}_*.csv'
log_files = glob.glob(log_pattern)

if log_files:
    # Read and combine all log files
    all_results = []
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            all_results.append(df)
        except Exception as e:
            print(f'Warning: Could not read {log_file}: {e}')
    
    if all_results:
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Create final aggregated CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_csv = f'./logs/final_processing_summary_${MODE}_${CSV_TYPE}_{timestamp}.csv'
        combined_df.to_csv(final_csv, index=False)
        
        # Print summary statistics
        total_labels = len(combined_df)
        success_count = len(combined_df[combined_df['status'] == 'SUCCESS'])
        error_count = len(combined_df[combined_df['status'] == 'ERROR'])
        warning_count = len(combined_df[combined_df['status'] == 'WARNING'])
        
        segmentation_found = len(combined_df[combined_df['has_segmentation'] == True])
        
        print(f'')
        print(f'=== PROCESSING SUMMARY (${MODE} mode, ${CSV_TYPE} labels) ===')
        print(f'Total labels processed: {total_labels}')
        print(f'Successful: {success_count} ({success_count/total_labels*100:.1f}%)')
        print(f'Errors: {error_count} ({error_count/total_labels*100:.1f}%)')
        print(f'Warnings: {warning_count} ({warning_count/total_labels*100:.1f}%)')
        print(f'Labels with segmentation: {segmentation_found} ({segmentation_found/total_labels*100:.1f}%)')
        print(f'')
        print(f'Detailed results saved to: {final_csv}')
    else:
        print('No valid log files found to aggregate.')
else:
    print('No log files found for mode ${MODE} with ${CSV_TYPE} labels.')
"
