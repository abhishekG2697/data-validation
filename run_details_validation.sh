#!/bin/bash

# FI Order Details Validation Runner Script
# This script demonstrates how to use the Order Details validation tool

echo "========================================"
echo "FI Order Details Validation Tool"
echo "========================================"

# Set file paths - Update these with your actual file paths
SECOR_DETAILS_FILE1="secor_output_details_chelsea_new_20250821_1_prod.csv"
SECOR_DETAILS_FILE2="secor_output_details_chelsea_new_20250821_2_prod.csv"
ICEBERG_DETAILS_FILE="OD_iceberg_latest_Chelsea_2108.csv"

# Output directory for results
OUTPUT_DIR="validation_results_details_$(date +%Y%m%d)"

# Example 1: Run basic validation
echo ""
echo "Running basic Order Details validation..."
python order_details_validation.py \
    --secor-files "$SECOR_DETAILS_FILE1" "$SECOR_DETAILS_FILE2" \
    --iceberg-file "$ICEBERG_DETAILS_FILE" \
    --output-dir "$OUTPUT_DIR"

# Example 2: Run validation ignoring specific columns
echo ""
echo "Running validation ignoring VariationID column..."
python order_details_validation.py \
    --secor-files "$SECOR_DETAILS_FILE1" "$SECOR_DETAILS_FILE2" \
    --iceberg-file "$ICEBERG_DETAILS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --ignore-columns VariationID \
    --export-differences

# Example 3: Run full validation with all exports
echo ""
echo "Running full validation with detailed exports..."
python order_details_validation.py \
    --secor-files "$SECOR_DETAILS_FILE1" "$SECOR_DETAILS_FILE2" \
    --iceberg-file "$ICEBERG_DETAILS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --export-differences \
    --export-missing \
    --export-order-summary \
    --tolerance 0.01

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Validation PASSED - No differences found!"
elif [ $? -eq 1 ]; then
    echo ""
    echo "⚠️  Validation completed with differences found."
    echo "Check the report in: $OUTPUT_DIR"
else
    echo ""
    echo "❌ Validation failed with errors."
fi

echo ""
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"

# Optional: Display the report
echo ""
read -p "Do you want to view the report now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Find the latest report file
    LATEST_REPORT=$(ls -t "$OUTPUT_DIR"/details_validation_report_*.txt 2>/dev/null | head -1)
    if [ -f "$LATEST_REPORT" ]; then
        less "$LATEST_REPORT"
    else
        echo "Report file not found"
    fi
fi