#!/bin/bash

# FI Order Header Validation Runner Script
# This script demonstrates how to use the validation tool

echo "========================================"
echo "FI Order Header Validation Tool"
echo "========================================"

# Set file paths - Update these with your actual file paths
SECOR_FILE1="secor_output_chelsea_new_20250821_1_prod.csv"
SECOR_FILE2="secor_output_chelsea_new_20250821_2_prod.csv"
ICEBERG_FILE="OH_iceberg_latest_Chelsea_2108.csv"

# Output directory for results
OUTPUT_DIR="validation_results_$(date +%Y%m%d)"

# Example 1: Run validation IGNORING the Reason column
echo ""
echo "Running validation with Reason column ignored..."
python order_header_validation.py \
    --secor-files "$SECOR_FILE1" "$SECOR_FILE2" \
    --iceberg-file "$ICEBERG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --ignore-reason

# Example 2: Run validation ignoring multiple columns
echo ""
echo "Running validation ignoring Reason and PromoCode columns..."
python order_header_validation.py \
    --secor-files "$SECOR_FILE1" "$SECOR_FILE2" \
    --iceberg-file "$ICEBERG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --ignore-columns Reason PromoCode \
    --export-differences

# Example 3: Run full validation with Reason column ignored and all exports
echo ""
echo "Running full validation with detailed exports (Reason ignored)..."
python order_header_validation.py \
    --secor-files "$SECOR_FILE1" "$SECOR_FILE2" \
    --iceberg-file "$ICEBERG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --ignore-reason \
    --export-differences \
    --export-missing \
    --tolerance 0.01

# Example 4: Run strict validation (no columns ignored)
echo ""
echo "Running strict validation (all columns checked)..."
python order_header_validation.py \
    --secor-files "$SECOR_FILE1" "$SECOR_FILE2" \
    --iceberg-file "$ICEBERG_FILE" \
    --output-dir "${OUTPUT_DIR}_strict" \
    --export-differences \
    --export-missing \
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
    LATEST_REPORT=$(ls -t "$OUTPUT_DIR"/validation_report_*.txt 2>/dev/null | head -1)
    if [ -f "$LATEST_REPORT" ]; then
        less "$LATEST_REPORT"
    else
        echo "Report file not found"
    fi
fi