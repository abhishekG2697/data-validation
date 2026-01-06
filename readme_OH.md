# FI Order Header Validation Tool

## Overview
This validation tool compares Secor (legacy) and Iceberg (new) outputs for FI Order Header feed migration. It identifies discrepancies at both row and column levels, with configurable tolerance for numeric comparisons.

## Features
- ✅ **Comprehensive Validation**: Row-level and column-level comparison
- ✅ **Numeric Tolerance**: Configurable tolerance for decimal values (default: 0.01)
- ✅ **Multiple Input Support**: Handles multiple Secor partition files
- ✅ **Detailed Reporting**: Text reports, CSV exports, and JSON results
- ✅ **Missing Record Detection**: Identifies orders present in one system but not the other
- ✅ **Leadership-Ready Output**: Professional reports suitable for stakeholder review

## Installation

### Requirements
Create a file named `requirements.txt` with:
```
pandas>=1.3.0
numpy>=1.21.0
```

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x order_header_validation.py
chmod +x run_validation.sh
```

## Usage

### Basic Usage
```bash
python order_header_validation.py \
    --secor-files secor_partition1.csv secor_partition2.csv \
    --iceberg-file iceberg_output.csv \
    --output-dir validation_results
```

### Ignoring the Reason Column (Recommended for initial validation)
```bash
# Using the shortcut flag
python order_header_validation.py \
    --secor-files secor_*.csv \
    --iceberg-file iceberg_output.csv \
    --ignore-reason \
    --output-dir validation_results

# Or using the generic ignore-columns option
python order_header_validation.py \
    --secor-files secor_*.csv \
    --iceberg-file iceberg_output.csv \
    --ignore-columns Reason \
    --output-dir validation_results
```

### Ignoring Multiple Columns
```bash
python order_header_validation.py \
    --secor-files secor_*.csv \
    --iceberg-file iceberg_output.csv \
    --ignore-columns Reason PromoCode PaymentType \
    --output-dir validation_results
```

### Advanced Usage with All Options
```bash
python order_header_validation.py \
    --secor-files secor_*.csv \
    --iceberg-file iceberg_output.csv \
    --output-dir validation_results \
    --tolerance 0.01 \
    --ignore-reason \
    --export-differences \
    --export-missing
```

### Using the Wrapper Script
```bash
# Edit the file paths in run_validation.sh first
./run_validation.sh
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--secor-files` | Path(s) to Secor CSV files (space-separated) | Required |
| `--iceberg-file` | Path to Iceberg CSV file | Required |
| `--tolerance` | Decimal tolerance for numeric comparisons | 0.01 |
| `--ignore-columns` | Column names to ignore during comparison (space-separated) | None |
| `--ignore-reason` | Shortcut flag to ignore the Reason column | False |
| `--output-dir` | Directory for saving results | validation_results |
| `--export-differences` | Export detailed differences to CSV | False |
| `--export-missing` | Export lists of missing orders | False |

### Column Ignore Feature

The validation script supports ignoring specific columns during comparison. This is useful when:
- Certain columns have known issues that will be fixed later
- Columns contain data that is expected to differ temporarily
- You want to focus on critical columns first

**Examples:**
```bash
# Ignore just the Reason column
python order_header_validation.py ... --ignore-reason

# Ignore multiple columns
python order_header_validation.py ... --ignore-columns Reason PromoCode

# Run strict validation (no columns ignored)
python order_header_validation.py ... 
```

When columns are ignored:
- They won't appear in the differences report
- They won't affect the match rate calculation
- The report will clearly show which columns were ignored

## Input File Format

The script expects pipe-delimited (`|`) CSV files with the following 17 columns:
1. OrderID
2. OrderNo
3. CustomerID
4. OrderDateUpdated
5. ShipDateToSend
6. TransactionType
7. SalesChannel
8. GoodsNetGBP
9. GoodsVatGBP
10. ShippingNetGBP
11. ShippingVatGBP
12. Currency
13. ConversionRate
14. ShippingCountry
15. Reason
16. PaymentType
17. PromoCode

## Output Files

The tool generates multiple output files in the specified directory:

### 1. Validation Report (validation_report_TIMESTAMP.txt)
```
📈 VALIDATION SUMMARY
----------------------------------------
Total Secor Records:    2,188
Total Iceberg Records:  2,192
Common Orders:          2,188
Perfect Matches:        2,180
Orders with Differences: 8
Match Rate:             99.63%
```

### 2. Differences CSV (differences_TIMESTAMP.csv)
Detailed breakdown of all field-level differences:
```csv
OrderNo,Column,Secor_Value,Iceberg_Value,Type,Numeric_Difference
21-9267-9569,GoodsVatGBP,14.16,14.17,numeric,0.01
```

### 3. Missing Orders Lists
- `orders_missing_in_iceberg.txt`: Orders in Secor but not in Iceberg
- `orders_missing_in_secor.txt`: Orders in Iceberg but not in Secor

### 4. JSON Results (validation_results_TIMESTAMP.json)
Machine-readable format for integration with other tools

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Validation passed - no differences found |
| 1 | Validation completed with differences |
| 2 | Validation failed with errors |

## Validation Logic

### Numeric Comparison
- Numeric fields are compared with configurable tolerance (default: 0.01)
- Fields: GoodsNetGBP, GoodsVatGBP, ShippingNetGBP, ShippingVatGBP, ConversionRate
- Example: 14.16 and 14.17 are considered matching with 0.01 tolerance

### String Comparison
- Exact match after trimming whitespace
- Quotes are stripped before comparison
- Empty strings and NULL values are treated as equivalent

### Missing Records
- Uses OrderNo as the unique identifier
- Records present in one dataset but not the other are flagged

## Example Output

### Sample Validation Report
```
================================================
FI ORDER HEADER VALIDATION REPORT
================================================
Generated: 2025-09-02 14:30:45

⚙️  VALIDATION CONFIGURATION
----------------------------------------
Decimal Tolerance: 0.01
Ignored Columns: Reason

📈 VALIDATION SUMMARY
----------------------------------------
Total Secor Records:    2,188
Total Iceberg Records:  2,192
Common Orders:          2,188
Perfect Matches:        2,180
Orders with Differences: 8
Match Rate:             99.63%

⚠️  ORDERS MISSING IN SECOR (4)
----------------------------------------
  1. 123-4567-8901
  2. 234-5678-9012
  3. 345-6789-0123
  4. 456-7890-1234

📊 COLUMN MISMATCH SUMMARY
----------------------------------------
  GoodsVatGBP         : 8 mismatches (0.37%)
  PaymentType         : 2 mismatches (0.09%)

🔍 SAMPLE ORDER DIFFERENCES (First 5)
----------------------------------------
Order: 21-9267-9569
  GoodsVatGBP:
    Secor:   14.16
    Iceberg: 14.17
    Diff:    0.0100
```

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Ensure all file paths are correct
   - Use absolute paths if running from different directory

2. **Memory Error with Large Files**
   - Process files in smaller batches
   - Increase system memory allocation

3. **Encoding Issues**
   - Ensure files are UTF-8 encoded
   - Use `iconv` to convert if needed: `iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv`

4. **Delimiter Issues**
   - Verify files use pipe delimiter (`|`)
   - Check for pipes within quoted fields

## Integration with CI/CD

### Jenkins Example
```groovy
stage('Data Validation') {
    steps {
        sh '''
            python order_header_validation.py \
                --secor-files ${SECOR_FILES} \
                --iceberg-file ${ICEBERG_FILE} \
                --output-dir ${WORKSPACE}/validation_results \
                --export-differences
        '''
    }
    post {
        always {
            archiveArtifacts artifacts: 'validation_results/**/*'
            publishHTML([
                reportDir: 'validation_results',
                reportFiles: 'validation_report_*.txt',
                reportName: 'Order Header Validation Report'
            ])
        }
    }
}
```

### GitHub Actions Example
```yaml
- name: Run Order Header Validation
  run: |
    python order_header_validation.py \
      --secor-files ${{ env.SECOR_FILES }} \
      --iceberg-file ${{ env.ICEBERG_FILE }} \
      --output-dir validation_results \
      --export-differences
      
- name: Upload Validation Results
  uses: actions/upload-artifact@v2
  with:
    name: validation-results
    path: validation_results/
```

## Performance Considerations

- **Small datasets (<100K records)**: ~1-2 seconds
- **Medium datasets (100K-1M records)**: ~5-30 seconds  
- **Large datasets (>1M records)**: Consider chunking or parallel processing

## Support

For issues or questions:
- Create a ticket in JIRA with tag `fi-feed-migration`
- Contact the Data Engineering team
- Refer to the migration documentation in Confluence

## Version History

- **v1.0** (2025-09-02): Initial release
  - Basic validation functionality
  - Support for multiple Secor files
  - Configurable numeric tolerance
  - Comprehensive reporting
