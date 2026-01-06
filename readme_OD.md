# FI Order Details Validation - Complete Setup Guide

## 📋 Overview
This guide provides step-by-step instructions to set up and run the Order Details validation script for comparing Secor and Iceberg outputs.

## 🚀 Quick Start

### Step 1: Save the Scripts
Save these files in your validation directory:
- `order_details_validation.py` - Main validation script
- `run_details_validation.sh` - Shell wrapper script

### Step 2: Install Dependencies
```bash
# Create requirements.txt file
cat > requirements.txt << EOF
pandas>=1.3.0
numpy>=1.21.0
EOF

# Install Python packages
pip install -r requirements.txt
```

### Step 3: Make Scripts Executable
```bash
chmod +x order_details_validation.py
chmod +x run_details_validation.sh
```

### Step 4: Run Your First Validation
```bash
python3 order_details_validation.py \
    --secor-files /path/to/secor_details_part1.csv /path/to/secor_details_part2.csv \
    --iceberg-file /path/to/iceberg_details.csv \
    --output-dir validation_output_OD_Chelsea
```

---

## 📁 File Format Requirements

### Expected Column Structure (9 columns)
The script expects pipe-delimited (`|`) CSV files with these columns in order:
1. **OrderID** - Unique identifier for the order
2. **OrderNo** - Order number (used as primary key with LineNumber)
3. **LineNumber** - Line item number within the order
4. **ConversionRate** - Currency conversion rate
5. **ProductID** - Product identifier
6. **VariationID** - Product variation identifier
7. **UnitPriceGBP** - Unit price in GBP
8. **UnitVatGBP** - VAT amount per unit in GBP
9. **Quantity** - Quantity ordered

### Sample Data Format
```
"ORD123"|"21-9267-9569"|"1"|"1.0"|"PROD456"|"VAR789"|"25.50"|"5.10"|"2"
"ORD123"|"21-9267-9569"|"2"|"1.0"|"PROD457"|"VAR790"|"15.00"|"3.00"|"1"
```

---

## 💻 Command Line Usage

### Basic Validation
```bash
python3 order_details_validation.py \
    --secor-files secor_details.csv \
    --iceberg-file iceberg_details.csv
```

### With Multiple Secor Files (Partitions)
```bash
python3 order_details_validation.py \
    --secor-files secor_part1.csv secor_part2.csv secor_part3.csv \
    --iceberg-file iceberg_details.csv \
    --output-dir my_validation_results
```

### Ignoring Specific Columns
```bash
# Ignore VariationID column if it has known issues
python3 order_details_validation.py \
    --secor-files secor_details_*.csv \
    --iceberg-file iceberg_details.csv \
    --ignore-columns VariationID
```

### Full Validation with All Exports
```bash
python3 order_details_validation.py \
    --secor-files secor_details_*.csv \
    --iceberg-file iceberg_details.csv \
    --output-dir validation_results \
    --tolerance 0.01 \
    --export-differences \
    --export-missing \
    --export-order-summary
```

---

## ⚙️ Configuration Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--secor-files` | Secor CSV file(s) path | Required | `file1.csv file2.csv` |
| `--iceberg-file` | Iceberg CSV file path | Required | `iceberg.csv` |
| `--tolerance` | Decimal tolerance for numeric fields | 0.01 | `--tolerance 0.001` |
| `--ignore-columns` | Columns to skip in validation | None | `--ignore-columns VariationID` |
| `--output-dir` | Directory for results | validation_results_details | `--output-dir my_results` |
| `--export-differences` | Export differences to CSV | False | `--export-differences` |
| `--export-missing` | Export missing items list | False | `--export-missing` |
| `--export-order-summary` | Export order-level stats | False | `--export-order-summary` |

---

## 📊 Understanding the Output

### 1. Console Output
```
📂 Loading Order Details data files...
✓ Read 5,432 line items from secor_details_part1.csv
✓ Read 3,891 line items from secor_details_part2.csv
Total Secor line items: 9,323
Unique orders in Secor: 2,188

✓ Read 9,327 line items from iceberg_details.csv
Unique orders in Iceberg: 2,192

============================================================
STARTING ORDER DETAILS VALIDATION
============================================================

📊 Line Item Count Summary:
   Secor line items: 9,323
   Iceberg line items: 9,327
   Common line items: 9,323
   Missing in Iceberg: 0
   Missing in Secor: 4

📦 Order-Level Summary:
   Unique orders in Secor: 2,188
   Unique orders in Iceberg: 2,192
   Orders only in Secor: 0
   Orders only in Iceberg: 4
```

### 2. Generated Files

#### A. Validation Report (`details_validation_report_TIMESTAMP.txt`)
```
================================================
FI ORDER DETAILS VALIDATION REPORT
================================================
Generated: 2025-09-02 15:45:30

⚙️  VALIDATION CONFIGURATION
----------------------------------------
Decimal Tolerance: 0.01
Ignored Columns: None

📈 VALIDATION SUMMARY
----------------------------------------
Total Secor Line Items:    9,323
Total Iceberg Line Items:  9,327
Common Line Items:         9,323
Perfect Matches:           9,315
Items with Differences:    8
Match Rate:                99.91%

📦 ORDER-LEVEL STATISTICS
----------------------------------------
Unique Orders (Secor):     2,188
Unique Orders (Iceberg):   2,192
Avg Items/Order (Secor):   4.26
Avg Items/Order (Iceberg): 4.25
Max Items/Order (Secor):   15
Max Items/Order (Iceberg): 15
```

#### B. Differences CSV (`details_differences_TIMESTAMP.csv`)
```csv
OrderNo,LineNumber,Column,Secor_Value,Iceberg_Value,Type,Numeric_Difference
21-9267-9569,1,UnitVatGBP,5.10,5.11,numeric,0.01
21-9267-9569,2,ProductID,PROD457,PROD458,string,
```

#### C. Missing Items Lists
- `line_items_missing_in_iceberg.txt` - Line items in Secor but not in Iceberg
- `line_items_missing_in_secor.txt` - Line items in Iceberg but not in Secor

#### D. JSON Results (`details_validation_results_TIMESTAMP.json`)
Machine-readable format containing all validation results

---

## 🎯 Real-World Usage Examples

### Example 1: Daily Validation Run
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
python3 order_details_validation.py \
    --secor-files /data/secor/details_${DATE}_*.csv \
    --iceberg-file /data/iceberg/details_${DATE}.csv \
    --output-dir /reports/validation_${DATE} \
    --export-differences \
    --tolerance 0.01
```

### Example 2: Your Specific Use Case
Based on your file paths:
```bash
python3 order_details_validation.py \
    --secor-files /Users/agarimella/partner_feeds_validation/data/OD/secor_details_chelsea_*.csv \
    --iceberg-file /Users/agarimella/partner_feeds_validation/data/OD/OD_iceberg_latest_Chelsea.csv \
    --output-dir validation_output_OD_Chelsea \
    --export-differences \
    --export-missing
```

### Example 3: Parallel Validation (Headers + Details)
```bash
#!/bin/bash
# Run both validations in parallel
echo "Starting Order Header validation..."
python3 order_header_validation.py \
    --secor-files secor_header_*.csv \
    --iceberg-file iceberg_header.csv \
    --ignore-reason \
    --output-dir validation_OH &

echo "Starting Order Details validation..."
python3 order_details_validation.py \
    --secor-files secor_details_*.csv \
    --iceberg-file iceberg_details.csv \
    --output-dir validation_OD &

# Wait for both to complete
wait
echo "Both validations completed!"
```

---

## 🔍 Key Differences from Order Header Validation

| Aspect | Order Header | Order Details |
|--------|--------------|---------------|
| **Primary Key** | OrderNo | OrderNo + LineNumber |
| **Record Type** | One per order | Multiple per order |
| **Columns** | 17 columns | 9 columns |
| **Validation Focus** | Order-level totals | Line-item accuracy |
| **Statistics** | Order counts | Line items + Order stats |

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "No such file or directory"
```bash
# Check file exists and path is correct
ls -la /path/to/your/file.csv
```

#### 2. "Invalid column names"
```bash
# Check your file structure
head -1 your_file.csv | tr '|' '\n' | nl
```

#### 3. Memory issues with large files
```bash
# Process in smaller chunks or increase memory
export PYTHONHASHSEED=0
python3 -u order_details_validation.py ...
```

#### 4. Encoding issues
```bash
# Convert to UTF-8 if needed
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
```

---

## 📈 Integration with CI/CD

### Jenkins Pipeline
```groovy
stage('Order Details Validation') {
    steps {
        sh '''
            python3 order_details_validation.py \
                --secor-files ${WORKSPACE}/data/secor_details_*.csv \
                --iceberg-file ${WORKSPACE}/data/iceberg_details.csv \
                --output-dir ${WORKSPACE}/validation_results \
                --export-differences \
                --tolerance 0.01
        '''
    }
    post {
        always {
            archiveArtifacts artifacts: 'validation_results/**/*'
            publishHTML([
                reportDir: 'validation_results',
                reportFiles: 'details_validation_report_*.txt',
                reportName: 'Order Details Validation Report'
            ])
        }
    }
}
```

### GitHub Actions
```yaml
name: Order Details Validation

on:
  schedule:
    - cron: '0 2 * * *'  # Run daily at 2 AM

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install pandas numpy
      
      - name: Run Order Details Validation
        run: |
          python3 order_details_validation.py \
            --secor-files data/secor_*.csv \
            --iceberg-file data/iceberg.csv \
            --output-dir results \
            --export-differences
      
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: validation-results
          path: results/
```

---

## 📊 Interpreting Results

### Success Indicators
- ✅ Match Rate > 99%
- ✅ No missing line items
- ✅ Numeric differences within tolerance
- ✅ Consistent order counts

### Warning Signs
- ⚠️ Match Rate < 95%
- ⚠️ Many missing line items
- ⚠️ Systematic column mismatches
- ⚠️ Different item counts per order

### Action Items Based on Results
1. **High match rate (>99%)**: Ready for UAT
2. **Medium match rate (95-99%)**: Review specific differences
3. **Low match rate (<95%)**: Investigation needed

---

## 🤝 Support & Contact

For issues or questions:
- Review error messages and logs
- Check file formats match expectations
- Ensure all dependencies are installed
- Contact your Data Engineering team

---

## 📝 Version History

- **v1.0** (2025-09-02): Initial release
  - Line item validation with composite keys
  - Order-level statistics
  - Multiple partition support
  - Configurable column ignore
  - Comprehensive reporting

---

*This guide is part of the FI Feeds Migration Project*