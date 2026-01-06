#!/usr/bin/env python3
"""
FI Order Details Validation Script
Compares Secor output with Iceberg output for data migration validation
Version: 1.1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os
from datetime import datetime
import json
import argparse
from pathlib import Path

# Define column headers as per FI Feed specification for Order Details
COLUMN_HEADERS = [
    "OrderID",
    "OrderNo",
    "LineNumber",
    "ConversionRate",
    "ProductID",
    "VariationID",
    "UnitPriceGBP",
    "UnitVatGBP",
    "Quantity"
]

# Numeric columns for decimal comparison
NUMERIC_COLUMNS = [
    "ConversionRate",
    "UnitPriceGBP",
    "UnitVatGBP",
    "Quantity"
]

# Integer columns (no decimal tolerance needed)
INTEGER_COLUMNS = [
    "LineNumber",
    "Quantity"  # Quantity is typically integer
]

# Tolerance for numeric comparisons
DECIMAL_TOLERANCE = 0.01

# Default columns to ignore (can be overridden at runtime)
DEFAULT_IGNORE_COLUMNS = []


class OrderDetailsValidator:
    """Validates Order Details data between Secor and Iceberg outputs"""

    def __init__(self, decimal_tolerance: float = DECIMAL_TOLERANCE,
                 ignore_columns: List[str] = None):
        self.decimal_tolerance = decimal_tolerance
        self.ignore_columns = ignore_columns if ignore_columns else []
        self.validation_results = {
            'summary': {},
            'differences': [],
            'missing_in_iceberg': [],
            'missing_in_secor': [],
            'column_mismatches': {},
            'numeric_variances': [],
            'ignored_columns': self.ignore_columns,
            'line_item_stats': {}
        }

    def read_secor_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Read and combine multiple Secor partition files

        Args:
            file_paths: List of paths to Secor CSV files

        Returns:
            Combined DataFrame with all Secor records
        """
        dfs = []

        for file_path in file_paths:
            try:
                # Read with pipe delimiter and no header
                df = pd.read_csv(
                    file_path,
                    sep='|',
                    header=None,
                    names=COLUMN_HEADERS,
                    dtype=str,  # Read all as string initially
                    quoting=3,  # QUOTE_NONE
                    na_values=[''],
                    keep_default_na=False
                )

                # Clean up quotes if present
                for col in df.columns:
                    df[col] = df[col].str.strip('"').str.strip()

                dfs.append(df)
                print(f" Read {len(df)} line items from {os.path.basename(file_path)}")

            except Exception as e:
                print(f" Error reading {file_path}: {str(e)}")
                raise

        # Combine all partitions
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"Total Secor line items: {len(combined_df)}")

        # Show unique order count
        if 'OrderNo' in combined_df.columns:
            unique_orders = combined_df['OrderNo'].nunique()
            print(f"Unique orders in Secor: {unique_orders}")

        return combined_df

    def read_iceberg_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Read and combine multiple Iceberg partition files

        Args:
            file_paths: List of paths to Iceberg CSV files

        Returns:
            Combined DataFrame with all Iceberg records
        """
        dfs = []

        for file_path in file_paths:
            try:
                # Read with pipe delimiter and no header
                df = pd.read_csv(
                    file_path,
                    sep='|',
                    header=None,
                    names=COLUMN_HEADERS,
                    dtype=str,  # Read all as string initially
                    quoting=3,  # QUOTE_NONE
                    na_values=[''],
                    keep_default_na=False
                )

                # Clean up quotes if present
                for col in df.columns:
                    df[col] = df[col].str.strip('"').str.strip()

                dfs.append(df)
                print(f"✓ Read {len(df)} line items from {os.path.basename(file_path)}")

            except Exception as e:
                print(f"✗ Error reading {file_path}: {str(e)}")
                raise

        # Combine all partitions
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"Total Iceberg line items: {len(combined_df)}")

        # Show unique order count
        if 'OrderNo' in combined_df.columns:
            unique_orders = combined_df['OrderNo'].nunique()
            print(f"Unique orders in Iceberg: {unique_orders}")

        return combined_df

    def create_composite_key(self, df: pd.DataFrame) -> pd.Series:
        """
        Create a composite key for line items (OrderNo + LineNumber)

        Args:
            df: DataFrame with order details

        Returns:
            Series with composite keys
        """
        return df['OrderNo'].astype(str) + '_' + df['LineNumber'].astype(str)

    def compare_numeric_values(self, val1: Any, val2: Any, tolerance: float, is_integer: bool = False) -> Tuple[
        bool, float]:
        """
        Compare two numeric values with tolerance

        Args:
            val1: First value
            val2: Second value
            tolerance: Acceptable difference
            is_integer: Whether the values should be integers

        Returns:
            Tuple of (is_match, difference)
        """
        try:
            # Handle empty/null values
            if pd.isna(val1) and pd.isna(val2):
                return True, 0.0
            if pd.isna(val1) or pd.isna(val2):
                return False, float('inf')

            # Convert to numeric
            num1 = float(str(val1).replace(',', ''))
            num2 = float(str(val2).replace(',', ''))

            # For integer columns, check if they're whole numbers
            if is_integer:
                if num1.is_integer() and num2.is_integer():
                    return int(num1) == int(num2), abs(num1 - num2)

            diff = abs(num1 - num2)
            return diff <= tolerance, diff

        except (ValueError, TypeError):
            # If conversion fails, do string comparison
            return str(val1) == str(val2), 0.0

    def compare_records(self, secor_row: pd.Series, iceberg_row: pd.Series) -> Dict[str, Any]:
        """
        Compare two order detail records field by field

        Args:
            secor_row: Secor record
            iceberg_row: Iceberg record

        Returns:
            Dictionary with comparison results
        """
        differences = {}

        for col in COLUMN_HEADERS:
            # Skip ignored columns
            if col in self.ignore_columns:
                continue

            secor_val = secor_row.get(col, '')
            iceberg_val = iceberg_row.get(col, '')

            # Handle None/NaN values
            if pd.isna(secor_val):
                secor_val = ''
            if pd.isna(iceberg_val):
                iceberg_val = ''

            if col in NUMERIC_COLUMNS:
                # Check if it's an integer column
                is_integer = col in INTEGER_COLUMNS

                # Numeric comparison with tolerance
                is_match, diff = self.compare_numeric_values(
                    secor_val,
                    iceberg_val,
                    self.decimal_tolerance,
                    is_integer
                )

                if not is_match:
                    differences[col] = {
                        'secor_value': str(secor_val),
                        'iceberg_value': str(iceberg_val),
                        'difference': diff,
                        'type': 'numeric'
                    }
            else:
                # String comparison
                if str(secor_val).strip() != str(iceberg_val).strip():
                    differences[col] = {
                        'secor_value': str(secor_val),
                        'iceberg_value': str(iceberg_val),
                        'type': 'string'
                    }

        return differences

    def validate_datasets(self, secor_df: pd.DataFrame, iceberg_df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive validation between datasets

        Args:
            secor_df: Secor DataFrame
            iceberg_df: Iceberg DataFrame

        Returns:
            Validation results dictionary
        """
        print("\n" + "=" * 60)
        print("STARTING ORDER DETAILS VALIDATION")
        print("=" * 60)

        # Create composite keys for line items
        secor_df['composite_key'] = self.create_composite_key(secor_df)
        iceberg_df['composite_key'] = self.create_composite_key(iceberg_df)

        # Use composite key for comparison
        key_column = "composite_key"

        # Set index for efficient lookup
        secor_indexed = secor_df.set_index(key_column)
        iceberg_indexed = iceberg_df.set_index(key_column)

        # Find missing line items
        secor_items = set(secor_indexed.index)
        iceberg_items = set(iceberg_indexed.index)

        missing_in_iceberg = secor_items - iceberg_items
        missing_in_secor = iceberg_items - secor_items
        common_items = secor_items & iceberg_items

        print(f"\n Line Item Count Summary:")
        print(f"   Secor line items: {len(secor_df)}")
        print(f"   Iceberg line items: {len(iceberg_df)}")
        print(f"   Common line items: {len(common_items)}")
        print(f"   Missing in Iceberg: {len(missing_in_iceberg)}")
        print(f"   Missing in Secor: {len(missing_in_secor)}")

        # Calculate order-level statistics
        secor_orders = set(secor_df['OrderNo'].unique())
        iceberg_orders = set(iceberg_df['OrderNo'].unique())

        print(f"\n Order-Level Summary:")
        print(f"   Unique orders in Secor: {len(secor_orders)}")
        print(f"   Unique orders in Iceberg: {len(iceberg_orders)}")
        print(f"   Orders only in Secor: {len(secor_orders - iceberg_orders)}")
        print(f"   Orders only in Iceberg: {len(iceberg_orders - secor_orders)}")

        # Store missing records
        self.validation_results['missing_in_iceberg'] = list(missing_in_iceberg)
        self.validation_results['missing_in_secor'] = list(missing_in_secor)

        # Store line item statistics
        self.validation_results['line_item_stats'] = {
            'avg_items_per_order_secor': float(len(secor_df) / len(secor_orders)) if secor_orders else 0,
            'avg_items_per_order_iceberg': float(len(iceberg_df) / len(iceberg_orders)) if iceberg_orders else 0,
            'max_items_per_order_secor': int(secor_df.groupby('OrderNo').size().max()) if len(secor_df) > 0 else 0,
            'max_items_per_order_iceberg': int(iceberg_df.groupby('OrderNo').size().max()) if len(iceberg_df) > 0 else 0
        }

        # Compare common line items
        print(f"\n Comparing {len(common_items)} common line items...")

        # Show ignored columns if any
        if self.ignore_columns:
            print(f"   Ignoring columns: {', '.join(self.ignore_columns)}")

        items_with_differences = []
        column_mismatch_counts = {col: 0 for col in COLUMN_HEADERS if col not in self.ignore_columns}

        for item_key in common_items:
            secor_row = secor_indexed.loc[item_key]
            iceberg_row = iceberg_indexed.loc[item_key]

            differences = self.compare_records(secor_row, iceberg_row)

            if differences:
                # Extract OrderNo and LineNumber from composite key
                order_no, line_no = item_key.rsplit('_', 1)

                items_with_differences.append({
                    'composite_key': item_key,
                    'order_no': order_no,
                    'line_number': line_no,
                    'differences': differences
                })

                # Count column mismatches
                for col in differences:
                    column_mismatch_counts[col] += 1

        self.validation_results['differences'] = items_with_differences
        self.validation_results['column_mismatches'] = {
            col: count for col, count in column_mismatch_counts.items() if count > 0
        }

        # Calculate summary statistics
        self.validation_results['summary'] = {
            'total_secor_line_items': len(secor_df),
            'total_iceberg_line_items': len(iceberg_df),
            'common_line_items': len(common_items),
            'items_with_differences': len(items_with_differences),
            'missing_in_iceberg': len(missing_in_iceberg),
            'missing_in_secor': len(missing_in_secor),
            'unique_orders_secor': len(secor_orders),
            'unique_orders_iceberg': len(iceberg_orders),
            'match_percentage': round(
                (len(common_items) - len(items_with_differences)) / len(common_items) * 100, 2
            ) if common_items else 0
        }

        return self.validation_results

    def generate_report(self, output_file: str = None):
        """
        Generate detailed validation report

        Args:
            output_file: Optional path to save report
        """
        report = []
        report.append("\n" + "=" * 80)
        report.append("FI ORDER DETAILS VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Show configuration
        report.append("\n  VALIDATION CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Decimal Tolerance: {self.decimal_tolerance}")
        if self.ignore_columns:
            report.append(f"Ignored Columns: {', '.join(self.ignore_columns)}")
        else:
            report.append("Ignored Columns: None")

        # Summary section
        summary = self.validation_results['summary']
        report.append("\n VALIDATION SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Secor Line Items:    {summary['total_secor_line_items']:,}")
        report.append(f"Total Iceberg Line Items:  {summary['total_iceberg_line_items']:,}")
        report.append(f"Common Line Items:         {summary['common_line_items']:,}")
        report.append(
            f"Perfect Matches:           {summary['common_line_items'] - summary['items_with_differences']:,}")
        report.append(f"Items with Differences:    {summary['items_with_differences']:,}")
        report.append(f"Match Rate:                {summary['match_percentage']:.2f}%")

        # Order-level summary
        report.append("\n ORDER-LEVEL STATISTICS")
        report.append("-" * 40)
        report.append(f"Unique Orders (Secor):     {summary['unique_orders_secor']:,}")
        report.append(f"Unique Orders (Iceberg):   {summary['unique_orders_iceberg']:,}")

        stats = self.validation_results.get('line_item_stats', {})
        if stats:
            report.append(f"Avg Items/Order (Secor):   {stats.get('avg_items_per_order_secor', 0):.2f}")
            report.append(f"Avg Items/Order (Iceberg): {stats.get('avg_items_per_order_iceberg', 0):.2f}")
            report.append(f"Max Items/Order (Secor):   {stats.get('max_items_per_order_secor', 0)}")
            report.append(f"Max Items/Order (Iceberg): {stats.get('max_items_per_order_iceberg', 0)}")

        # Missing line items section
        if self.validation_results['missing_in_iceberg']:
            report.append(f"\n  LINE ITEMS MISSING IN ICEBERG ({len(self.validation_results['missing_in_iceberg'])})")
            report.append("-" * 40)
            for i, item_key in enumerate(self.validation_results['missing_in_iceberg'][:10], 1):
                report.append(f"  {i}. {item_key}")
            if len(self.validation_results['missing_in_iceberg']) > 10:
                report.append(f"  ... and {len(self.validation_results['missing_in_iceberg']) - 10} more")

        if self.validation_results['missing_in_secor']:
            report.append(f"\n  LINE ITEMS MISSING IN SECOR ({len(self.validation_results['missing_in_secor'])})")
            report.append("-" * 40)
            for i, item_key in enumerate(self.validation_results['missing_in_secor'][:10], 1):
                report.append(f"  {i}. {item_key}")
            if len(self.validation_results['missing_in_secor']) > 10:
                report.append(f"  ... and {len(self.validation_results['missing_in_secor']) - 10} more")

        # Column mismatch summary
        if self.validation_results['column_mismatches']:
            report.append("\n COLUMN MISMATCH SUMMARY")
            report.append("-" * 40)
            for col, count in sorted(
                    self.validation_results['column_mismatches'].items(),
                    key=lambda x: x[1],
                    reverse=True
            ):
                percentage = (count / summary['common_line_items'] * 100) if summary['common_line_items'] > 0 else 0
                report.append(f"  {col:20s}: {count:5d} mismatches ({percentage:.2f}%)")

        # Sample differences
        if self.validation_results['differences']:
            report.append("\n SAMPLE LINE ITEM DIFFERENCES (First 5)")
            report.append("-" * 40)

            for item_data in self.validation_results['differences'][:5]:
                report.append(f"\nOrder: {item_data['order_no']}, Line: {item_data['line_number']}")
                for col, diff in item_data['differences'].items():
                    if diff['type'] == 'numeric':
                        report.append(f"  {col}:")
                        report.append(f"    Secor:   {diff['secor_value']}")
                        report.append(f"    Iceberg: {diff['iceberg_value']}")
                        report.append(f"    Diff:    {diff.get('difference', 'N/A'):.4f}")
                    else:
                        report.append(f"  {col}:")
                        report.append(f"    Secor:   '{diff['secor_value']}'")
                        report.append(f"    Iceberg: '{diff['iceberg_value']}'")

        # Print report
        report_text = '\n'.join(report)
        print(report_text)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\n✅ Report saved to: {output_file}")

        return report_text

    def export_differences_to_csv(self, output_file: str):
        """
        Export detailed differences to CSV for further analysis

        Args:
            output_file: Path to output CSV file
        """
        if not self.validation_results['differences']:
            print("No differences to export")
            return

        # Flatten differences for CSV export
        rows = []
        for item_data in self.validation_results['differences']:
            order_no = item_data['order_no']
            line_no = item_data['line_number']

            for col, diff in item_data['differences'].items():
                row = {
                    'OrderNo': order_no,
                    'LineNumber': line_no,
                    'Column': col,
                    'Secor_Value': diff['secor_value'],
                    'Iceberg_Value': diff['iceberg_value'],
                    'Type': diff['type']
                }
                if diff['type'] == 'numeric':
                    row['Numeric_Difference'] = diff.get('difference', '')
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"✅ Differences exported to: {output_file}")

    def export_missing_items(self, output_dir: str):
        """
        Export lists of missing line items to separate files

        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Export missing in Iceberg
        if self.validation_results['missing_in_iceberg']:
            file_path = os.path.join(output_dir, 'line_items_missing_in_iceberg.txt')
            with open(file_path, 'w') as f:
                for item in self.validation_results['missing_in_iceberg']:
                    f.write(f"{item}\n")
            print(f"✅ Missing in Iceberg list saved to: {file_path}")

        # Export missing in Secor
        if self.validation_results['missing_in_secor']:
            file_path = os.path.join(output_dir, 'line_items_missing_in_secor.txt')
            with open(file_path, 'w') as f:
                for item in self.validation_results['missing_in_secor']:
                    f.write(f"{item}\n")
            print(f"✅ Missing in Secor list saved to: {file_path}")

    def export_order_level_summary(self, output_file: str):
        """
        Export order-level summary showing line item counts per order

        Args:
            output_file: Path to output CSV file
        """
        # Get the dataframes from the last validation
        # This would need to be stored during validation for this to work
        print(f"Order-level summary export would be saved to: {output_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Validate FI Order Details data between Secor and Iceberg outputs'
    )

    parser.add_argument(
        '--secor-files',
        nargs='+',
        required=True,
        help='Path(s) to Secor CSV file(s). Multiple files will be combined.'
    )

    parser.add_argument(
        '--iceberg-files',
        nargs='+',
        required=True,
        help='Path(s) to Iceberg CSV file(s). Multiple files will be combined.'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=DECIMAL_TOLERANCE,
        help=f'Decimal tolerance for numeric comparisons (default: {DECIMAL_TOLERANCE})'
    )

    parser.add_argument(
        '--ignore-columns',
        nargs='*',
        default=DEFAULT_IGNORE_COLUMNS,
        help='Column names to ignore during comparison (space-separated). Example: --ignore-columns VariationID'
    )

    parser.add_argument(
        '--output-dir',
        default='validation_results_details',
        help='Directory to save validation results (default: validation_results_details)'
    )

    parser.add_argument(
        '--export-differences',
        action='store_true',
        help='Export detailed differences to CSV'
    )

    parser.add_argument(
        '--export-missing',
        action='store_true',
        help='Export lists of missing line items'
    )

    parser.add_argument(
        '--export-order-summary',
        action='store_true',
        help='Export order-level summary statistics'
    )

    args = parser.parse_args()

    # Handle ignore columns configuration
    ignore_columns = list(args.ignore_columns) if args.ignore_columns else []

    # Validate that ignored columns are valid
    invalid_columns = [col for col in ignore_columns if col not in COLUMN_HEADERS]
    if invalid_columns:
        print(f"  Warning: Invalid column names will be ignored: {', '.join(invalid_columns)}")
        ignore_columns = [col for col in ignore_columns if col in COLUMN_HEADERS]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize validator with ignore columns
    validator = OrderDetailsValidator(
        decimal_tolerance=args.tolerance,
        ignore_columns=ignore_columns
    )

    try:
        # Read data files
        print("\n Loading Order Details data files...")
        secor_df = validator.read_secor_files(args.secor_files)
        iceberg_df = validator.read_iceberg_files(args.iceberg_files)

        # Perform validation
        results = validator.validate_datasets(secor_df, iceberg_df)

        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(args.output_dir, f'details_validation_report_{timestamp}.txt')
        validator.generate_report(report_file)

        # Export additional files if requested
        if args.export_differences:
            diff_file = os.path.join(args.output_dir, f'details_differences_{timestamp}.csv')
            validator.export_differences_to_csv(diff_file)

        if args.export_missing:
            validator.export_missing_items(args.output_dir)

        if args.export_order_summary:
            summary_file = os.path.join(args.output_dir, f'order_summary_{timestamp}.csv')
            validator.export_order_level_summary(summary_file)

        # Save JSON results for programmatic access
        json_file = os.path.join(args.output_dir, f'details_validation_results_{timestamp}.json')
        with open(json_file, 'w') as f:
            # Convert sets to lists and numpy types to native Python types for JSON serialization
            json_results = results.copy()
            json_results['missing_in_iceberg'] = list(results['missing_in_iceberg'])
            json_results['missing_in_secor'] = list(results['missing_in_secor'])

            # Convert numpy int64/float64 to native Python types
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            json_results = convert_numpy_types(json_results)
            json.dump(json_results, f, indent=2)
        print(f"✅ JSON results saved to: {json_file}")

        # Exit with appropriate code
        if results['summary']['items_with_differences'] > 0:
            sys.exit(1)  # Validation found differences
        else:
            sys.exit(0)  # Validation passed

    except Exception as e:
        print(f"\n❌ Validation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
