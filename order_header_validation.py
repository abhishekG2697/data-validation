#!/usr/bin/env python3
"""
FI Order Header Validation Script
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

# Define column headers as per FI Feed specification
COLUMN_HEADERS = [
    "OrderID",
    "OrderNo",
    "CustomerID",
    "OrderDateUpdated",
    "ShipDateToSend",
    "TransactionType",
    "SalesChannel",
    "GoodsNetGBP",
    "GoodsVatGBP",
    "ShippingNetGBP",
    "ShippingVatGBP",
    "Currency",
    "ConversionRate",
    "ShippingCountry",
    "Reason",
    "PaymentType",
    "PromoCode"
]

# Numeric columns for decimal comparison
NUMERIC_COLUMNS = [
    "GoodsNetGBP",
    "GoodsVatGBP",
    "ShippingNetGBP",
    "ShippingVatGBP",
    "ConversionRate"
]

# Tolerance for numeric comparisons
DECIMAL_TOLERANCE = 0.01

# Default columns to ignore (can be overridden at runtime)
DEFAULT_IGNORE_COLUMNS = []


class OrderHeaderValidator:
    """Validates Order Header data between Secor and Iceberg outputs"""

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
            'ignored_columns': self.ignore_columns
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


                for col in df.columns:
                    df[col] = df[col].str.strip('"').str.strip()

                dfs.append(df)
                print(f"✓ Read {len(df)} records from {os.path.basename(file_path)}")

            except Exception as e:
                print(f"✗ Error reading {file_path}: {str(e)}")
                raise

        # Combine all partitions
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"Total Secor records: {len(combined_df)}")

        return combined_df

    def read_iceberg_file(self, file_path: str) -> pd.DataFrame:
        """
        Read Iceberg output file

        Args:
            file_path: Path to Iceberg CSV file

        Returns:
            DataFrame with Iceberg records
        """
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


            for col in df.columns:
                df[col] = df[col].str.strip('"').str.strip()

            print(f"✓ Read {len(df)} records from {os.path.basename(file_path)}")
            return df

        except Exception as e:
            print(f"✗ Error reading {file_path}: {str(e)}")
            raise

    def compare_numeric_values(self, val1: Any, val2: Any, tolerance: float) -> Tuple[bool, float]:
        """
        Compare two numeric values with tolerance

        Args:
            val1: First value
            val2: Second value
            tolerance: Acceptable difference

        Returns:
            Tuple of (is_match, difference)
        """
        try:
            # Handle empty/null values
            if pd.isna(val1) and pd.isna(val2):
                return True, 0.0
            if pd.isna(val1) or pd.isna(val2):
                return False, float('inf')

            # Convert to float
            num1 = float(str(val1).replace(',', ''))
            num2 = float(str(val2).replace(',', ''))

            diff = abs(num1 - num2)
            return diff <= tolerance, diff

        except (ValueError, TypeError):
            # If conversion fails, do string comparison
            return str(val1) == str(val2), 0.0

    def compare_records(self, secor_row: pd.Series, iceberg_row: pd.Series) -> Dict[str, Any]:
        """
        Compare two order records field by field

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
                # Numeric comparison with tolerance
                is_match, diff = self.compare_numeric_values(
                    secor_val,
                    iceberg_val,
                    self.decimal_tolerance
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
        Perform validation between datasets

        Args:
            secor_df: Secor DataFrame
            iceberg_df: Iceberg DataFrame

        Returns:
            Validation results dictionary
        """
        print("\n" + "=" * 60)
        print("STARTING VALIDATION")
        print("=" * 60)

        # Use OrderNo as the key for comparison
        key_column = "OrderNo"

        # Set index for efficient lookup
        secor_indexed = secor_df.set_index(key_column)
        iceberg_indexed = iceberg_df.set_index(key_column)

        # Find missing records
        secor_orders = set(secor_indexed.index)
        iceberg_orders = set(iceberg_indexed.index)

        missing_in_iceberg = secor_orders - iceberg_orders
        missing_in_secor = iceberg_orders - secor_orders
        common_orders = secor_orders & iceberg_orders

        print(f"\n Record Count Summary:")
        print(f"   Secor records: {len(secor_df)}")
        print(f"   Iceberg records: {len(iceberg_df)}")
        print(f"   Common orders: {len(common_orders)}")
        print(f"   Missing in Iceberg: {len(missing_in_iceberg)}")
        print(f"   Missing in Secor: {len(missing_in_secor)}")

        # Store missing records
        self.validation_results['missing_in_iceberg'] = list(missing_in_iceberg)
        self.validation_results['missing_in_secor'] = list(missing_in_secor)

        # Compare common records
        print(f"\n Comparing {len(common_orders)} common orders...")

        # Show ignored columns if any
        if self.ignore_columns:
            print(f"     Ignoring columns: {', '.join(self.ignore_columns)}")

        orders_with_differences = []
        column_mismatch_counts = {col: 0 for col in COLUMN_HEADERS if col not in self.ignore_columns}

        for order_no in common_orders:
            secor_row = secor_indexed.loc[order_no]
            iceberg_row = iceberg_indexed.loc[order_no]

            differences = self.compare_records(secor_row, iceberg_row)

            if differences:
                orders_with_differences.append({
                    'order_no': order_no,
                    'differences': differences
                })

                # Count column mismatches
                for col in differences:
                    column_mismatch_counts[col] += 1

        self.validation_results['differences'] = orders_with_differences
        self.validation_results['column_mismatches'] = {
            col: count for col, count in column_mismatch_counts.items() if count > 0
        }

        # Calculate summary statistics
        self.validation_results['summary'] = {
            'total_secor_records': len(secor_df),
            'total_iceberg_records': len(iceberg_df),
            'common_orders': len(common_orders),
            'orders_with_differences': len(orders_with_differences),
            'missing_in_iceberg': len(missing_in_iceberg),
            'missing_in_secor': len(missing_in_secor),
            'match_percentage': round(
                (len(common_orders) - len(orders_with_differences)) / len(common_orders) * 100, 2
            ) if common_orders else 0
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
        report.append("FI ORDER HEADER VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Show configuration
        report.append("\n️  VALIDATION CONFIGURATION")
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
        report.append(f"Total Secor Records:    {summary['total_secor_records']:,}")
        report.append(f"Total Iceberg Records:  {summary['total_iceberg_records']:,}")
        report.append(f"Common Orders:          {summary['common_orders']:,}")
        report.append(f"Perfect Matches:        {summary['common_orders'] - summary['orders_with_differences']:,}")
        report.append(f"Orders with Differences: {summary['orders_with_differences']:,}")
        report.append(f"Match Rate:             {summary['match_percentage']:.2f}%")

        # Missing records section
        if self.validation_results['missing_in_iceberg']:
            report.append(f"\n️  ORDERS MISSING IN ICEBERG ({len(self.validation_results['missing_in_iceberg'])})")
            report.append("-" * 40)
            for i, order_no in enumerate(self.validation_results['missing_in_iceberg'][:10], 1):
                report.append(f"  {i}. {order_no}")
            if len(self.validation_results['missing_in_iceberg']) > 10:
                report.append(f"  ... and {len(self.validation_results['missing_in_iceberg']) - 10} more")

        if self.validation_results['missing_in_secor']:
            report.append(f"\n️  ORDERS MISSING IN SECOR ({len(self.validation_results['missing_in_secor'])})")
            report.append("-" * 40)
            for i, order_no in enumerate(self.validation_results['missing_in_secor'][:10], 1):
                report.append(f"  {i}. {order_no}")
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
                percentage = (count / summary['common_orders'] * 100) if summary['common_orders'] > 0 else 0
                report.append(f"  {col:20s}: {count:5d} mismatches ({percentage:.2f}%)")

        # Sample differences
        if self.validation_results['differences']:
            report.append("\n SAMPLE ORDER DIFFERENCES (First 5)")
            report.append("-" * 40)

            for order_data in self.validation_results['differences'][:5]:
                report.append(f"\nOrder: {order_data['order_no']}")
                for col, diff in order_data['differences'].items():
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
            print(f"\n Report saved to: {output_file}")

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
        for order_data in self.validation_results['differences']:
            order_no = order_data['order_no']
            for col, diff in order_data['differences'].items():
                row = {
                    'OrderNo': order_no,
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
        print(f" Differences exported to: {output_file}")

    def export_missing_orders(self, output_dir: str):
        """
        Export lists of missing orders to separate files

        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Export missing in Iceberg
        if self.validation_results['missing_in_iceberg']:
            file_path = os.path.join(output_dir, 'orders_missing_in_iceberg.txt')
            with open(file_path, 'w') as f:
                for order in self.validation_results['missing_in_iceberg']:
                    f.write(f"{order}\n")
            print(f" Missing in Iceberg list saved to: {file_path}")

        # Export missing in Secor
        if self.validation_results['missing_in_secor']:
            file_path = os.path.join(output_dir, 'orders_missing_in_secor.txt')
            with open(file_path, 'w') as f:
                for order in self.validation_results['missing_in_secor']:
                    f.write(f"{order}\n")
            print(f" Missing in Secor list saved to: {file_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Validate FI Order Header data between Secor and Iceberg outputs'
    )

    parser.add_argument(
        '--secor-files',
        nargs='+',
        required=True,
        help='Path(s) to Secor CSV file(s). Multiple files will be combined.'
    )

    parser.add_argument(
        '--iceberg-file',
        required=True,
        help='Path to Iceberg CSV file'
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
        help='Column names to ignore during comparison (space-separated). Example: --ignore-columns Reason PromoCode'
    )

    parser.add_argument(
        '--ignore-reason',
        action='store_true',
        help='Shortcut to ignore the Reason column (equivalent to --ignore-columns Reason)'
    )

    parser.add_argument(
        '--output-dir',
        default='validation_results',
        help='Directory to save validation results (default: validation_results)'
    )

    parser.add_argument(
        '--export-differences',
        action='store_true',
        help='Export detailed differences to CSV'
    )

    parser.add_argument(
        '--export-missing',
        action='store_true',
        help='Export lists of missing orders'
    )

    args = parser.parse_args()

    # Handle ignore columns configuration
    ignore_columns = list(args.ignore_columns) if args.ignore_columns else []

    # Add Reason to ignore list if flag is set
    if args.ignore_reason and 'Reason' not in ignore_columns:
        ignore_columns.append('Reason')

    # Validate that ignored columns are valid
    invalid_columns = [col for col in ignore_columns if col not in COLUMN_HEADERS]
    if invalid_columns:
        print(f"  Warning: Invalid column names will be ignored: {', '.join(invalid_columns)}")
        ignore_columns = [col for col in ignore_columns if col in COLUMN_HEADERS]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize validator with ignore columns
    validator = OrderHeaderValidator(
        decimal_tolerance=args.tolerance,
        ignore_columns=ignore_columns
    )

    try:
        # Read data files
        print("\n Loading data files...")
        secor_df = validator.read_secor_files(args.secor_files)
        iceberg_df = validator.read_iceberg_file(args.iceberg_file)

        # Perform validation
        results = validator.validate_datasets(secor_df, iceberg_df)

        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(args.output_dir, f'validation_report_{timestamp}.txt')
        validator.generate_report(report_file)

        # Export additional files if requested
        if args.export_differences:
            diff_file = os.path.join(args.output_dir, f'differences_{timestamp}.csv')
            validator.export_differences_to_csv(diff_file)

        if args.export_missing:
            validator.export_missing_orders(args.output_dir)

        # Save JSON results for programmatic access
        json_file = os.path.join(args.output_dir, f'validation_results_{timestamp}.json')
        with open(json_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            json_results = results.copy()
            json_results['missing_in_iceberg'] = list(results['missing_in_iceberg'])
            json_results['missing_in_secor'] = list(results['missing_in_secor'])
            json.dump(json_results, f, indent=2)
        print(f" JSON results saved to: {json_file}")

        # Exit with appropriate code
        if results['summary']['orders_with_differences'] > 0:
            sys.exit(1)  # Validation found differences
        else:
            sys.exit(0)  # Validation passed

    except Exception as e:
        print(f"\n Validation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()