#!/usr/bin/env python3
"""
FI Fact Report Validation Script
Compares Secor output with Iceberg output for data migration validation
Version: 1.0
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

# Define column headers as per FI Fact Report specification
COLUMN_HEADERS = [
    "order_ref_num",
    "partnersiteid",
    "pid",
    "item_id",
    "retail_price",
    "quantity_on_hand",
    "quantity_on_order",
    "demand_units",
    "demand_net",
    "fulfillment_units",
    "fulfillment_net",
    "return_units",
    "return_net",
    "cancel_units",
    "cancel_net"
]

# Numeric columns for decimal comparison
NUMERIC_COLUMNS = [
    "retail_price",
    "quantity_on_hand",
    "quantity_on_order",
    "demand_units",
    "demand_net",
    "fulfillment_units",
    "fulfillment_net",
    "return_units",
    "return_net",
    "cancel_units",
    "cancel_net"
]

# Integer columns (for exact matching or with tolerance)
INTEGER_COLUMNS = [
    "quantity_on_hand",
    "quantity_on_order",
    "demand_units",
    "fulfillment_units",
    "return_units",
    "cancel_units"
]

# Tolerance for numeric comparisons
DECIMAL_TOLERANCE = 0.01
INTEGER_TOLERANCE = 0  # Exact match for integers by default

# Default columns to ignore (can be overridden at runtime)
DEFAULT_IGNORE_COLUMNS = []


class FIFactReportValidator:
    """Validates FI Fact Report data between Secor and Iceberg outputs"""

    def __init__(self, decimal_tolerance: float = DECIMAL_TOLERANCE,
                 integer_tolerance: int = INTEGER_TOLERANCE,
                 ignore_columns: List[str] = None):
        self.decimal_tolerance = decimal_tolerance
        self.integer_tolerance = integer_tolerance
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

                # Clean quotes and whitespace
                for col in df.columns:
                    df[col] = df[col].str.strip('"').str.strip()

                dfs.append(df)
                print(f"📁 Read {len(df)} records from {os.path.basename(file_path)}")

            except Exception as e:
                print(f"❌ Error reading {file_path}: {str(e)}")
                raise

        # Combine all partitions
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"Total Secor records: {len(combined_df)}")

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

                # Clean quotes and whitespace
                for col in df.columns:
                    df[col] = df[col].str.strip('"').str.strip()

                dfs.append(df)
                print(f"📁 Read {len(df)} records from {os.path.basename(file_path)}")

            except Exception as e:
                print(f"❌ Error reading {file_path}: {str(e)}")
                raise

        # Combine all partitions
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        print(f"Total Iceberg records: {len(combined_df)}")

        return combined_df

    def create_composite_key(self, df: pd.DataFrame) -> pd.Series:
        """
        Create a composite key for FI Fact Report records
        Using order_ref_num + item_id as the unique identifier

        Args:
            df: DataFrame with FI Fact Report data

        Returns:
            Series with composite keys
        """
        return df['order_ref_num'].astype(str) + '_' + df['item_id'].astype(str)

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
        Compare two FI Fact Report records field by field

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

            if col in INTEGER_COLUMNS:
                # Integer comparison with tolerance
                is_match, diff = self.compare_numeric_values(
                    secor_val,
                    iceberg_val,
                    self.integer_tolerance
                )

                if not is_match:
                    differences[col] = {
                        'secor_value': str(secor_val),
                        'iceberg_value': str(iceberg_val),
                        'difference': diff,
                        'type': 'integer'
                    }
            elif col in NUMERIC_COLUMNS:
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
        print("STARTING FI FACT REPORT VALIDATION")
        print("=" * 60)

        # Create composite keys for comparison
        print("\n📊 Creating composite keys (order_ref_num + item_id)...")
        secor_df['composite_key'] = self.create_composite_key(secor_df)
        iceberg_df['composite_key'] = self.create_composite_key(iceberg_df)

        # Set index for efficient lookup
        secor_indexed = secor_df.set_index('composite_key')
        iceberg_indexed = iceberg_df.set_index('composite_key')

        # Find missing records
        secor_keys = set(secor_indexed.index)
        iceberg_keys = set(iceberg_indexed.index)

        missing_in_iceberg = secor_keys - iceberg_keys
        missing_in_secor = iceberg_keys - secor_keys
        common_keys = secor_keys & iceberg_keys

        print(f"\n📈 Record Count Summary:")
        print(f"   Secor records: {len(secor_df)}")
        print(f"   Iceberg records: {len(iceberg_df)}")
        print(f"   Common records: {len(common_keys)}")
        print(f"   Missing in Iceberg: {len(missing_in_iceberg)}")
        print(f"   Missing in Secor: {len(missing_in_secor)}")

        # Store missing records with details
        self.validation_results['missing_in_iceberg'] = []
        for key in missing_in_iceberg:
            row = secor_indexed.loc[key]
            self.validation_results['missing_in_iceberg'].append({
                'composite_key': key,
                'order_ref_num': row.get('order_ref_num', ''),
                'item_id': row.get('item_id', ''),
                'pid': row.get('pid', '')
            })

        self.validation_results['missing_in_secor'] = []
        for key in missing_in_secor:
            row = iceberg_indexed.loc[key]
            self.validation_results['missing_in_secor'].append({
                'composite_key': key,
                'order_ref_num': row.get('order_ref_num', ''),
                'item_id': row.get('item_id', ''),
                'pid': row.get('pid', '')
            })

        # Compare common records
        print(f"\n🔍 Comparing {len(common_keys)} common records...")

        # Show ignored columns if any
        if self.ignore_columns:
            print(f"   ⚠️ Ignoring columns: {', '.join(self.ignore_columns)}")

        records_with_differences = []
        column_mismatch_counts = {col: 0 for col in COLUMN_HEADERS if col not in self.ignore_columns}
        numeric_variance_summary = {col: [] for col in NUMERIC_COLUMNS if col not in self.ignore_columns}

        for composite_key in common_keys:
            secor_row = secor_indexed.loc[composite_key]
            iceberg_row = iceberg_indexed.loc[composite_key]

            differences = self.compare_records(secor_row, iceberg_row)

            if differences:
                records_with_differences.append({
                    'composite_key': composite_key,
                    'order_ref_num': secor_row.get('order_ref_num', ''),
                    'item_id': secor_row.get('item_id', ''),
                    'differences': differences
                })

                # Count column mismatches and track numeric variances
                for col, diff in differences.items():
                    column_mismatch_counts[col] += 1
                    if col in NUMERIC_COLUMNS and diff['type'] in ['numeric', 'integer']:
                        if diff['difference'] != float('inf'):
                            numeric_variance_summary[col].append(diff['difference'])

        self.validation_results['differences'] = records_with_differences
        self.validation_results['column_mismatches'] = {
            col: count for col, count in column_mismatch_counts.items() if count > 0
        }

        # Calculate numeric variance statistics
        self.validation_results['numeric_variances'] = {}
        for col, variances in numeric_variance_summary.items():
            if variances:
                self.validation_results['numeric_variances'][col] = {
                    'count': len(variances),
                    'mean': np.mean(variances),
                    'median': np.median(variances),
                    'max': np.max(variances),
                    'min': np.min(variances),
                    'std': np.std(variances)
                }

        # Calculate summary statistics
        self.validation_results['summary'] = {
            'total_secor_records': len(secor_df),
            'total_iceberg_records': len(iceberg_df),
            'common_records': len(common_keys),
            'records_with_differences': len(records_with_differences),
            'missing_in_iceberg': len(missing_in_iceberg),
            'missing_in_secor': len(missing_in_secor),
            'match_percentage': round(
                (len(common_keys) - len(records_with_differences)) / len(common_keys) * 100, 2
            ) if common_keys else 0
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
        report.append("FI FACT REPORT VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Show configuration
        report.append("\n⚙️ VALIDATION CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Decimal Tolerance: {self.decimal_tolerance}")
        report.append(f"Integer Tolerance: {self.integer_tolerance}")
        if self.ignore_columns:
            report.append(f"Ignored Columns: {', '.join(self.ignore_columns)}")
        else:
            report.append("Ignored Columns: None")

        # Summary section
        summary = self.validation_results['summary']
        report.append("\n📊 VALIDATION SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Secor Records:     {summary['total_secor_records']:,}")
        report.append(f"Total Iceberg Records:   {summary['total_iceberg_records']:,}")
        report.append(f"Common Records:          {summary['common_records']:,}")
        report.append(f"Perfect Matches:         {summary['common_records'] - summary['records_with_differences']:,}")
        report.append(f"Records with Differences: {summary['records_with_differences']:,}")
        report.append(f"Match Rate:              {summary['match_percentage']:.2f}%")

        # Missing records section
        if self.validation_results['missing_in_iceberg']:
            count = len(self.validation_results['missing_in_iceberg'])
            report.append(f"\n❌ RECORDS MISSING IN ICEBERG ({count})")
            report.append("-" * 40)
            for i, record in enumerate(self.validation_results['missing_in_iceberg'][:10], 1):
                report.append(f"  {i}. Order: {record['order_ref_num']}, Item: {record['item_id']}, PID: {record['pid']}")
            if count > 10:
                report.append(f"  ... and {count - 10} more")

        if self.validation_results['missing_in_secor']:
            count = len(self.validation_results['missing_in_secor'])
            report.append(f"\n❌ RECORDS MISSING IN SECOR ({count})")
            report.append("-" * 40)
            for i, record in enumerate(self.validation_results['missing_in_secor'][:10], 1):
                report.append(f"  {i}. Order: {record['order_ref_num']}, Item: {record['item_id']}, PID: {record['pid']}")
            if count > 10:
                report.append(f"  ... and {count - 10} more")

        # Column mismatch summary
        if self.validation_results['column_mismatches']:
            report.append("\n📉 COLUMN MISMATCH SUMMARY")
            report.append("-" * 40)
            for col, count in sorted(
                    self.validation_results['column_mismatches'].items(),
                    key=lambda x: x[1],
                    reverse=True
            ):
                percentage = (count / summary['common_records'] * 100) if summary['common_records'] > 0 else 0
                report.append(f"  {col:25s}: {count:6d} mismatches ({percentage:.2f}%)")

        # Numeric variance statistics
        if self.validation_results['numeric_variances']:
            report.append("\n📐 NUMERIC VARIANCE STATISTICS")
            report.append("-" * 40)
            for col, stats in self.validation_results['numeric_variances'].items():
                report.append(f"\n  {col}:")
                report.append(f"    Count:  {stats['count']}")
                report.append(f"    Mean:   {stats['mean']:.4f}")
                report.append(f"    Median: {stats['median']:.4f}")
                report.append(f"    Min:    {stats['min']:.4f}")
                report.append(f"    Max:    {stats['max']:.4f}")
                report.append(f"    StdDev: {stats['std']:.4f}")

        # Sample differences
        if self.validation_results['differences']:
            report.append("\n🔍 SAMPLE RECORD DIFFERENCES (First 5)")
            report.append("-" * 40)

            for record_data in self.validation_results['differences'][:5]:
                report.append(f"\nOrder: {record_data['order_ref_num']}, Item: {record_data['item_id']}")
                for col, diff in record_data['differences'].items():
                    if diff['type'] in ['numeric', 'integer']:
                        report.append(f"  {col}:")
                        report.append(f"    Secor:   {diff['secor_value']}")
                        report.append(f"    Iceberg: {diff['iceberg_value']}")
                        if diff.get('difference') != float('inf'):
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
        for record_data in self.validation_results['differences']:
            order_ref_num = record_data['order_ref_num']
            item_id = record_data['item_id']
            for col, diff in record_data['differences'].items():
                row = {
                    'order_ref_num': order_ref_num,
                    'item_id': item_id,
                    'Column': col,
                    'Secor_Value': diff['secor_value'],
                    'Iceberg_Value': diff['iceberg_value'],
                    'Type': diff['type']
                }
                if diff['type'] in ['numeric', 'integer']:
                    row['Numeric_Difference'] = diff.get('difference', '')
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"✅ Differences exported to: {output_file}")

    def export_missing_records(self, output_dir: str):
        """
        Export lists of missing records to separate CSV files

        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Export missing in Iceberg
        if self.validation_results['missing_in_iceberg']:
            file_path = os.path.join(output_dir, 'records_missing_in_iceberg.csv')
            df = pd.DataFrame(self.validation_results['missing_in_iceberg'])
            df.to_csv(file_path, index=False)
            print(f"✅ Missing in Iceberg list saved to: {file_path}")

        # Export missing in Secor
        if self.validation_results['missing_in_secor']:
            file_path = os.path.join(output_dir, 'records_missing_in_secor.csv')
            df = pd.DataFrame(self.validation_results['missing_in_secor'])
            df.to_csv(file_path, index=False)
            print(f"✅ Missing in Secor list saved to: {file_path}")

    def export_variance_analysis(self, output_file: str):
        """
        Export detailed numeric variance analysis to CSV

        Args:
            output_file: Path to output CSV file
        """
        if not self.validation_results['numeric_variances']:
            print("No numeric variances to export")
            return

        rows = []
        for col, stats in self.validation_results['numeric_variances'].items():
            rows.append({
                'Column': col,
                'Count': stats['count'],
                'Mean_Difference': stats['mean'],
                'Median_Difference': stats['median'],
                'Min_Difference': stats['min'],
                'Max_Difference': stats['max'],
                'StdDev_Difference': stats['std']
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"✅ Variance analysis exported to: {output_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Validate FI Fact Report data between Secor and Iceberg outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python fi_fact_validation.py --secor-files secor_*.csv --iceberg-files iceberg_*.csv

  # With custom tolerances
  python fi_fact_validation.py --secor-files secor.csv --iceberg-files iceberg.csv --tolerance 0.001 --int-tolerance 1

  # Ignore specific columns
  python fi_fact_validation.py --secor-files secor.csv --iceberg-files iceberg.csv --ignore-columns quantity_on_hand quantity_on_order

  # Export all results
  python fi_fact_validation.py --secor-files secor.csv --iceberg-files iceberg.csv --export-all
        """
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
        '--int-tolerance',
        type=int,
        default=INTEGER_TOLERANCE,
        help=f'Integer tolerance for unit/quantity comparisons (default: {INTEGER_TOLERANCE})'
    )

    parser.add_argument(
        '--ignore-columns',
        nargs='*',
        default=DEFAULT_IGNORE_COLUMNS,
        help='Column names to ignore during comparison (space-separated)'
    )

    parser.add_argument(
        '--ignore-inventory',
        action='store_true',
        help='Shortcut to ignore inventory columns (quantity_on_hand, quantity_on_order)'
    )

    parser.add_argument(
        '--output-dir',
        default='fi_fact_validation_results',
        help='Directory to save validation results (default: fi_fact_validation_results)'
    )

    parser.add_argument(
        '--export-differences',
        action='store_true',
        help='Export detailed differences to CSV'
    )

    parser.add_argument(
        '--export-missing',
        action='store_true',
        help='Export lists of missing records'
    )

    parser.add_argument(
        '--export-variance',
        action='store_true',
        help='Export numeric variance analysis'
    )

    parser.add_argument(
        '--export-all',
        action='store_true',
        help='Export all available outputs (differences, missing, variance)'
    )

    args = parser.parse_args()

    # Handle ignore columns configuration
    ignore_columns = list(args.ignore_columns) if args.ignore_columns else []

    # Add inventory columns to ignore list if flag is set
    if args.ignore_inventory:
        inventory_cols = ['quantity_on_hand', 'quantity_on_order']
        for col in inventory_cols:
            if col not in ignore_columns:
                ignore_columns.append(col)

    # Validate that ignored columns are valid
    invalid_columns = [col for col in ignore_columns if col not in COLUMN_HEADERS]
    if invalid_columns:
        print(f"⚠️ Warning: Invalid column names will be ignored: {', '.join(invalid_columns)}")
        ignore_columns = [col for col in ignore_columns if col in COLUMN_HEADERS]

    # Handle export-all flag
    if args.export_all:
        args.export_differences = True
        args.export_missing = True
        args.export_variance = True

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize validator
    validator = FIFactReportValidator(
        decimal_tolerance=args.tolerance,
        integer_tolerance=args.int_tolerance,
        ignore_columns=ignore_columns
    )

    try:
        # Read data files
        print("\n📂 Loading data files...")
        secor_df = validator.read_secor_files(args.secor_files)
        iceberg_df = validator.read_iceberg_files(args.iceberg_files)

        # Perform validation
        results = validator.validate_datasets(secor_df, iceberg_df)

        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(args.output_dir, f'fi_fact_validation_report_{timestamp}.txt')
        validator.generate_report(report_file)

        # Export additional files if requested
        if args.export_differences:
            diff_file = os.path.join(args.output_dir, f'fi_fact_differences_{timestamp}.csv')
            validator.export_differences_to_csv(diff_file)

        if args.export_missing:
            validator.export_missing_records(args.output_dir)

        if args.export_variance:
            variance_file = os.path.join(args.output_dir, f'fi_fact_variance_analysis_{timestamp}.csv')
            validator.export_variance_analysis(variance_file)

        # Save JSON results for programmatic access
        json_file = os.path.join(args.output_dir, f'fi_fact_validation_results_{timestamp}.json')
        with open(json_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
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
            
            json_results = convert_numpy_types(results)
            json.dump(json_results, f, indent=2)
        print(f"✅ JSON results saved to: {json_file}")

        # Print final status
        print("\n" + "=" * 60)
        if results['summary']['records_with_differences'] > 0:
            print("⚠️ VALIDATION COMPLETED WITH DIFFERENCES FOUND")
            sys.exit(1)  # Validation found differences
        else:
            print("✅ VALIDATION PASSED - NO DIFFERENCES FOUND")
            sys.exit(0)  # Validation passed

    except Exception as e:
        print(f"\n❌ Validation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()