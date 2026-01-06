import pandas as pd
import numpy as np


def round_decimal_values(input_file, output_file, decimal_places=0):
    """
    Round decimal values in the order details file

    Parameters:
    input_file: Path to input CSV file
    output_file: Path to output CSV file
    decimal_places: Number of decimal places to round to (default: 0)
    """

    # Read the pipe-delimited file
    df = pd.read_csv(input_file, sep='|')

    print(f"Loaded {len(df)} rows from {input_file}")
    print(f"Columns: {df.columns.tolist()}")

    # Identify numeric columns that need rounding
    # Based on your file, these are UnitPrice and UnitVat
    numeric_columns = ['UnitPrice', 'UnitVat']

    # Round the numeric columns
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric type first (in case there are any string values)
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Round to specified decimal places
            if decimal_places == 0:
                # Round to integer
                df[col] = df[col].round(0).astype('Int64')
            else:
                # Round to specified decimal places
                df[col] = df[col].round(decimal_places)

            print(f"✓ Rounded {col} to {decimal_places} decimal places")

    # Save the rounded data
    df.to_csv(output_file, sep='|', index=False)

    print(f"\n✓ Successfully saved rounded data to {output_file}")

    # Display sample of the changes
    print("\nSample of rounded data (first 10 rows):")
    print(df[['OrderNo', 'ProductID', 'UnitPrice', 'UnitVat', 'Quantity']].head(10))

    # Show statistics
    print("\nStatistics:")
    print(f"Total rows processed: {len(df)}")
    for col in numeric_columns:
        if col in df.columns:
            print(f"{col} - Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")

    return df


def compare_before_after(input_file, output_file):
    """
    Compare the original and rounded values to show the differences
    """
    df_original = pd.read_csv(input_file, sep='|')
    df_rounded = pd.read_csv(output_file, sep='|')

    # Compare UnitPrice and UnitVat
    print("\n=== Comparison Report ===")

    for col in ['UnitPrice', 'UnitVat']:
        if col in df_original.columns:
            df_original[col] = pd.to_numeric(df_original[col], errors='coerce')
            df_rounded[col] = pd.to_numeric(df_rounded[col], errors='coerce')

            # Calculate differences
            diff = (df_original[col] - df_rounded[col]).abs()

            print(f"\n{col}:")
            print(f"  Max difference from rounding: {diff.max():.2f}")
            print(f"  Average difference: {diff.mean():.4f}")
            print(f"  Number of values changed: {(diff > 0).sum()}")

            # Show some examples where rounding made a difference
            changed_rows = df_original[diff > 0.4].head(5)
            if len(changed_rows) > 0:
                print(f"  Examples of significant rounding (original → rounded):")
                for idx, row in changed_rows.iterrows():
                    original_val = df_original.loc[idx, col]
                    rounded_val = df_rounded.loc[idx, col]
                    print(f"    Row {idx}: {original_val:.2f} → {rounded_val}")


# Main execution
if __name__ == "__main__":
    # File paths
    input_file = "/Users/agarimella/partner_feeds_validation/data/OD/WorldRugby/iceberg_OD_10-09.csv"  # Your input file
    output_file = "/Users/agarimella/partner_feeds_validation/data/OD/WorldRugby/iceberg_OD_10-09_rounded.csv"  # Output file with rounded values

    # Round to whole numbers (0 decimal places)
    # Change to 2 if you want to keep 2 decimal places
    decimal_places = 0

    # Process the file
    df_rounded = round_decimal_values(input_file, output_file, decimal_places)

    # Optional: Compare before and after
    print("\n" + "=" * 50)
    compare_before_after(input_file, output_file)

    print("\n=== Processing Complete ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Decimal places: {decimal_places}")