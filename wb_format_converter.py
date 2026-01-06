#!/usr/bin/env python3
"""
Removes the ConversionRate column (column 4) and reformats the data.
"""

import csv
import sys
from pathlib import Path


def transform_wb_file(input_file='WB_09-09.txt', output_file='WB_09-09_formatted.txt'):
    """
    Transform WB file to match WorldRugby format:
    - Remove quotes and change separator from "|" to |
    - Remove ConversionRate column (index 3)
    - Add header row matching WorldRugby format
    """
    
    try:
        # Read the input file
        print(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        # Process data
        processed_rows = []
        
        # Add header row (matching WorldRugby format but without ConversionRate)
        header = "OrderNo|LineNumber|ProductID|VariationID|UnitPriceGBP|UnitVatGBP|Quantity"
        processed_rows.append(header)
        
        # Process each data row
        for i, line in enumerate(lines):
            # Remove leading/trailing whitespace
            line = line.strip()
            
            if not line:
                continue
            
            # Split by "|" including the quotes
            # The format is ""|"value1"|"value2"|...
            parts = line.split('"|"')
            
            # Clean up the parts
            cleaned_parts = []
            for part in parts:
                # Remove quotes and extra characters
                cleaned = part.replace('"', '').replace('|', '')
                cleaned_parts.append(cleaned)
            
            # Skip empty first column and ConversionRate column (index 3)
            # Columns: [empty, OrderNo, LineNumber, ConversionRate, ProductID, VariationID, UnitPriceGBP, UnitVatGBP, Quantity]
            # We want: [OrderNo, LineNumber, ProductID, VariationID, UnitPriceGBP, UnitVatGBP, Quantity]
            if len(cleaned_parts) >= 9:
                # Select specific columns (skip index 0 and 3)
                selected_data = [
                    cleaned_parts[1],  # OrderNo
                    cleaned_parts[2],  # LineNumber
                    cleaned_parts[4],  # ProductID (skip ConversionRate at index 3)
                    cleaned_parts[5],  # VariationID
                    cleaned_parts[6],  # UnitPriceGBP
                    cleaned_parts[7],  # UnitVatGBP
                    cleaned_parts[8]   # Quantity
                ]
                
                # Join with pipe separator
                formatted_row = '|'.join(selected_data)
                processed_rows.append(formatted_row)
        
        # Write the output file
        print(f"Writing output file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for row in processed_rows:
                outfile.write(row + '\n')
        
        print(f" Successfully converted {len(processed_rows)-1} rows")
        print(f" Output saved to: {output_file}")
        
        # Show sample of the output
        print("\nSample of converted data (first 5 rows):")
        print("-" * 60)
        for i, row in enumerate(processed_rows[:6]):
            print(row)
        
        return True
        
    except FileNotFoundError:
        print(f" Error: File '{input_file}' not found.")
        print("Please ensure the file is in the same directory as this script.")
        return False
    except Exception as e:
        print(f" Error processing file: {str(e)}")
        return False


def main():
    """Main function to run the transformation."""
    print("=" * 60)
    print("WB File Format Converter")
    print("Converting WB_09-09.txt to WorldRugby format")
    print("=" * 60)
    
    # You can customize the input and output file names here
    input_file = '/Users/agarimella/partner_feeds_validation/data/OD/HNF/iceberg_16-09.csv'
    output_file = '/Users/agarimella/partner_feeds_validation/data/OD/HNF/HNF_FIOrderDetails_20250916_test.txt'
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"\n Input file '{input_file}' not found in current directory.")
        print(f"Current directory: {Path.cwd()}")
        print("\nPlease ensure the WB_09-09.txt file is in the same folder as this script.")
        sys.exit(1)
    
    # Run the transformation
    success = transform_wb_file(input_file, output_file)
    
    if success:
        print("\n Conversion completed successfully!")
        print(f"The formatted file '{output_file}' is ready for use.")
    else:
        print("\n Conversion failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
