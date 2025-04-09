import pandas as pd
import argparse
import os

def fix_filenames_in_csv(args):
    input_csv_path = args.input_csv
    output_csv_path = args.output_csv
    filename_col = 'filename'
    extension = '.jpg'

    # Validate Input
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at '{input_csv_path}'")
        return

    print(f"Loading annotations from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if filename_col not in df.columns:
        print(f"Error: Column '{filename_col}' not found in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return

    print(f"Checking and adding '{extension}' to '{filename_col}' column where needed")
    updated_count = 0
    skipped_count = 0

    # Function to apply to each filename
    def add_ext(fname):
        nonlocal updated_count, skipped_count
        # Convert to string just in case there are non-string types
        fname_str = str(fname)
        # Check if it already ends with .jpg (case-insensitive)
        if fname_str.lower().endswith(extension.lower()):
            skipped_count += 1
            return fname_str # Return as is
        else:
            updated_count += 1
            return fname_str + extension # Append extension

    # Apply the function to the filename column
    df[filename_col] = df[filename_col].apply(add_ext)

    print(f"Saving updated annotations to: {output_csv_path}")
    try:
        # Ensure output directory exists if it's specified in the path
        output_dir = os.path.dirname(output_csv_path)
        if output_dir: # If path includes a directory
             os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_csv_path, index=False)
        print("-" * 30)
        print("Processing Complete!")
        print(f"Updated CSV saved to: {output_csv_path}")
        print(f"Filenames modified (extension added): {updated_count}")
        print(f"Filenames skipped (already had extension): {skipped_count}")
        print("-" * 30)

    except Exception as e:
        print(f"Error writing updated CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add '.jpg' extension to filenames in a CSV.")
    parser.add_argument("--input_csv", required=True,
                        help="Path to the original input CSV file (e.g., labels_with_split.csv).")
    parser.add_argument("--output_csv", required=True,
                        help="Path where the fixed output CSV file will be saved (e.g., labels_with_split_fixed.csv).")

    args = parser.parse_args()
    fix_filenames_in_csv(args)