import pandas as pd
import cv2
import os
import argparse
from tqdm import tqdm

def resize_and_adjust_boxes(args):
    # config
    target_size = (640, 640)
    interpolation = cv2.INTER_LINEAR

    # Load Annotations
    try:
        print(f"Loading annotations from: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        print(f"Found {len(df)} annotations.")
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {args.csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Prepare Output
    print(f"Preparing output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, "resized_annotations.csv")
    new_annotations = []

    # Create subdirectories for splits
    for split in df['split'].unique():
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    # Process Images
    print(f"Processing and resizing images to {target_size}...")
    skipped_count = 0

    # Group by filename to process each image once, even if it has multiple boxes
    for filename, group in tqdm(df.groupby('filename'), desc="Processing Images"):
        original_image_path = os.path.join(args.image_dir, filename)

        if not os.path.exists(original_image_path):
            print(f"Warning: Image file not found: {original_image_path}. Skipping this image and its annotations.")
            skipped_count += len(group)
            continue

        try:
            # Load original image
            img = cv2.imread(original_image_path)
            if img is None:
                print(f"Warning: Failed to load image: {original_image_path}. Skipping.")
                skipped_count += len(group)
                continue

            original_h, original_w = img.shape[:2]

            # Resize image
            resized_img = cv2.resize(img, target_size, interpolation=interpolation)
            target_w, target_h = target_size

            # Calculate scaling factors
            x_scale = target_w / original_w if original_w > 0 else 0
            y_scale = target_h / original_h if original_h > 0 else 0

            # Determine output path based on the split (use the split from the first row of the group)
            split_type = group['split'].iloc[0]
            relative_output_filename = os.path.join(split_type, filename) # e.g., train/image1.jpg
            output_image_path = os.path.join(args.output_dir, relative_output_filename)

            # Save resized image
            cv2.imwrite(output_image_path, resized_img)

            # Adjust Bounding Boxes for this image
            for _, row in group.iterrows():
                # Original box coordinates
                xmin_orig = row['xmin']
                ymin_orig = row['ymin']
                xmax_orig = row['xmax']
                ymax_orig = row['ymax']

                # Apply scaling
                xmin_new = xmin_orig * x_scale
                ymin_new = ymin_orig * y_scale
                xmax_new = xmax_orig * x_scale
                ymax_new = ymax_orig * y_scale

                # Convert to integer and change coordinates to be within new image bounds
                xmin_new = max(0, int(xmin_new))
                ymin_new = max(0, int(ymin_new))
                xmax_new = min(target_w - 1, int(xmax_new)) # Ensure max is within bounds (0 to width-1)
                ymax_new = min(target_h - 1, int(ymax_new)) # Ensure max is within bounds (0 to height-1)

                # Basic check for invalid boxes after resize
                if xmin_new >= xmax_new or ymin_new >= ymax_new:
                    print(f"Warning: Bounding box for class '{row['class']}' in image '{filename}' became invalid after resize (min>=max). Original: [{xmin_orig},{ymin_orig},{xmax_orig},{ymax_orig}], New (raw scaled): [{xmin_orig*x_scale:.2f},{ymin_orig*y_scale:.2f},{xmax_orig*x_scale:.2f},{ymax_orig*y_scale:.2f}]. Clamped: [{xmin_new},{ymin_new},{xmax_new},{ymax_new}]. Skipping this box.")
                    skipped_count += 1
                    continue # Skip adding this specific box annotation

                # Store new annotation data
                new_annotations.append({
                    'filename': relative_output_filename, # Path relative to output_dir
                    'width': target_w,
                    'height': target_h,
                    'class': row['class'],
                    'xmin': xmin_new,
                    'ymin': ymin_new,
                    'xmax': xmax_new,
                    'ymax': ymax_new,
                    'split': split_type
                })

        except Exception as e:
            print(f"Error processing image {filename}: {e}. Skipping.")
            skipped_count += len(group) # Skip all annotations for this failed image
            continue

    # Save New Annotations CSV
    if not new_annotations:
        print("Warning: No valid annotations were processed. Output CSV will be empty.")
        # create an empty file with headers
        final_df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'split'])
    else:
       final_df = pd.DataFrame(new_annotations)

    final_df.to_csv(output_csv_path, index=False)

    print("-" * 30)
    print(f"Processing Complete!")
    print(f"Resized images saved in: {args.output_dir}")
    print(f"Updated annotations saved to: {output_csv_path}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} annotations due to missing images, loading errors, or invalid boxes after resize.")
    print(f"Total annotations processed and saved: {len(final_df)}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images and adjust bounding box annotations.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    resize_and_adjust_boxes(args)