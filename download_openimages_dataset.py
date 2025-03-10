#!/usr/bin/env python3
import os
import openimages

def main():
    dataset_dir = "openimages_apple"
    os.makedirs(dataset_dir, exist_ok=True)
    print("Starting download of Apple images...")

    openimages.download_dataset(
        dest_dir=dataset_dir,       # Where to save images + labels
        labels=["Apple"],           # Labels (list of strings)
        annotation_format="yolo",   # YOLO label format
        limit=1000                  # For example, limit to 1,000 images
    )

    print(f"Download completed. Check '{os.path.abspath(dataset_dir)}'")

if __name__ == "__main__":
    main()
