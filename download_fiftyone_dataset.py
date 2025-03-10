#!/usr/bin/env python
"""
Download Open Images V7 for a specified class (train + val),
combine them into one FiftyOne dataset, and then export
in YOLO format for Ultralytics.
"""

import argparse
import fiftyone as fo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class_name",
        type=str,
        default="Apple",
        help="Name of the class to download (e.g. 'Apple', 'Person', etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="open_images_yolo_dataset",
        help="Directory for the YOLO dataset output"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples for each split (optional)"
    )
    args = parser.parse_args()

    class_name = args.class_name
    output_dir = args.output_dir
    max_samples = args.max_samples

    print(f"=== Downloading train split for '{class_name}' ===")
    train_ds = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=[class_name],
        dataset_name="temp_train_dataset",
        max_samples=max_samples,
    )
    print(f"Train dataset size: {len(train_ds)} samples\n")

    print(f"=== Downloading validation split for '{class_name}' ===")
    val_ds = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=[class_name],
        dataset_name="temp_val_dataset",
        max_samples=max_samples,
    )
    print(f"Validation dataset size: {len(val_ds)} samples\n")

    #
    # Create a new combined dataset
    #
    combined_name = f"open-images-{class_name.lower()}-combined"
    print(f"=== Creating combined dataset '{combined_name}' ===")
    if combined_name in fo.list_datasets():
        # If you run this script repeatedly, you might have an existing dataset
        print(f"Dataset '{combined_name}' already exists; deleting it first...")
        fo.delete_dataset(combined_name)

    combined_dataset = fo.Dataset(name=combined_name)

    # Add the train samples to the combined dataset
    combined_dataset.add_collection(train_ds)
    # Add the validation samples to the combined dataset
    combined_dataset.add_collection(val_ds)

    # Assign "train" or "val" tag so we know which is which
    for sample in train_ds:
        sample.tags.append("train")
        sample.save()

    for sample in val_ds:
        sample.tags.append("val")
        sample.save()

    print(
        f"Combined dataset '{combined_name}' has {len(combined_dataset)} total samples\n"
    )

    #
    # We now create two views on the combined dataset:
    #   - train_view
    #   - val_view
    #
    train_view = combined_dataset.match_tags("train")
    val_view = combined_dataset.match_tags("val")

    #
    # Export each view to YOLO format, specifying the *same* class list
    #
    # This will create a folder structure:
    #   open_images_yolo_dataset/
    #       ├── train/
    #       │   ├── images/
    #       │   └── labels/
    #       └── val/
    #           ├── images/
    #           └── labels/
    #
    print(f"=== Exporting train set to '{output_dir}' in YOLO format ===")
    train_view.export(
        export_dir=output_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train",
        classes=[class_name],  # ensure consistent classes
    )

    print(f"\n=== Exporting val set to '{output_dir}' in YOLO format ===")
    val_view.export(
        export_dir=output_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="val",
        classes=[class_name],  # ensure consistent classes
    )

    print(f"\nExport complete! YOLO dataset directory: {output_dir}\n")

    #
    # Optional: Launch FiftyOne App to inspect
    #
    # session = fo.launch_app(combined_dataset)
    # session.wait()


if __name__ == "__main__":
    main()
