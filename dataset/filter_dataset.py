import os
import json
import shutil

def filter_dataset_by_classes(dataset_dir, output_dir, class_names):
    # Define input and output paths
    input_annotation_dirs = {
        "train": os.path.join(dataset_dir, "det_annotations/train"),
        "val": os.path.join(dataset_dir, "det_annotations/val")
    }
    input_image_dirs = {
        "train": os.path.join(dataset_dir, "images/train"),
        "val": os.path.join(dataset_dir, "images/val")
    }
    output_annotation_dirs = {
        "train": os.path.join(output_dir, "det_annotations/train"),
        "val": os.path.join(output_dir, "det_annotations/val")
    }
    output_image_dirs = {
        "train": os.path.join(output_dir, "images/train"),
        "val": os.path.join(output_dir, "images/val")
    }

    # Ensure output directories exist
    for path in output_annotation_dirs.values():
        os.makedirs(path, exist_ok=True)
    for path in output_image_dirs.values():
        os.makedirs(path, exist_ok=True)

    # Process each split (train and val)
    for split in ["train", "val"]:
        annotation_dir = input_annotation_dirs[split]
        image_dir = input_image_dirs[split]
        output_annotation_dir = output_annotation_dirs[split]
        output_image_dir = output_image_dirs[split]

        # List all annotation files
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith(".json")]

        for annotation_file in annotation_files:
            annotation_path = os.path.join(annotation_dir, annotation_file)

            # Load the annotation JSON
            with open(annotation_path, "r") as f:
                annotation_data = json.load(f)

            # Check if any object in the annotation matches the given class names
            has_matching_class = any(
                obj["category"] in class_names
                for frame in annotation_data.get("frames", [])
                for obj in frame.get("objects", [])
            )

            if has_matching_class:
                # Copy the corresponding image to the output directory
                image_name = annotation_data["name"] + ".jpg"  # Assuming images are in .jpg format
                image_path = os.path.join(image_dir, image_name)
                if os.path.exists(image_path):
                    shutil.copy(image_path, os.path.join(output_image_dir, image_name))

                # Copy the annotation file to the output directory
                output_annotation_path = os.path.join(output_annotation_dir, annotation_file)
                with open(output_annotation_path, "w") as f:
                    json.dump(annotation_data, f, indent=4)

                print(f"Copied {image_name} and its annotation to the output directory.")

    print("Filtering completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter dataset by class names and copy to output directory.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory of the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the filtered dataset.")
    parser.add_argument("--class_names", nargs="+", required=True, help="List of class names to filter by.")
    args = parser.parse_args()

    filter_dataset_by_classes(args.dataset_dir, args.output_dir, args.class_names)