import os
import json

def convert_coco_to_custom(image_dir, coco_annotation_file, output_annotation_dir):
    # Ensure the output directory exists
    os.makedirs(output_annotation_dir, exist_ok=True)

    # Load the COCO annotation file
    with open(coco_annotation_file, "r") as f:
        coco_data = json.load(f)

    # Create a mapping from image IDs to image metadata
    image_id_to_metadata = {img["id"]: img for img in coco_data["images"]}

    # Create a mapping from category IDs to category names
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Group annotations by image ID
    annotations_by_image = {}
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Process each image and create custom annotation files
    for image_id, annotations in annotations_by_image.items():
        # Get the image metadata
        image_metadata = image_id_to_metadata[image_id]
        image_basename = os.path.splitext(image_metadata["file_name"])[0]

        # Prepare the custom annotation structure
        custom_annotation = {
            "name": image_basename,
            "frames": [
                {
                    "timestamp": 0,  # Placeholder timestamp
                    "objects": []
                }
            ],
            "attributes": {
                "weather": "clear",  # Placeholder attribute
                "scene": "city street",   # Placeholder attribute
                "timeofday": "daytime"  # Placeholder attribute
            }
        }

        # Add objects to the custom annotation
        for annotation in annotations:
            category_name = category_id_to_name[annotation["category_id"]]
            bbox = annotation["bbox"]  # COCO format: [x, y, width, height]
            custom_object = {
                "category": category_name,
                "id": annotation["id"],
                "attributes": {
                    "occluded": False,  # Placeholder attribute
                    "truncated": False,  # Placeholder attribute
                    "trafficLightColor": "none"  # Placeholder attribute
                },
                "box2d": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[0] + bbox[2],
                    "y2": bbox[1] + bbox[3]
                }
            }
            custom_annotation["frames"][0]["objects"].append(custom_object)

        # Save the custom annotation to a JSON file
        output_annotation_path = os.path.join(output_annotation_dir, f"{image_basename}.json")
        with open(output_annotation_path, "w") as f:
            json.dump(custom_annotation, f, indent=4)

        print(f"Converted annotations for image: {image_metadata['file_name']}")

    print("Conversion completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert COCO annotations to custom annotation format.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--coco_annotation_file", type=str, required=True, help="Path to the COCO annotation file.")
    parser.add_argument("--output_annotation_dir", type=str, required=True, help="Directory to save the custom annotation files.")
    args = parser.parse_args()

    convert_coco_to_custom(args.image_dir, args.coco_annotation_file, args.output_annotation_dir)