import os
import sys
from PIL import Image

def resize_images(folder, width, height):
    # Check if the folder exists
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        sys.exit(1)

    # Iterate through all files in the folder recursively
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Resize the image
                    resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
                    # Overwrite the original image
                    resized_img.save(file_path)
                    print(f"Resized and saved: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python resize_images.py <folder> <width> <height>")
        sys.exit(1)

    folder = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])

    resize_images(folder, width, height)