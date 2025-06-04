import os
import cv2
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # For displaying images in Tkinter

def load_annotations(annotation_path):
    """
    Load annotations from a JSON file.
    """
    try:
        with open(annotation_path, "r") as f:
            data = json.load(f)
        # Access the first frame and its objects
        return data["frames"][0]["objects"]
    except Exception as e:
        print(f"Error loading annotations from {annotation_path}: {e}")
        return []

def draw_bounding_boxes(image, annotations):
    """
    Draw bounding boxes on the image based on the annotations.
    """
    for obj in annotations:
        # Access the "box2d" key for bounding box coordinates
        box = obj.get("box2d")
        if not box:
            continue
        category = obj.get("category", "unknown")
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put the category name below the bounding box
        cv2.putText(image, category, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def view_annotations(image_dir, annotation_dir):
    """
    Main function to view images with bounding boxes.
    """
    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
    total_images = len(image_files)
    current_index = 0

    def update_image():
        """
        Update the displayed image and progress label.
        """
        nonlocal current_index
        img_path = os.path.join(image_dir, image_files[current_index])
        print(f"Loading image: {img_path}")
        annotation_path = os.path.join(annotation_dir, os.path.splitext(image_files[current_index])[0] + ".json")

        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            progress_var.set(f"Image {current_index + 1}/{total_images} (Image not found)")
            return

        # Load the annotations
        if os.path.exists(annotation_path):
            annotations = load_annotations(annotation_path)
            img = draw_bounding_boxes(img, annotations)

        # Convert the image to a format suitable for Tkinter
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(img)  # Convert to PIL Image
        img = img.resize((canvas.winfo_width(), canvas.winfo_height()))  # Resize to fit canvas
        img_tk = ImageTk.PhotoImage(img)

        # Update the canvas with the new image
        canvas.image = img_tk  # Keep a reference to avoid garbage collection
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        # Update the progress label
        progress_var.set(f"Image {current_index + 1}/{total_images}")

    def next_image():
        """
        Show the next image.
        """
        nonlocal current_index
        if current_index < total_images - 1:
            current_index += 1
            update_image()

    def previous_image():
        """
        Show the previous image.
        """
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            update_image()

    def quit_program():
        """
        Quit the program.
        """
        root.destroy()

    # Create a simple GUI for navigation
    root = tk.Tk()
    root.title("Image Viewer")

    # Set larger initial window size and allow resizing
    root.geometry("1200x800")  # Set the initial size of the window
    root.resizable(True, True)  # Allow resizing of the window

    # Create a canvas for displaying the image
    canvas = tk.Canvas(root, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)  # Make the canvas fill the window

    # Progress label
    progress_var = tk.StringVar()
    progress_label = tk.Label(root, textvariable=progress_var)
    progress_label.pack()

    # Navigation buttons
    tk.Button(root, text="Previous (b)", command=previous_image).pack(fill="x", pady=2)
    tk.Button(root, text="Next (n)", command=next_image).pack(fill="x", pady=2)
    tk.Button(root, text="Quit (q)", command=quit_program).pack(fill="x", pady=5)

    # Bind keyboard shortcuts
    root.bind("<b>", lambda event: previous_image())
    root.bind("<n>", lambda event: next_image())
    root.bind("<q>", lambda event: quit_program())

    # Show the first image
    root.update_idletasks()  # Ensure the window is fully initialized
    update_image()

    # Start the GUI loop
    root.mainloop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="View images with bounding boxes from annotations.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory with JSON annotations")
    args = parser.parse_args()

    view_annotations(args.image_dir, args.annotation_dir)