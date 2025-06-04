import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk

def main(image_dir, output_dir, append=False):
    os.makedirs(output_dir, exist_ok=True)

    current_polygon = []
    all_polygons = []
    save_flag = False
    quit_flag = False
    next_flag = False
    back_flag = False
    finish_polygon_flag = False
    current_index = 0
    total_images = 0
    all_image_files = []

    def launch_gui():
        nonlocal save_flag, quit_flag, next_flag, back_flag, finish_polygon_flag

        def save_only():
            nonlocal save_flag
            save_flag = True

        def next_image():
            nonlocal next_flag
            next_flag = True

        def back_image():
            nonlocal back_flag
            back_flag = True

        def quit_program():
            nonlocal quit_flag
            quit_flag = True

        def finish_polygon():
            nonlocal finish_polygon_flag
            finish_polygon_flag = True

        root = tk.Tk()
        root.title("Drivable Area Annotation Tool")

        # Set larger initial window size
        root.geometry("300x350")

        tk.Button(root, text="Save (s)", command=save_only).pack(fill='x', pady=2)
        tk.Button(root, text="Next (n)", command=next_image).pack(fill='x', pady=2)
        tk.Button(root, text="Back (b)", command=back_image).pack(fill='x', pady=2)
        tk.Button(root, text="Finish Polygon (f)", command=finish_polygon).pack(fill='x', pady=2)
        tk.Button(root, text="Quit (q)", command=quit_program).pack(fill='x', pady=5)

        tk.Label(root, text="Instructions:").pack()
        tk.Label(root, text="Draw polygons: click to add points\nFinish polygon: button or 'f'\nSave: button or 's'\nNext: button or 'n'\nBack: button or 'b'\nQuit: button or 'q'").pack()

        return root

    def draw_polygon(event, x, y, flags, param):
        """
        Handle mouse events for drawing polygons.
        """
        nonlocal current_polygon, all_polygons
        img_disp = param["img_disp"]

        if event == cv2.EVENT_LBUTTONDOWN:  # Add a point to the current polygon
            current_polygon.append((x, y))
            redraw_all(img_disp)

    def redraw_all(img_disp):
        """
        Redraw all polygons and the current polygon on the image.
        """
        img_disp[:] = img.copy()
        if mask_overlay is not None:
            # Draw the existing mask on the image
            mask_indices = np.where(mask_overlay == 255)
            img_disp[mask_indices] = (0, 255, 0)  # Green color for drivable areas

        # Fill existing polygons
        for polygon in all_polygons:
            cv2.fillPoly(img_disp, [np.array(polygon, dtype=np.int32)], (0, 255, 0))  # Green fill

        # Draw the outline of the current polygon (auto-closing)
        if len(current_polygon) > 1:
            # Add the first point to close the polygon visually
            closed_polygon = current_polygon + [current_polygon[0]]
            cv2.polylines(img_disp, [np.array(closed_polygon, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow outline

    def draw_polygon_on_image(img, polygon):
        """
        Draw a polygon on the image.
        """
        if len(polygon) < 2:
            return
        for i in range(len(polygon) - 1):
            cv2.line(img, polygon[i], polygon[i + 1], (0, 255, 0), 2)
        if len(polygon) > 2:  # Close the polygon
            cv2.line(img, polygon[-1], polygon[0], (0, 255, 0), 2)

    def save_mask(image_path, all_polygons, output_dir, existing_mask=None):
        """
        Save the drivable area polygons as a binary mask.
        Combine new polygons with the existing mask if it exists.
        """
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw new polygons on the mask
        for polygon in all_polygons:
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)

        # Include the current (unfinished) polygon
        if len(current_polygon) > 2:
            cv2.fillPoly(mask, [np.array(current_polygon, dtype=np.int32)], 1)

        # Combine with the existing mask if it exists
        if existing_mask is not None:
            mask = cv2.bitwise_or(mask, existing_mask)

        mask_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")
        cv2.imwrite(mask_path, mask * 255)  # Save as binary mask (0 and 255)
        print(f"Saved mask: {mask_path}")

    def load_existing_mask(image_path, output_dir):
        """
        Load an existing mask if it exists and return it.
        """
        mask_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return mask
        return None

    def mask_to_polygons(mask):
        """
        Convert an existing mask to polygons using contours.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [contour.squeeze().tolist() for contour in contours if contour.shape[0] > 2]
        return polygons

    # List all images
    all_image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
    total_images = len(all_image_files)

    # Handle append mode
    if append:
        for i, image_path in enumerate(all_image_files):
            mask_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")
            if not os.path.exists(mask_path):
                current_index = i
                break
        else:
            print("All images already have masks. Starting from the first image.")
            current_index = 0

    # Launch the GUI
    root = launch_gui()

    # Main loop for annotation
    while 0 <= current_index < total_images:
        if quit_flag:
            break
        img_path = all_image_files[current_index]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            current_index += 1
            continue

        img_disp = img.copy()
        existing_mask = load_existing_mask(img_path, output_dir)
        mask_overlay = existing_mask.copy() if existing_mask is not None else None
        current_polygon = []
        all_polygons = mask_to_polygons(existing_mask) if existing_mask is not None else []

        param = {"img_disp": img_disp}
        window_title = f"Drivable Area Annotator - Image {current_index + 1}/{total_images}"
        cv2.namedWindow(window_title)
        cv2.moveWindow(window_title, 100, 100)  # Set the window position (x=100, y=100)
        cv2.setMouseCallback(window_title, draw_polygon, param)

        # Ensure the mask is drawn when the image is first loaded
        redraw_all(img_disp)

        while True:
            cv2.imshow(window_title, img_disp)
            root.update()
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):  # Save shortcut
                save_flag = True
            elif key == ord('n'):
                next_flag = True
            elif key == ord('b'):
                back_flag = True
            elif key == ord('q'):
                quit_flag = True
            elif key == ord('f'):  # Finish the current polygon
                finish_polygon_flag = True

            if finish_polygon_flag:
                if current_polygon:
                    all_polygons.append(current_polygon)
                    current_polygon = []
                    redraw_all(img_disp)
                finish_polygon_flag = False

            if next_flag or back_flag or quit_flag:
                break
            if save_flag:
                save_mask(img_path, all_polygons, output_dir, existing_mask)
                save_flag = False  # Allow multiple saves

        cv2.destroyAllWindows()

        if next_flag:
            current_index += 1
            next_flag = False  # Reset the flag
        elif back_flag:
            current_index = max(0, current_index - 1)
            back_flag = False  # Reset the flag

    root.destroy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drivable area annotation tool")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output masks")
    parser.add_argument("--append", action="store_true", help="Start from the latest image without a mask")
    args = parser.parse_args()
    main(args.image_dir, args.output_dir, append=args.append)