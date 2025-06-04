import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk

def main(image_dir, output_dir, lane_width=10, append=False):
    os.makedirs(output_dir, exist_ok=True)

    drawing = False
    current_path = []
    all_paths = []
    save_flag = False
    quit_flag = False
    next_flag = False
    back_flag = False
    finish_path_flag = False
    current_index = 0
    total_images = 0
    all_image_files = []

    def launch_gui():
        nonlocal save_flag, quit_flag, next_flag, back_flag, finish_path_flag

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

        def finish_path():
            nonlocal finish_path_flag
            finish_path_flag = True

        root = tk.Tk()
        root.title("Lane Annotation Tool")

        # Set larger initial window size
        root.geometry("300x350")

        tk.Button(root, text="Save (s)", command=save_only).pack(fill='x', pady=2)
        tk.Button(root, text="Next (n)", command=next_image).pack(fill='x', pady=2)
        tk.Button(root, text="Back (b)", command=back_image).pack(fill='x', pady=2)
        tk.Button(root, text="Finish Path (f)", command=finish_path).pack(fill='x', pady=2)
        tk.Button(root, text="Quit (q)", command=quit_program).pack(fill='x', pady=5)

        tk.Label(root, text="Instructions:").pack()
        tk.Label(root, text="Draw lanes: click to add points\nFinish path: button or 'f'\nSave: button or 's'\nNext: button or 'n'\nBack: button or 'b'\nQuit: button or 'q'").pack()

        return root

    def draw_lane(event, x, y, flags, param):
        """
        Handle mouse events for drawing lane paths.
        """
        nonlocal drawing, current_path, all_paths
        img_disp = param["img_disp"]

        if event == cv2.EVENT_LBUTTONDOWN:  # Add a point to the current path
            current_path.append((x, y))
            redraw_all(img_disp)
        elif event == cv2.EVENT_RBUTTONDOWN:  # Finish the current path
            if current_path:
                all_paths.append(current_path)
                current_path = []
                redraw_all(img_disp)

    def redraw_all(img_disp):
        """
        Redraw all paths and the current path on the image.
        """
        img_disp[:] = img.copy()
        if mask_overlay is not None:
            # Draw the existing mask on the image
            mask_indices = np.where(mask_overlay == 255)
            img_disp[mask_indices] = (0, 255, 0)  # Green color for lanes
        for path in all_paths:
            draw_path_as_polygon(img_disp, path)
        if current_path:
            draw_path_as_polygon(img_disp, current_path)

    def draw_path_as_polygon(img, path):
        """
        Draw a path as a polygon with the specified lane width.
        """
        if len(path) < 2:
            return
        for i in range(len(path) - 1):
            cv2.line(img, path[i], path[i + 1], (0, 255, 0), lane_width)

    def save_mask(image_path, all_paths, output_dir, existing_mask=None):
        """
        Save the lane paths as a binary mask.
        Combine new lanes with the existing mask if it exists.
        """
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw new paths on the mask
        for path in all_paths:
            for i in range(len(path) - 1):
                cv2.line(mask, path[i], path[i + 1], 1, lane_width)

        # Include the current (unfinished) path in the mask
        if len(current_path) > 1:
            for i in range(len(current_path) - 1):
                cv2.line(mask, current_path[i], current_path[i + 1], 1, lane_width)

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
        current_path = []
        all_paths = []

        param = {"img_disp": img_disp}
        window_title = f"Lane Annotator - Image {current_index + 1}/{total_images}"
        cv2.namedWindow(window_title)
        cv2.moveWindow(window_title, 100, 100)  # Set the window position (x=100, y=100)
        cv2.setMouseCallback(window_title, draw_lane, param)

        # Ensure the mask is drawn when the image is first loaded
        redraw_all(img_disp)

        while True:
            cv2.imshow(window_title, img_disp)
            root.update()
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):  # Save shortcut changed to 's'
                save_flag = True
            elif key == ord('n'):
                next_flag = True
            elif key == ord('b'):
                back_flag = True
            elif key == ord('q'):
                quit_flag = True
            elif key == ord('f'):  # Finish the current path
                finish_path_flag = True

            if finish_path_flag:
                if current_path:
                    all_paths.append(current_path)
                    current_path = []
                    redraw_all(img_disp)
                finish_path_flag = False

            if next_flag or back_flag or quit_flag:
                break
            if save_flag:
                save_mask(img_path, all_paths, output_dir, existing_mask)
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
    parser = argparse.ArgumentParser(description="Lane annotation tool")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output masks")
    parser.add_argument("--lane_width", type=int, default=10, help="Width of the lane polygons")
    parser.add_argument("--append", action="store_true", help="Start from the latest image without a mask")
    args = parser.parse_args()
    main(args.image_dir, args.output_dir, lane_width=args.lane_width, append=args.append)