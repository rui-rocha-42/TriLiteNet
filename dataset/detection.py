import cv2
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk

def main(image_dir, output_dir, append=False):
    os.makedirs(output_dir, exist_ok=True)

    drawing = False
    ix, iy = -1, -1
    boxes = []
    category = "forb_speed_over_50"
    save_flag = False
    quit_flag = False
    next_flag = False
    back_flag = False
    occluded = False
    truncated = False
    img_shape = (0, 0)
    current_index = 0
    total_images = 0
    all_image_files = []

    global_attributes = {
        "weather": "clear",
        "scene": "city street",
        "timeofday": "daytime"
    }

    def launch_gui():
        nonlocal category, global_attributes, save_flag, quit_flag, next_flag, back_flag, occluded, truncated

        def set_category(event=None):
            nonlocal category
            category = category_var.get()

        def set_weather(event=None):
            global_attributes["weather"] = weather_var.get()

        def set_scene(event=None):
            global_attributes["scene"] = scene_var.get()

        def set_timeofday(event=None):
            global_attributes["timeofday"] = time_var.get()

        def toggle_occluded():
            nonlocal occluded
            occluded = not occluded
            occluded_var.set(occluded)

        def toggle_truncated():
            nonlocal truncated
            truncated = not truncated
            truncated_var.set(truncated)

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

        root = tk.Tk()
        root.title("Annotation Controls")

        # Photo progress label
        progress_var = tk.StringVar()
        progress_label = tk.Label(root, textvariable=progress_var)
        progress_label.pack()

        def update_progress():
            progress_var.set(f"Image {current_index+1}/{total_images}")

        root.update_progress = update_progress

        tk.Label(root, text="Object Class:").pack()
        category_var = tk.StringVar(value=category)
        class_options = [
            "forb_speed_over_50",
            "forb_speed_over_80",
            "info_crosswalk",
            "prio_give_way",
            "warn_other_dangers",
            "prio_stop"
        ]
        class_menu = ttk.Combobox(root, textvariable=category_var, values=class_options, state="readonly")
        class_menu.pack()
        class_menu.bind("<<ComboboxSelected>>", set_category)

        tk.Label(root, text="Weather:").pack()
        weather_var = tk.StringVar(value=global_attributes["weather"])
        weather_options = ["rainy", "snowy", "clear", "overcast", "undefined", "partly cloudy", "foggy"]
        weather_menu = ttk.Combobox(root, textvariable=weather_var, values=weather_options, state="readonly")
        weather_menu.pack()
        weather_menu.bind("<<ComboboxSelected>>", set_weather)

        tk.Label(root, text="Scene:").pack()
        scene_var = tk.StringVar(value=global_attributes["scene"])
        scene_options = ["tunnel", "residential", "parking lot", "undefined", "city street", "gas stations", "highway"]
        scene_menu = ttk.Combobox(root, textvariable=scene_var, values=scene_options, state="readonly")
        scene_menu.pack()
        scene_menu.bind("<<ComboboxSelected>>", set_scene)

        tk.Label(root, text="Time of Day:").pack()
        time_var = tk.StringVar(value=global_attributes["timeofday"])
        time_options = ["daytime", "night", "dawn/dusk", "undefined"]
        time_menu = ttk.Combobox(root, textvariable=time_var, values=time_options, state="readonly")
        time_menu.pack()
        time_menu.bind("<<ComboboxSelected>>", set_timeofday)

        occluded_var = tk.BooleanVar(value=occluded)
        truncated_var = tk.BooleanVar(value=truncated)
        occluded_check = tk.Checkbutton(root, text="Occluded", variable=occluded_var, command=toggle_occluded)
        occluded_check.pack()
        truncated_check = tk.Checkbutton(root, text="Truncated", variable=truncated_var, command=toggle_truncated)
        truncated_check.pack()

        tk.Button(root, text="Save (s)", command=save_only).pack(fill='x', pady=2)
        tk.Button(root, text="Next (n)", command=next_image).pack(fill='x', pady=2)
        tk.Button(root, text="Back (b)", command=back_image).pack(fill='x', pady=2)
        tk.Button(root, text="Quit (q)", command=quit_program).pack(fill='x', pady=5)

        tk.Label(root, text="Instructions:").pack()
        tk.Label(root, text="Draw boxes: drag with left mouse\nSave: button or Ctrl+S\nNext: button or 'n'\nBack: button or 'b'\nQuit: button or 'q'").pack()

        root.geometry("300x600")
        return root

    def draw_box(event, x, y, flags, param):
        nonlocal ix, iy, drawing, boxes, category, occluded, truncated, img_shape
        img = param["img"]
        img_disp = param["img_disp"]
        h, w = img_shape
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = max(0, min(x, w-1)), max(0, min(y, h-1))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                x_clamped = max(0, min(x, w-1))
                y_clamped = max(0, min(y, h-1))
                img_disp[:] = img.copy()
                cv2.rectangle(img_disp, (ix, iy), (x_clamped, y_clamped), (0, 255, 0), 2)
                redraw_all(img_disp)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = ix, iy
            x2, y2 = max(0, min(x, w-1)), max(0, min(y, h-1))
            if x1 == x2 or y1 == y2:
                return  # Ignore zero-area boxes
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            box_obj = {
                "category": category,
                "id": len(boxes),
                "attributes": {
                    "occluded": occluded,
                    "truncated": truncated
                },
                "box2d": {
                    "x1": float(min(x1, x2)),
                    "y1": float(min(y1, y2)),
                    "x2": float(max(x1, x2)),
                    "y2": float(max(y1, y2))
                }
            }
            boxes.append(box_obj)
            print(f"Added box: {category} ({x1},{y1},{x2},{y2})")

    def redraw_all(img_disp):
        for box in boxes:
            b = box["box2d"]
            cv2.rectangle(img_disp, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 255, 0), 2)

    def save_bdd_json(image_path, boxes, output_dir):
        name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = int(datetime.now().timestamp() * 1000)
        objects = []
        for box in boxes:
            objects.append(box)
        bdd_json = {
            "name": name,
            "frames": [
                {
                    "timestamp": timestamp,
                    "objects": objects
                }
            ],
            "attributes": global_attributes.copy()
        }
        with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
            json.dump(bdd_json, f, indent=4)
        print(f"Saved {name}.json")

    # List all images
    all_image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]

    # If append, start at the first image without annotation
    current_index = 0
    if append:
        existing_jsons = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.lower().endswith(".json")}
        for idx, img in enumerate(all_image_files):
            img_name = os.path.splitext(os.path.basename(img))[0]
            if img_name not in existing_jsons:
                current_index = idx
                break

    total_images = len(all_image_files)

    gui = launch_gui()

    while 0 <= current_index < total_images:
        img_path = all_image_files[current_index]
        print(f"\nAnnotating {img_path}")
        img = cv2.imread(img_path)
        img_disp = img.copy()
        img_shape = img.shape[:2]
        param = {"img": img, "img_disp": img_disp}
        boxes = []  # Clear boxes for the new image

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", draw_box, param)

        while True:
            cv2.imshow("Annotator", img_disp)
            gui.update()
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):  # Save
                save_bdd_json(img_path, boxes, output_dir)
            elif key == ord('n'):  # Next
                current_index += 1
                break
            elif key == ord('b'):  # Back
                current_index = max(0, current_index - 1)
                break
            elif key == ord('q'):  # Quit
                current_index = -1
                break
            redraw_all(img_disp)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BBox annotation tool")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output JSONs")
    parser.add_argument("--append", action="store_true", help="Skip images that already have a JSON annotation")
    args = parser.parse_args()
    main(args.image_dir, args.output_dir, append=args.append)