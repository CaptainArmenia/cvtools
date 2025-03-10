import random
from collections import defaultdict
from argparse import ArgumentParser
import os

from ultralytics import YOLO
import cv2
import numpy as np

# For reproducible random colors (remove seed if you want truly random every run)
random.seed(42)

# Dictionary to store a unique color for each class ID
class_colors = defaultdict(lambda: get_random_color())

def get_random_color():
    """Returns a random BGR color tuple."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_transparent_box(
    frame, 
    top_left, 
    bottom_right, 
    color, 
    alpha=0.4, 
    thickness=2, 
    corner_length=0
):
    """
    Draw a semi-transparent filled rectangle with highlighted corners.
    """
    overlay = frame.copy()

    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw the rectangle border and stylized corners
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)

    # Optional corner highlights
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)

def put_label_with_background(
    frame, 
    text, 
    org, 
    font_scale=0.6, 
    color=(255, 255, 255), 
    bg_color=(0, 0, 0), 
    alpha=0.6
):
    """
    Draw text with a semi-transparent background rectangle behind it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x, y = org
    x2, y2 = x + text_w + 10, y + text_h + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, text, (x + 5, y + text_h + 3), font, font_scale, color, 1, cv2.LINE_AA)



def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Model path")
    parser.add_argument("--video", type=str, required=True, help="Video path")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to track")
    args = parser.parse_args()


    # Load the YOLO model
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video {args.video}")
        return

    # Get video properties for writing output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize max dimension to 1024
    inference_max_dim = 1920
    inference_scale = min(inference_max_dim / width, inference_max_dim / height)
    inference_width, inference_height = int(width * inference_scale), int(height * inference_scale)

    vis_max_dim = 1200
    vis_scale = min(vis_max_dim / width, vis_max_dim / height)
    vis_width, vis_height = int(width * vis_scale), int(height * vis_scale)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback if FPS is invalid/unavailable
        fps = 30

    # Create a VideoWriter to save MP4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_file = args.output if args.output else f"output_{os.path.basename(args.video)}"

    out = cv2.VideoWriter(output_file, fourcc, fps, (vis_width, vis_height))

    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1

        if frame_index % 2 == 1:
            continue
       
        vis_frame = cv2.resize(frame, (0, 0), fx=vis_scale, fy=vis_scale)
        inference_frame = cv2.resize(frame, (0, 0), fx=inference_scale, fy=inference_scale)

        results = model.predict(inference_frame, classes=args.classes, imgsz=inference_width)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xywh.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x, y, w, h = box
                top_left = (int(x - w / 2), int(y - h / 2))
                bottom_right = (int(x + w / 2), int(y + h / 2))

                # Convert from inference frame coordinates to vis frame coordinates
                top_left = (int(top_left[0] / inference_scale * vis_scale), int(top_left[1] / inference_scale * vis_scale))
                bottom_right = (int(bottom_right[0] / inference_scale * vis_scale), int(bottom_right[1] / inference_scale * vis_scale))

                # Draw the box
                #color = get_random_color()
                color = class_colors[int(results[0].boxes.cls[i].cpu())]
                draw_transparent_box(vis_frame, top_left, bottom_right, color)

                # Draw the label
                label = f"{results[0].names[int(results[0].boxes.cls[i].cpu())]} {results[0].boxes.conf[i]:.2f}"
                put_label_with_background(vis_frame, label, top_left)


        # Show the annotated frame
        cv2.imshow("Object detection", vis_frame)
        out.write(vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
