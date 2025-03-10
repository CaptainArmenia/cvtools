import random
import argparse
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

# Ultralytics and boxmot imports
from ultralytics import YOLO
from boxmot import BotSort  # pip install boxmot
import torch

# For reproducible random colors (remove seed if you want truly random every run)
random.seed(42)

# Dictionary to store a unique color for each track ID (for trails)
track_colors = {}

def get_random_color():
    """Returns a random BGR color tuple."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_transparent_box(frame, top_left, bottom_right, color, thickness=2, corner_length=20):
    """
    Draw a bounding box with only the edges (completely transparent interior)
    and optional stylized corners.
    """
    # Draw the rectangle edges only (no filled color)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    
    # Draw stylized corners
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)

def draw_filled_transparent_box(frame, top_left, bottom_right, color, alpha=0.3, thickness=2):
    """
    Draw a filled bounding box with a semi-transparent color and a solid border.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Draw a solid border over the filled box
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)

def put_label_with_background(frame, text, org, font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.6):
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

def draw_fading_polyline(frame, points, color):
    """
    Draw a continuous polyline that fades from tail to head.
    
    'points' is a list of dicts: 
       [{ 'coord': (x, y), 'fade': int }, ... ]
    'color' is the base track color (B, G, R).
    """
    if len(points) < 2:
        return

    overlay = frame.copy()

    for i in range(1, len(points)):
        pt1 = points[i - 1]
        pt2 = points[i]

        fade_avg = (pt1['fade'] + pt2['fade']) / 2.0
        alpha = fade_avg / 255.0
        alpha = max(0.0, min(1.0, alpha))  # clamp

        thickness = 1 + int(10 * alpha)
        cv2.line(overlay, pt1['coord'], pt2['coord'], color, thickness)

    cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Model path")
    parser.add_argument("--video", type=str, required=True, help="Video path")
    parser.add_argument("--output", type=str, default="tracking_output.mp4", help="Output video path")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to track")
    # New flags for visualization options
    parser.add_argument("--trails", action="store_true", help="Enable tracking trails visualization")
    parser.add_argument("--bbox", action="store_true", help="Enable visualization of bounding boxes around tracked objects")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Load the YOLO model (Ultralytics)
    # -------------------------------------------------------------------------
    model = YOLO(args.model)

    # -------------------------------------------------------------------------
    # 2. Initialize the BotSort tracker
    #    - Customize reid_weights, device, etc. as needed
    # -------------------------------------------------------------------------
    # For CPU:
    device = torch.device("cpu")
    # For GPU (if available):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tracker = BotSort(
        reid_weights=Path("osnet_x0_25_msmt17.pt"),  # or your re-ID model
        device=device,
        half=False,  # set True if using GPU + half precision
    )

    # -------------------------------------------------------------------------
    # 3. Video capture setup
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video {args.video}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional: Resize if dimensions too large
    max_dim = 1200
    scale = min(max_dim / width, max_dim / height)
    width, height = int(width * scale), int(height * scale)

    fps = cap.get(cv2.CAP_PROP_FPS) or 60  # default to 60 if unavailable
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # -------------------------------------------------------------------------
    # 4. Track state data structures
    # -------------------------------------------------------------------------
    track_history = defaultdict(list)  # track_id -> list of points { 'coord':(x,y), 'fade':int }
    track_timers = {}  # track_id -> elapsed time in seconds
    FADE_STEP = 5

    frame_index = 0

    # -------------------------------------------------------------------------
    # 5. Main loop: detection -> tracking -> annotation
    # -------------------------------------------------------------------------
    while True:
        success, frame_raw = cap.read()
        if not success:
            break
        frame_index += 1

        # Resize the frame if it's bigger than our max_dim
        if frame_raw.shape[0] > max_dim or frame_raw.shape[1] > max_dim:
            frame = cv2.resize(frame_raw, (width, height))
        else:
            frame = frame_raw

        annotated_frame = frame.copy()

        # -----------------------------------------------------
        # (A) Inference with YOLO (Ultralytics)
        # -----------------------------------------------------
        #   classes=args.classes restricts detection to certain class IDs
        results = model.predict(frame, imgsz=1024, classes=args.classes)
        # results is a list of ultralytics.engine.results.Results
        # We'll take results[0] for the first (and only) image in this batch
        if not results or len(results) == 0:
            # No detections at all, skip
            out.write(annotated_frame)
            cv2.imshow("BoXMOT + YOLO", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Convert YOLO output to a NumPy array: [x1, y1, x2, y2, conf, class_id]
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Nx4
        confs = results[0].boxes.conf.cpu().numpy()  # Nx1
        cls_ids = results[0].boxes.cls.cpu().numpy()  # Nx1

        # Make a single array for BotSort
        # shape: Nx6 -> [x1, y1, x2, y2, score, class_id]
        detections = np.concatenate([
            boxes, 
            confs.reshape(-1, 1),
            cls_ids.reshape(-1, 1)
        ], axis=1)

        # -----------------------------------------------------
        # (B) Update BotSort tracker
        # -----------------------------------------------------
        # BotSort returns an array of shape Nx8 (by default):
        # [x1, y1, x2, y2, track_id, conf, class, ...
        tracks = tracker.update(detections, frame)

        # -----------------------------------------------------
        # (C) Annotate the frame with tracking info
        # -----------------------------------------------------
        for t in tracks:
            x1, y1, x2, y2, track_id, conf, cls_id = t[:7]

            # Convert floats to int for drawing
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            cls_id = int(cls_id)

            # Update timer: if new, start at 0; otherwise, increment by frame duration.
            if track_id not in track_timers:
                track_timers[track_id] = 0.0
            else:
                track_timers[track_id] += 1.0 / fps

            elapsed_time = track_timers[track_id]
            # Use a threshold (e.g. 10s => ratio=1.0) over which the box becomes fully red.
            ratio = min(elapsed_time / 50.0, 1.0)
            dynamic_color = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR from green to red

            # Label: object name, track ID, elapsed time
            # If you only have one class or custom classes, adjust name accordingly
            class_name = results[0].names.get(cls_id, f"cls{cls_id}")
            label_text = f"{class_name} | ID: {track_id} | {elapsed_time:.1f}s"
            label_org = (x1, max(y1 - 25, 0))
            put_label_with_background(annotated_frame, label_text, label_org,
                                      font_scale=0.6, color=(255, 255, 255),
                                      bg_color=dynamic_color, alpha=0.6)

            # Optionally draw the bounding box
            if args.bbox:
                draw_filled_transparent_box(annotated_frame, (x1, y1), (x2, y2), dynamic_color)

            # If trails are enabled, record the center point
            if args.trails:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if track_id not in track_colors:
                    track_colors[track_id] = get_random_color()
                track_history[track_id].append({
                    'coord': (cx, cy),
                    'fade': 255  # start fully visible
                })

        # -----------------------------------------------------
        # (D) Draw fade-out trails if enabled
        # -----------------------------------------------------
        if args.trails:
            # Fade out old points
            for t_id in list(track_history.keys()):
                for pt in track_history[t_id]:
                    pt['fade'] -= FADE_STEP
                track_history[t_id] = [p for p in track_history[t_id] if p['fade'] > 0]
                if not track_history[t_id]:
                    track_history.pop(t_id, None)
                    track_colors.pop(t_id, None)

            # Draw the polylines
            for t_id, points in track_history.items():
                if t_id in track_colors:
                    draw_fading_polyline(annotated_frame, points, track_colors[t_id])

        # -----------------------------------------------------
        # (E) Show the frame
        # -----------------------------------------------------
        cv2.imshow("BoXMOT + YOLO", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
