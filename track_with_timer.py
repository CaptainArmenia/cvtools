import random
from collections import defaultdict
from argparse import ArgumentParser

from ultralytics import YOLO
import cv2
import numpy as np

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
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Model path")
    parser.add_argument("--video", type=str, required=True, help="Video path")
    parser.add_argument("--output", type=str, default="tracking_output.mp4", help="Output video path")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to track")
    # New flags for visualization options
    parser.add_argument("--trails", action="store_true", help="Enable tracking trails visualization")
    parser.add_argument("--bbox", action="store_true", help="Enable visualization of bounding boxes around tracked objects")
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

    # Resize max dimension to 2048
    max_dim = 1200
    scale = min(max_dim / width, max_dim / height)
    width, height = int(width * scale), int(height * scale)

    fps = cap.get(cv2.CAP_PROP_FPS) or 60  # use video's FPS, defaulting to 60 if unavailable
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    ################################################
    # Track state data structures
    ################################################
    track_history = defaultdict(list)  # track_id -> list of dicts { 'coord':(x,y), 'fade':int }
    track_timers = {}  # track_id -> elapsed time in seconds
    FADE_STEP = 5  # fade out by 5 per frame

    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_index += 1

        if frame.shape[0] > max_dim or frame.shape[1] > max_dim:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        annotated_frame = frame.copy()

        results = model.track(frame, persist=True, classes=args.classes, imgsz=1024, tracker="botsort.yaml")

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()  # get all class IDs

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x_center, y_center, w, h = box
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Update timer: if new, start at 0; otherwise, increment by frame duration.
                if track_id not in track_timers:
                    track_timers[track_id] = 0.0
                else:
                    track_timers[track_id] += 1.0 / fps

                elapsed_time = track_timers[track_id]
                # Use a threshold (10 seconds) over which the box becomes fully red.
                ratio = min(elapsed_time / 50.0, 1.0)
                # Interpolate from green (0,255,0) to red (0,0,255). (BGR format)
                dynamic_color = (0, int(255 * (1 - ratio)), int(255 * ratio))

                # Label now includes class name, object id, and elapsed time (formatted to one decimal)
                label_text = f"{results[0].names[int(cls_id)]} | ID: {int(track_id)} | {elapsed_time:.1f}s"
                label_org = (x1, max(y1 - 25, 0))
                put_label_with_background(
                    annotated_frame,
                    label_text,
                    label_org,
                    font_scale=0.6,
                    color=(255, 255, 255),
                    bg_color=dynamic_color,
                    alpha=0.6
                )

                # Optionally draw the filled bounding box with the dynamic color
                if args.bbox:
                    draw_filled_transparent_box(annotated_frame, (x1, y1), (x2, y2), dynamic_color)

                # If trails are enabled, add the center point to the track with full fade
                if args.trails:
                    cx, cy = int(x_center), int(y_center)
                    if track_id not in track_colors:
                        track_colors[track_id] = get_random_color()
                    track_history[track_id].append({
                        'coord': (cx, cy),
                        'fade': 255  # start fully visible
                    })

        if args.trails:
            # Fade out old points & clean up
            for t_id in list(track_history.keys()):
                for pt in track_history[t_id]:
                    pt['fade'] -= FADE_STEP
                track_history[t_id] = [p for p in track_history[t_id] if p['fade'] > 0]
                if not track_history[t_id]:
                    track_history.pop(t_id, None)
                    track_colors.pop(t_id, None)

            # Draw the polylines (fading trails) for all remaining tracks
            for t_id, points in track_history.items():
                if t_id in track_colors:
                    draw_fading_polyline(annotated_frame, points, track_colors[t_id])

        # Show the annotated frame
        cv2.imshow("Double-Line Gate Method", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
