import random
from collections import defaultdict
from argparse import ArgumentParser

from ultralytics import YOLO
import cv2
import numpy as np

# For reproducible random colors (remove seed if you want truly random every run)
random.seed(42)

# Dictionary to store a unique color for each track ID
track_colors = {}

def get_random_color():
    """Returns a random BGR color tuple."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))



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
    thickness = 1

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x, y = org
    x2, y2 = x + text_w + 10, y + text_h + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, text, (x + 5, y + text_h + 3), font, font_scale, color, 1, cv2.LINE_AA)

def draw_fading_polyline(frame, points, color, scale=1.0):
    """
    Draw a continuous polyline that fades from tail to head.
    
    'points' is a list of dicts: 
       [{ 'coord': (x, y), 'fade': int }, ... ]
    'color' is the base track color (B, G, R).
    'scale' is how we scale the points for display in 'frame'.
    """
    if len(points) < 2:
        return

    overlay = frame.copy()

    for i in range(1, len(points)):
        pt1 = points[i - 1]
        pt2 = points[i]

        # Average fade for this segment => brightness factor
        fade_avg = (pt1['fade'] + pt2['fade']) / 2.0
        alpha = max(0.0, min(1.0, fade_avg / 255.0))  # clamp 0..1

        # Optionally fade color, here we just scale line thickness
        thickness = 1 + int(3 * alpha)
        scaled_coord1 = (int(pt1['coord'][0] * scale), int(pt1['coord'][1] * scale))
        scaled_coord2 = (int(pt2['coord'][0] * scale), int(pt2['coord'][1] * scale))
        cv2.line(overlay, scaled_coord1, scaled_coord2, color, thickness)

    cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)

def get_side_of_line(point, line_start, line_end):
    """
    Returns +1 if 'point' is on one side of the line,
    -1 if it's on the other side,
     0 if it's exactly on the line.
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    side = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if side > 0:
        return 1
    elif side < 0:
        return -1
    else:
        return 0

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Model path")
    parser.add_argument("--video", type=str, required=True, help="Video path")
    parser.add_argument("--output", type=str, default="counting_output.mp4", help="Output video path")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to track")
    
    # Although the script still accepts these arguments,
    # the interactive gate selection below will override them.
    parser.add_argument("--line-a", type=float, nargs=4, default=[0.6, 0.0, 0.6, 1.0],
                        help="Line A in normalized coords, e.g. 0.6 0.0 0.6 1.0")
    parser.add_argument("--line-b", type=float, nargs=4, default=[0.61, 0.0, 0.61, 1.0],
                        help="Line B in normalized coords, e.g. 0.61 0.0 0.61 1.0")

    # Inference and display sizes
    parser.add_argument("--max-dim", type=int, default=2560,
                        help="Max dimension for YOLO inference (model will letterbox internally)")
    parser.add_argument("--vis-max-dim", type=int, default=1200,
                        help="Max dimension for the display window/video output")

    # Fade and ignore parameters
    parser.add_argument("--fade-step", type=int, default=5, help="Fade-out step per frame")
    parser.add_argument("--ignore-frames", type=int, default=5,
                        help="Number of frames to ignore after a crossing event")

    # Optionally override the video FPS if needed
    parser.add_argument("--override-fps", type=float, default=0.0,
                        help="If > 0, use this FPS for the output video instead of source FPS")

    # Visualization toggles
    parser.add_argument("--show-class", action="store_true",
                        help="If set, display class name on the bounding box label")
    parser.add_argument("--show-bbox", action="store_true",
                        help="If set, display bounding boxes")
    parser.add_argument("--show-trails", action="store_true",
                        help="If set, display fading trails for tracks")

    args = parser.parse_args()

    # Load the YOLO model
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video {args.video}")
        return

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------------------------------------------------
    # Interactive Gate Selection: Draw Two Lines
    # ---------------------------------------------------
    # Read the first frame
    success, first_frame = cap.read()
    if not success:
        print("Error: could not read first frame")
        return

    # Scale the first frame for interactive selection so that it is at most 1024 pixels wide.
    max_interactive_width = 1024
    scale_factor = 1.0
    if first_frame.shape[1] > max_interactive_width:
        scale_factor = max_interactive_width / first_frame.shape[1]
        temp_frame = cv2.resize(first_frame, (int(first_frame.shape[1] * scale_factor),
                                              int(first_frame.shape[0] * scale_factor)))
    else:
        temp_frame = first_frame.copy()

    gate_points = []  # will store 4 points: two for Line A and two for Line B

    instruction_text = ("Select Gate Lines:\n"
                        " - Click two points for Line A (displayed in red).\n"
                        " - Then click two points for Line B (displayed in yellow).\n"
                        "Press 'q' to abort.")
    print(instruction_text)

    def gate_callback(event, x, y, flags, param):
        nonlocal gate_points, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(gate_points) < 4:
                gate_points.append((x, y))
                # Draw a small circle at the clicked point
                cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
                # If two points have been selected, draw Line A in red.
                if len(gate_points) == 2:
                    cv2.line(temp_frame, gate_points[0], gate_points[1], (0, 0, 255), 2)
                # If four points have been selected, draw Line B in yellow.
                elif len(gate_points) == 4:
                    cv2.line(temp_frame, gate_points[2], gate_points[3], (0, 255, 255), 2)
        elif event == cv2.EVENT_MOUSEMOVE:
            # For dynamic drawing: show a temporary line from the last clicked point to current mouse pos
            temp_frame_copy = temp_frame.copy()
            if len(gate_points) in [1, 3]:
                last_pt = gate_points[-1]
                color = (0, 0, 255) if len(gate_points) == 1 else (0, 255, 255)
                cv2.line(temp_frame_copy, last_pt, (x, y), color, 2)
            cv2.imshow("Select Gate Lines", temp_frame_copy)

    cv2.namedWindow("Select Gate Lines")
    cv2.setMouseCallback("Select Gate Lines", gate_callback)

    while True:
        cv2.imshow("Select Gate Lines", temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if len(gate_points) == 4:
            break
        if key == ord('q'):
            print("Gate selection aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select Gate Lines")
    # Convert the interactive (scaled) gate points back to original coordinates.
    gate_points = [(int(x / scale_factor), int(y / scale_factor)) for (x, y) in gate_points]

    # Define the two gate lines from the collected points
    LINEA_P1, LINEA_P2 = gate_points[0], gate_points[1]
    LINEB_P1, LINEB_P2 = gate_points[2], gate_points[3]

    # Reset video pointer to first frame for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # For final on-screen display size
    if original_width > args.vis_max_dim or original_height > args.vis_max_dim:
        vis_scale = min(args.vis_max_dim / original_width, args.vis_max_dim / original_height)
        vis_width, vis_height = int(original_width * vis_scale), int(original_height * vis_scale)
    else:
        vis_scale = 1.0
        vis_width, vis_height = original_width, original_height

    # Determine FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback if not available
    if args.override_fps > 0:
        fps = args.override_fps

    # Create a VideoWriter to save MP4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (vis_width, vis_height))

    ################################################
    # Track state data structures (Double-Line Logic)
    ################################################
    track_history = defaultdict(list)  # track_id -> list of dicts { 'coord':(x,y), 'fade':int }
    FADE_STEP = args.fade_step

    # Double-line crossing counters
    crossing_count_forward = 0   # A -> B
    crossing_count_backward = 0  # B -> A

    # For crossing states and ignore periods
    IGNORE_FRAMES = args.ignore_frames
    track_sideA = defaultdict(lambda: 0)  # track_id -> side wrt Line A
    track_sideB = defaultdict(lambda: 0)  # track_id -> side wrt Line B
    track_last_line_crossed = defaultdict(lambda: None)  # 'A', 'B', or None
    track_ignore_until = defaultdict(int)

    frame_index = 0

    # If the model has names, we can retrieve them (some custom models might not)
    class_names = model.names if hasattr(model, 'names') else {}

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_index += 1

        # Copy for annotation (will be resized for display only)
        annotated_frame = frame.copy()
        if vis_scale != 1.0:
            annotated_frame = cv2.resize(annotated_frame, (vis_width, vis_height))

        # Draw the two gate lines on the annotated frame for visualization
        LINEA_P1_scaled = (int(LINEA_P1[0] * vis_scale), int(LINEA_P1[1] * vis_scale))
        LINEA_P2_scaled = (int(LINEA_P2[0] * vis_scale), int(LINEA_P2[1] * vis_scale))
        LINEB_P1_scaled = (int(LINEB_P1[0] * vis_scale), int(LINEB_P1[1] * vis_scale))
        LINEB_P2_scaled = (int(LINEB_P2[0] * vis_scale), int(LINEB_P2[1] * vis_scale))

        cv2.line(annotated_frame, LINEA_P1_scaled, LINEA_P2_scaled, (255, 0, 0), 3)
        cv2.line(annotated_frame, LINEB_P1_scaled, LINEB_P2_scaled, (0, 255, 0), 3)

        # Run YOLO tracking on the *original* frame
        results = model.track(
            frame,
            persist=True,
            classes=args.classes,
            imgsz=args.max_dim  # let YOLO handle internal resizing
        )

        if len(results) > 0 and results[0].boxes and results[0].boxes.id is not None:
            xywh_boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # Optionally retrieve confidence and class info:
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None
            class_idxs = None
            if results[0].boxes.cls is not None:
                class_idxs = results[0].boxes.cls.cpu().numpy()

            for i, (box, track_id) in enumerate(zip(xywh_boxes, track_ids)):
                x_center, y_center, w, h = box
                cx, cy = int(x_center), int(y_center)

                # Assign a random color if not already assigned
                if track_id not in track_colors:
                    track_colors[track_id] = get_random_color()
                color = track_colors[track_id]

                # --------------------------------------------------------------
                # DRAW BOUNDING BOX (only if enabled)
                # --------------------------------------------------------------
                if args.show_bbox:
                    # Original coords:
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    # Scale for display
                    x1_disp = int(x1 * vis_scale)
                    y1_disp = int(y1 * vis_scale)
                    x2_disp = int(x2 * vis_scale)
                    y2_disp = int(y2 * vis_scale)

                    cv2.rectangle(annotated_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 1)

                    # Optionally show class name or track ID in a label
                    cls_id = int(class_idxs[i]) if class_idxs is not None else -1
                    cls_name = class_names.get(cls_id, str(cls_id))
                    label_str = f"{'car'} | id: {int(track_id)}"
                    if args.show_class and class_idxs is not None:
                        label_str += f" ({cls_name})"

                    # Put label above the box
                    label_org = (x1_disp, max(0, y1_disp - 25))
                    put_label_with_background(
                        annotated_frame,
                        label_str,
                        label_org,
                        font_scale=0.3,
                        color=(255, 255, 255),
                        bg_color=color,
                        alpha=0.6
                    )

                # --------------------------------------------------------------
                # DOUBLE-LINE CROSSING LOGIC and track history update
                # --------------------------------------------------------------
                track_history[track_id].append({
                    'coord': (cx, cy),
                    'fade': 255
                })

                current_sideA = get_side_of_line((cx, cy), LINEA_P1, LINEA_P2)
                current_sideB = get_side_of_line((cx, cy), LINEB_P1, LINEB_P2)

                old_sideA = track_sideA[track_id]
                old_sideB = track_sideB[track_id]

                # Use the previous non-zero side if needed
                effective_sideA = current_sideA if current_sideA != 0 else (old_sideA if old_sideA != 0 else 0)
                effective_sideB = current_sideB if current_sideB != 0 else (old_sideB if old_sideB != 0 else 0)

                crossedA = (old_sideA != 0 and effective_sideA != old_sideA)
                crossedB = (old_sideB != 0 and effective_sideB != old_sideB)

                track_sideA[track_id] = effective_sideA
                track_sideB[track_id] = effective_sideB

                # Check crossing of Line A
                if crossedA:
                    if track_last_line_crossed[track_id] == 'B':
                        crossing_count_backward += 1
                        track_last_line_crossed[track_id] = None
                        track_ignore_until[track_id] = frame_index + IGNORE_FRAMES
                    else:
                        track_last_line_crossed[track_id] = 'A'

                # Check crossing of Line B
                if crossedB:
                    if track_last_line_crossed[track_id] == 'A':
                        crossing_count_forward += 1
                        track_last_line_crossed[track_id] = None
                        track_ignore_until[track_id] = frame_index + IGNORE_FRAMES
                    else:
                        track_last_line_crossed[track_id] = 'B'

        # --------------------------------------------------------------
        # Fade out old points & clean up track history
        # --------------------------------------------------------------
        for t_id in list(track_history.keys()):
            for pt in track_history[t_id]:
                pt['fade'] -= FADE_STEP
            track_history[t_id] = [p for p in track_history[t_id] if p['fade'] > 0]
            if not track_history[t_id]:
                track_history.pop(t_id, None)
                track_colors.pop(t_id, None)
                track_sideA.pop(t_id, None)
                track_sideB.pop(t_id, None)
                track_last_line_crossed.pop(t_id, None)
                track_ignore_until.pop(t_id, None)

        # --------------------------------------------------------------
        # Draw the fading trails if enabled
        # --------------------------------------------------------------
        if args.show_trails:
            for t_id, points in track_history.items():
                if t_id in track_colors:
                    draw_fading_polyline(annotated_frame, points, track_colors[t_id], vis_scale)

        # --------------------------------------------------------------
        # Display crossing counters on-screen
        # --------------------------------------------------------------
        counter_text = (
            f"-> (In)  : {crossing_count_forward}    "
            f"<- (Out)  : {crossing_count_backward}"
        )
        put_label_with_background(
            annotated_frame,
            counter_text,
            (20, 20),
            font_scale=0.7,
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
            alpha=0.6
        )

        # Show and write the annotated frame
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
