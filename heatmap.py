import random
from collections import defaultdict
from argparse import ArgumentParser
import math

from ultralytics import YOLO
import cv2
import numpy as np

# For reproducible random colors
random.seed(42)

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

###############################################################################
# Helper function to keep visuals consistent for any 'proc_width'.
###############################################################################
def scale_value(value_final: float, proc_width: int, final_width: int) -> int:
    """
    Convert a 'desired size' in the final dimension to an integer in the
    processing dimension. For line thickness, corner lengths, circle radii, etc.
    """
    if final_width <= 0:
        return int(value_final)
    scale = proc_width / float(final_width)
    return max(1, int(round(value_final * scale)))

def scale_font(value_final: float, proc_width: int, final_width: int) -> float:
    """
    Same concept but returns a float for font scale (for text).
    """
    if final_width <= 0:
        return value_final
    scale = proc_width / float(final_width)
    return value_final * scale

###############################################################################
# Modified drawing functions that call 'scale_value' or 'scale_font'
###############################################################################
def draw_transparent_box(frame, top_left, bottom_right, color, alpha=0.4,
                         thickness=2, corner_length=20,
                         proc_width=1280, final_width=1280):
    """
    thickness, corner_length are in 'final' units; we scale them to 'proc_width'.
    """
    t = scale_value(thickness, proc_width, final_width)
    cl = scale_value(corner_length, proc_width, final_width)

    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.rectangle(frame, top_left, bottom_right, color, t)
    x1, y1 = top_left
    x2, y2 = bottom_right

    # corners
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, t)
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, t)
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, t)
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, t)

def put_label_with_background(frame, text, org,
                              font_scale=0.6,
                              color=(255, 255, 255),
                              bg_color=(0, 0, 0),
                              alpha=0.6,
                              proc_width=1280,
                              final_width=1280):
    """
    Font scale is in 'final' units => convert to 'proc_width'.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness_final = 1
    thickness_proc = scale_value(thickness_final, proc_width, final_width)
    font_scale_proc = scale_font(font_scale, proc_width, final_width)

    text_size, _ = cv2.getTextSize(text, font, font_scale_proc, thickness_proc)
    text_w, text_h = text_size
    x, y = org
    x2, y2 = x + text_w + 10, y + text_h + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, text, (x + 5, y + text_h + 3),
                font, font_scale_proc, color, thickness_proc, cv2.LINE_AA)

def draw_fading_polyline(frame, points, color,
                         base_thickness_final=1,
                         max_thickness_final=10,
                         proc_width=1280,
                         final_width=1280):
    """
    We'll interpret 'base_thickness_final' and 'max_thickness_final' as thickness
    in final resolution, then scale them to 'proc_width'.
    """
    if len(points) < 2:
        return

    overlay = frame.copy()
    base_th = scale_value(base_thickness_final, proc_width, final_width)
    max_th = scale_value(max_thickness_final, proc_width, final_width)

    for i in range(1, len(points)):
        pt1 = points[i - 1]
        pt2 = points[i]
        fade_avg = (pt1['fade'] + pt2['fade']) / 2.0
        alpha = max(0.0, min(1.0, fade_avg / 255.0))
        thickness_proc = int(base_th + alpha * (max_th - base_th))
        cv2.line(overlay, pt1['coord'], pt2['coord'], color, thickness_proc)

    cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)

###############################################################################
# Main script
###############################################################################
def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Model path")
    parser.add_argument("--video", type=str, required=True, help="Video path")
    parser.add_argument("--output", type=str, default="heatmap_output.mp4", help="Output video path")
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to track")
    parser.add_argument("--visualization", type=str, choices=["trails", "bbox"], default="trails",
                        help="Visualization mode: 'trails' or 'bbox'")
    parser.add_argument("--max_inference_width", type=int, default=3840,
                        help="Max width used for inference/processing pipeline")
    parser.add_argument("--max_visual_width", type=int, default=1280,
                        help="Max width for final visualization/output")
    parser.add_argument("--blur_kernel", type=int, default=15,
                        help="Gaussian blur kernel size for heatmap smoothing")
    parser.add_argument("--min_box_area", type=float, default=4000.0,
                        help="Minimum bounding box area (pixels) to keep; discard smaller objects")
    parser.add_argument("--heat_factor", type=float, default=0.5,
                        help="Heat per pixel of distance traveled (in final coords)")
    parser.add_argument("--init_heat", type=float, default=0.0,
                        help="Heat added when a track first appears (optional)")
    parser.add_argument("--max_frame_gap", type=int, default=1,
                        help="If an object is missing > this frames, treat re-appearance as new track.")
    parser.add_argument("--max_teleport_dist", type=float, default=9999.0,
                        help="If distance > this, treat as new track (prevents hot lines).")
    parser.add_argument("--min_move_dist", type=float, default=0.0,
                        help="Minimum average speed to be considered moving.")
    parser.add_argument("--speed_alpha", type=float, default=0.3,
                        help="Alpha for exponential moving average of speed (0 < alpha < 1).")
    args = parser.parse_args()

    # YOLO model
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video {args.video}")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Original video: {orig_width}x{orig_height}, {fps:.2f} FPS")

    # 1) Compute the 'processing' dimension
    if orig_width > args.max_inference_width:
        proc_scale = args.max_inference_width / orig_width
    else:
        proc_scale = 1.0

    proc_width = int(orig_width * proc_scale)
    proc_height = int(orig_height * proc_scale)

    # 2) Compute final dimension
    if proc_width > args.max_visual_width:
        vis_scale = args.max_visual_width / proc_width
    else:
        vis_scale = 1.0

    final_width = int(proc_width * vis_scale)
    final_height = int(proc_height * vis_scale)
    print(f"Processing: {proc_width}x{proc_height}, Final: {final_width}x{final_height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (final_width, final_height))

    # Data
    heatmap = np.zeros((proc_height, proc_width), dtype=np.float32)
    max_heat = 50.0

    track_history = defaultdict(list)
    track_colors = {}

    track_distance_final = defaultdict(float)  # store distance in "final" coords
    track_deposit = defaultdict(float)
    track_speed_ema = defaultdict(float)

    last_pos_proc = {}        # last position in "proc" coords
    last_pos_final = {}       # last position in "final" coords
    last_frame_seen = {}

    FADE_STEP = 5
    frame_idx = 0

    def reset_track(tid, cx_p, cy_p, cx_f, cy_f):
        """Initialize a track at both processing coords and final coords."""
        track_distance_final[tid] = 0.0
        track_deposit[tid] = args.init_heat
        track_speed_ema[tid] = 0.0
        last_pos_proc[tid] = (cx_p, cy_p)
        last_pos_final[tid] = (cx_f, cy_f)

        if args.init_heat > 0:
            # draw a circle in processing dimension
            radius_proc = scale_value(15, proc_width, final_width)
            tmp = np.zeros_like(heatmap)
            cv2.circle(tmp, (cx_p, cy_p), radius_proc, args.init_heat, -1, lineType=cv2.LINE_AA)
            heatmap[:] += tmp

    while True:
        success, frame_orig = cap.read()
        if not success:
            break
        frame_idx += 1

        # Resize to processing dimension
        frame_proc = cv2.resize(frame_orig, (proc_width, proc_height))

        # YOLO inference
        results = model.track(
            frame_proc,
            persist=True,
            classes=args.classes,
            imgsz=proc_width
        )

        annotated_frame = frame_proc.copy()

        # Optional blur
        if args.blur_kernel > 1:
            heatmap_smooth = cv2.GaussianBlur(heatmap, (args.blur_kernel, args.blur_kernel), 0)
        else:
            heatmap_smooth = heatmap

        # Color heatmap
        heatmap_clipped = np.clip(heatmap_smooth, 0, max_heat)
        heatmap_disp = np.uint8((heatmap_clipped / max_heat) * 255)
        colored_heatmap = cv2.applyColorMap(heatmap_disp, cv2.COLORMAP_JET)
        cv2.addWeighted(colored_heatmap, 0.4, annotated_frame, 0.6, 0, annotated_frame)

        if len(results) > 0 and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for box, t_id, c_id in zip(boxes, track_ids, class_ids):
                x_center_p, y_center_p, w_p, h_p = box  # in processing coords
                area_p = w_p * h_p
                if area_p < args.min_box_area:
                    continue

                # Convert center to int
                cx_p = int(x_center_p)
                cy_p = int(y_center_p)

                # Convert to final coords => to measure distances in final dimension
                cx_f = int(round(cx_p * (final_width / proc_width)))
                cy_f = int(round(cy_p * (final_height / proc_height)))

                # BBox corners in proc coords for visualization
                x1 = int(x_center_p - w_p/2)
                y1 = int(y_center_p - h_p/2)
                x2 = int(x_center_p + w_p/2)
                y2 = int(y_center_p + h_p/2)

                # Assign color
                if t_id not in track_colors:
                    track_colors[t_id] = get_random_color()
                color = track_colors[t_id]

                # Check reappearance
                gap = frame_idx - last_frame_seen.get(t_id, -9999)
                reappeared = (gap > args.max_frame_gap)

                if t_id not in last_pos_proc or reappeared:
                    # new track
                    reset_track(t_id, cx_p, cy_p, cx_f, cy_f)
                    if args.visualization == "trails":
                        track_history[t_id] = []
                else:
                    last_cx_p, last_cy_p = last_pos_proc[t_id]
                    last_cx_f, last_cy_f = last_pos_final[t_id]

                    # Dist in processing coords (for line endpoints)
                    dist_proc = math.hypot(cx_p - last_cx_p, cy_p - last_cy_p)

                    # Dist in final coords (for consistent total heat)
                    dist_final = math.hypot(cx_f - last_cx_f, cy_f - last_cy_f)

                    # Teleport check based on proc dist or final dist—choose either
                    if dist_proc > args.max_teleport_dist:
                        reset_track(t_id, cx_p, cy_p, cx_f, cy_f)
                        if args.visualization == "trails":
                            track_history[t_id] = []
                    else:
                        # speed EMA in final coords
                        old_ema = track_speed_ema[t_id]
                        alpha = args.speed_alpha
                        new_ema = alpha * old_ema + (1 - alpha) * dist_final
                        track_speed_ema[t_id] = new_ema

                        if new_ema < args.min_move_dist:
                            # skip
                            last_pos_proc[t_id] = (cx_p, cy_p)
                            last_pos_final[t_id] = (cx_f, cy_f)
                            last_frame_seen[t_id] = frame_idx
                            continue

                        # normal deposit
                        # 1) increment distance in FINAL coords
                        track_distance_final[t_id] += dist_final

                        desired_heat = track_distance_final[t_id] * args.heat_factor
                        current_deposit = track_deposit[t_id]
                        new_heat = desired_heat - current_deposit

                        if new_heat > 0 and dist_proc > 0:
                            # draw line in proc coords with thickness scaled for final dimension
                            thickness_proc = scale_value(15, proc_width, final_width)
                            temp = np.zeros_like(heatmap)
                            cv2.line(
                                temp,
                                (last_cx_p, last_cy_p),
                                (cx_p, cy_p),
                                1.0,
                                thickness=thickness_proc,
                                lineType=cv2.LINE_AA
                            )
                            temp *= new_heat
                            heatmap += temp
                            track_deposit[t_id] += new_heat

                        last_pos_proc[t_id] = (cx_p, cy_p)
                        last_pos_final[t_id] = (cx_f, cy_f)

                last_frame_seen[t_id] = frame_idx

                # Visualization
                if args.visualization == "bbox":
                    # Draw bounding box in proc coords
                    draw_transparent_box(
                        annotated_frame,
                        (x1, y1), (x2, y2),
                        color,
                        alpha=0.4,
                        thickness=2,
                        corner_length=0,
                        proc_width=proc_width,
                        final_width=final_width
                    )
                    # Label
                    label_org = (x1, max(y1 - 25, 0))
                    name_str = results[0].names[int(c_id)]
                    put_label_with_background(
                        annotated_frame,
                        f"{name_str}",
                        label_org,
                        font_scale=0.4,
                        color=(255, 255, 255),
                        bg_color=color,
                        alpha=0.6,
                        proc_width=proc_width,
                        final_width=final_width
                    )
                else:  # "trails"
                    track_history[t_id].append({'coord': (cx_p, cy_p), 'fade': 255})

        # Fade out old trails
        if args.visualization == "trails":
            for t_id in list(track_history.keys()):
                for pt in track_history[t_id]:
                    pt['fade'] -= FADE_STEP
                track_history[t_id] = [p for p in track_history[t_id] if p['fade'] > 0]
                if not track_history[t_id]:
                    track_history.pop(t_id, None)
                    track_colors.pop(t_id, None)
                    if t_id in last_pos_proc:
                        last_pos_proc.pop(t_id)
                    if t_id in last_pos_final:
                        last_pos_final.pop(t_id)
                    if t_id in track_distance_final:
                        track_distance_final.pop(t_id)
                    if t_id in track_deposit:
                        track_deposit.pop(t_id)
                    if t_id in track_speed_ema:
                        track_speed_ema.pop(t_id)
                    if t_id in last_frame_seen:
                        last_frame_seen.pop(t_id)

            # draw polylines
            for t_id, pts in track_history.items():
                if t_id in track_colors:
                    draw_fading_polyline(
                        annotated_frame,
                        pts,
                        track_colors[t_id],
                        base_thickness_final=1,
                        max_thickness_final=10,
                        proc_width=proc_width,
                        final_width=final_width
                    )

        # Finally resize to final
        annotated_frame_final = cv2.resize(annotated_frame, (final_width, final_height))
        cv2.imshow("Traffic Heatmap Overlay", annotated_frame_final)
        out.write(annotated_frame_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
