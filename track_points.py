import argparse
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from scipy.spatial import cKDTree

# Import Kornia modules for DISK and LightGlue
from kornia.feature import DISK, LightGlue, DeDoDe
###############################################################################
# Tracker: remains mostly unchanged.
###############################################################################
class PointTracker:
    """
    Tracks keypoints between frames, maintaining a history of their positions
    and handling lost tracks by eventual deletion.
    """
    def __init__(
        self,
        min_movement: int = 5,
        point_lifespan: int = 10,
        static_damping_frames: int = 1,
        max_speed: int = 30
    ):
        self.tracks = {}
        self.next_track_id = 0
        self.min_movement = min_movement
        self.point_lifespan = point_lifespan
        self.static_damping_frames = static_damping_frames
        self.max_speed = max_speed
        self.keypoint_id_to_track_id = {}
        self.frame_counter = 0

    def update(self, prev_kpts, curr_kpts, matches):
        """
        Update tracks using chain matching. For each match (prev_idx -> curr_idx),
        if the previous keypoint was already tracked, continue that track;
        otherwise, start a new track with the previous and current keypoints.
        """
        self.frame_counter += 1
        new_mapping = {}  # mapping: current keypoint index -> track id

        # Process each match (assuming matches array aligns with prev_kpts)
        for prev_idx, curr_idx in enumerate(matches):
            # Skip if no valid match.
            if curr_idx == -1:
                continue

            if prev_idx in self.keypoint_id_to_track_id:
                # Continue an existing track.
                track_id = self.keypoint_id_to_track_id[prev_idx]
                self.tracks[track_id]["pts"].append((self.frame_counter, curr_kpts[curr_idx]))
                self.tracks[track_id]["miss_count"] = 0
            else:
                # Start a new track.
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "pts": [
                        (self.frame_counter - 1, prev_kpts[prev_idx]),
                        (self.frame_counter, curr_kpts[curr_idx])
                    ],
                    "miss_count": 0,
                    "static_frames": 0
                }
            new_mapping[curr_idx] = track_id

        # Increment miss count for tracks without a new match.
        assigned_track_ids = set(new_mapping.values())
        for track_id in self.tracks:
            if track_id not in assigned_track_ids:
                self.tracks[track_id]["miss_count"] = self.tracks[track_id].get("miss_count", 0) + 1

        # Prune tracks that have been missing for too long.
        self.tracks = {track_id: track for track_id, track in self.tracks.items() if track["miss_count"] < 1}

        # Prune old points based on lifespan.
        for track in self.tracks.values():
            if "pts" in track:
                track["pts"] = [pt for pt in track["pts"] if (self.frame_counter - pt[0]) < self.point_lifespan]

        self.keypoint_id_to_track_id = new_mapping

        # Update static_frames counter based on recent movement.
        for track in self.tracks.values():
            if "pts" in track and len(track["pts"]) >= 3:
                pts_only = np.array([p[1] for p in track["pts"][-3:]])
                dx = pts_only[:, 0].max() - pts_only[:, 0].min()
                dy = pts_only[:, 1].max() - pts_only[:, 1].min()
                spread = np.sqrt(dx**2 + dy**2)
                track["static_frames"] = track.get("static_frames", 0) + 1 if spread < self.min_movement else 0

    def get_tracks(self, min_length=5):
        valid_tracks = []
        for track in self.tracks.values():
            if "pts" in track and len(track["pts"]) >= min_length:
                pts = [pt[1] for pt in track["pts"]]
                valid_tracks.append((pts, track.get("static_frames", 0)))
        return valid_tracks

    def draw_tracks(self, frame, tracks, scale=1.0):
        """
        Draw tracks on the provided frame.
        If scale != 1.0, the coordinates are multiplied by the scale factor.
        """
        stroke = 3
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                                cv2.COLORMAP_VIRIDIS).reshape(256, 3)
        
        for pts, static_frames in tracks:
            if len(pts) < 5 or static_frames >= self.static_damping_frames:
                continue
            # Scale the points for visualization.
            pts_scaled = np.array([(int(x * scale), int(y * scale)) for (x, y) in pts])
            diffs = pts_scaled[1:] - pts_scaled[:-1]
            speeds = np.linalg.norm(diffs, axis=1)
            if len(speeds) >= 3:
                kernel = np.ones(3) / 3.0
                speeds = np.convolve(speeds, kernel, mode='same')
            norm_speeds = np.clip((speeds / self.max_speed) * 255, 0, 255).astype(np.uint8)
            for i, speed_val in enumerate(norm_speeds):
                color = lut[speed_val].tolist()
                pt1 = tuple(pts_scaled[i])
                pt2 = tuple(pts_scaled[i + 1])
                cv2.line(frame, pt1, pt2, color, stroke)

def preprocess_image(img):
    """
    Enhance image contrast and sharpen details.
    - Convert BGR to LAB.
    - Apply CLAHE on the L channel.
    - Convert back to BGR.
    - Apply a sharpening filter.
    """
    # Convert to LAB and split channels.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    # Sharpen the image.
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

class KorniaLightGlueMatcher:
    """
    Feature matcher using Kornia's DISK for keypoint detection/description and
    LightGlue for matching.
    """
    def __init__(self, device: torch.device, lightglue_weights: str = "lightglue_outdoor"):
        self.device = device
        # Initialize DISK (expects an RGB image).
        self.detector = DISK.from_pretrained("depth").to(device).eval()
        # Initialize LightGlue (using the DISK configuration).
        self.lightglue = LightGlue("disk", weights=lightglue_weights, filter_threshold=0.1).to(device).eval()

    def detect(self, img, num_keypoints=2000):
        """
        Detect keypoints and compute descriptors for a single image.
        Args:
            img: Input image in BGR format.
            num_keypoints: Number of keypoints to detect.
        Returns:
            keypoints: Tensor of shape (N, 2)
            descriptors: Tensor of shape (N, D)
        """
        # Preprocess the image to enhance contrast and sharpen details.
        img_pre = preprocess_image(img)
        # Convert image from BGR to RGB.
        img_rgb = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float().to(self.device) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, H, W]
        outputs = self.detector(tensor, num_keypoints, pad_if_not_divisible=True, window_size=3)
        kp = outputs[0].keypoints  # shape: (N, 2)
        desc = outputs[0].descriptors  # shape: (1, N, D)
        desc = desc.squeeze(0)
        return kp, desc

    def match(self, prev_data, curr_data, image0, image1):
        """
        Match keypoints between two frames using LightGlue.
        Args:
            prev_data: Tuple (prev_kp, prev_desc) from the previous frame.
            curr_data: Tuple (curr_kp, curr_desc) from the current frame.
            image0: Previous frame (used for image size info).
            image1: Current frame.
        Returns:
            matches0: Array of indices (into prev_kp) for the matched keypoints.
        """
        prev_kp, prev_desc = prev_data  # (N,2) and (N, D)
        curr_kp, curr_desc = curr_data  # (M,2) and (M, D)

        # Add batch dimension.
        prev_kp_b = prev_kp.unsqueeze(0)       # shape: (1, N, 2)
        prev_desc_b = prev_desc.unsqueeze(0)     # shape: (1, N, D)
        curr_kp_b = curr_kp.unsqueeze(0)         # shape: (1, M, 2)
        curr_desc_b = curr_desc.unsqueeze(0)     # shape: (1, M, D)

        # Prepare image size info (width, height).
        image_shape = torch.tensor(image0.shape[:2][::-1]).to(self.device).unsqueeze(0)
        lg_input = {
            "image0": {"keypoints": prev_kp_b, "descriptors": prev_desc_b, "image_size": image_shape},
            "image1": {"keypoints": curr_kp_b, "descriptors": curr_desc_b, "image_size": image_shape},
        }
        with torch.no_grad():
            result = self.lightglue(lg_input)

        # 'matches0' and 'matches1' are indices into the keypoints.
        matches0 = result["matches0"][0].cpu().numpy()  # shape: (N,)
        return matches0


###############################################################################
# Main loop: use DISK+LightGlue for matching and tracking.
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Real-time matching & tracking using Kornia DISK+LightGlue"
    )
    # Separate widths: one for inference, one for visualization.
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to the output video file.")
    parser.add_argument("--inference_width", type=int, default=512, help="Maximum width for inference (detection/matching).")
    parser.add_argument("--viz_width", type=int, default=1920, help="Maximum width for visualization (output).")
    parser.add_argument("--keypoint_lifespan", type=int, default=15, help="Frames to keep a keypoint.")
    parser.add_argument("--min_movement", type=int, default=5, help="Pixel distance threshold for static detection.")
    parser.add_argument("--max_speed", type=int, default=20, help="Maximum speed for track drawing normalization.")
    parser.add_argument("--lightglue_weights", type=str, default="lightglue_outdoor", help="Weights for LightGlue ('lightglue_indoor' or 'lightglue_outdoor').")
    parser.add_argument("--num_keypoints", type=int, default=2000, help="Number of keypoints to detect.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize matcher and tracker.
    matcher = KorniaLightGlueMatcher(device=device, lightglue_weights=args.lightglue_weights)
    tracker = PointTracker(
        min_movement=args.min_movement,
        point_lifespan=args.keypoint_lifespan,
        max_speed=args.max_speed
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video}")
    print(f"Resolution: {orig_width}x{orig_height}, FPS: {fps:.2f}, Total frames: {total_frames}")

    # Compute inference resize dimensions.
    if orig_width > args.inference_width:
        inf_scale = args.inference_width / orig_width
        inf_width = args.inference_width
        inf_height = int(orig_height * inf_scale)
    else:
        inf_width, inf_height = orig_width, orig_height

    # Compute visualization resize dimensions.
    if orig_width > args.viz_width:
        viz_scale = args.viz_width / orig_width
        viz_width = args.viz_width
        viz_height = int(orig_height * viz_scale)
    else:
        viz_width, viz_height = orig_width, orig_height

    # We'll need a scaling factor from inference to visualization coordinates.
    scale_viz = viz_width / inf_width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (viz_width, viz_height))

    prev_frame = None
    prev_data = None  # (prev_kp, prev_desc)
    frame_count = 0

    print("Processing video with DISK+LightGlue matching and tracking...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

        # Resize frame for inference and visualization.
        inference_frame = cv2.resize(frame, (inf_width, inf_height))
        viz_frame = cv2.resize(frame, (viz_width, viz_height))

        if prev_frame is None:
            prev_frame = inference_frame.copy()
            prev_data = matcher.detect(prev_frame)
            continue

        curr_data = matcher.detect(inference_frame, args.num_keypoints)
        # Convert keypoints (which are in inference space) to numpy arrays.
        prev_kpts = prev_data[0].cpu().numpy()
        curr_kpts = curr_data[0].cpu().numpy()
        matches = matcher.match(prev_data, curr_data, prev_frame, inference_frame)
        tracker.update(prev_kpts, curr_kpts, matches)
        tracks = tracker.get_tracks(min_length=3)
        # Scale tracks from inference to visualization coordinate system.
        scaled_tracks = []
        for pts, static_frames in tracks:
            scaled_pts = [(pt[0] * scale_viz, pt[1] * scale_viz) for pt in pts]
            scaled_tracks.append((scaled_pts, static_frames))
        tracker.draw_tracks(viz_frame, scaled_tracks)

        cv2.putText(viz_frame, f"Tracks: {len(tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out.write(viz_frame)
        cv2.imshow("Kornia DISK+LightGlue Tracking", viz_frame)

        prev_frame = inference_frame.copy()
        prev_data = curr_data

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
