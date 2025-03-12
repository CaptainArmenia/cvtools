import cv2
import numpy as np
import torch
import torchvision.models.optical_flow as of
import torchvision.transforms as T
import os
import argparse
from collections import deque
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random

# Fix potential warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optical flow visualization with motion trails')
    parser.add_argument('--input', type=str, default='/home/andy/datasets/videos/industria/windmills.mp4',
                        help='Path to input video')
    parser.add_argument('--output', type=str, default='flow_overlay.mp4',
                        help='Path to output video')
    parser.add_argument('--inference_width', type=int, default=1024,
                        help='Width for inference (0=half of original size)')
    parser.add_argument('--visualization_width', type=int, default=1200, 
                        help='Width for visualization (0=original size)')
    parser.add_argument('--min_magnitude', type=float, default=1.0,
                        help='Minimum flow magnitude to visualize')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Overlay opacity (0-1)')
    parser.add_argument('--max_circles', type=int, default=500,
                        help='Maximum number of circles to spawn')
    parser.add_argument('--spawn_rate', type=float, default=0.9,
                        help='Probability of spawning a new circle each frame')
    parser.add_argument('--fade_rate', type=float, default=0.05,
                        help='Rate at which static circles fade (0-1)')
    parser.add_argument('--static_threshold', type=float, default=1.0,
                        help='Flow magnitude below which a circle is considered static')
    parser.add_argument('--circle_radius', type=int, default=30,
                        help='Radius of circles to track with flow')
    parser.add_argument('--exclusion_radius', type=int, default=3,
                        help='Radius of exclusion zone where no new trails can spawn')
    parser.add_argument('--trail_length', type=int, default=20,
                        help='Length of motion trail (number of points)')
    parser.add_argument('--trail_thickness', type=int, default=3,
                        help='Thickness of motion trail lines')
    parser.add_argument('--flow_history_frames', type=int, default=5,
                        help='Number of frames to keep in flow history for spawning')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug visualizations (circles, spawn zones, etc.)')
    parser.add_argument('--record', action='store_true',
                        help='Record output video')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS for output video')
    parser.add_argument('--trail_fade_steps', type=int, default=3,
                        help='Number of steps for trail to gradually appear (1=immediate, higher=more gradual)')
    parser.add_argument('--max_trail_age', type=int, default=25,
                        help='Maximum frames a trail can exist before removal')
    parser.add_argument('--invisible_frames', type=int, default=10,
                        help='Number of frames a trail remains invisible before starting to appear')
    return parser.parse_args()

def calculate_circle_flow(flow, circle_pos, radius, min_magnitude=0.0, top_n=5):
    """
    Calculate average flow under a circle using only the top N highest magnitude flow values.
    This makes the circles more responsive to the strongest motion in their area.
    """
    x, y = circle_pos
    h, w = flow.shape[:2]
    
    # Create a circular mask
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
    circle_mask = dist_from_center <= radius
    
    # Calculate flow magnitude for all points
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Apply minimum magnitude threshold
    magnitude_mask = flow_magnitude >= min_magnitude
    
    # Combine masks to get valid flow points within circle
    combined_mask = circle_mask & magnitude_mask
    
    if np.sum(combined_mask) > 0:
        # Get all valid flow magnitudes and their corresponding flow vectors
        valid_magnitudes = flow_magnitude[combined_mask]
        valid_flow_x = flow[..., 0][combined_mask]
        valid_flow_y = flow[..., 1][combined_mask]
        
        # If we have fewer than top_n points, use all of them
        n_points = min(top_n, len(valid_magnitudes))
        
        if n_points > 0:
            # Find indices of top N highest magnitude points
            top_indices = np.argsort(valid_magnitudes)[-n_points:]
            
            # Calculate average flow from top N points
            avg_flow_x = np.mean(valid_flow_x[top_indices])
            avg_flow_y = np.mean(valid_flow_y[top_indices])
            avg_magnitude = np.mean(valid_magnitudes[top_indices])
            
            return avg_flow_x, avg_flow_y, avg_magnitude
    
    # Fallback if no points above threshold or not enough points
    if np.sum(circle_mask) > 0:
        # Use all points in circle with reduced weight
        avg_flow_x = np.mean(flow[..., 0][circle_mask])
        avg_flow_y = np.mean(flow[..., 1][circle_mask])
        avg_mag = np.mean(flow_magnitude[circle_mask])
        return avg_flow_x * 0.5, avg_flow_y * 0.5, avg_mag  # Reduce movement when below threshold
    
    return 0, 0, 0

def find_spawn_locations(flow, historical_flow_mask, min_magnitude, existing_circles, circle_radius, exclusion_radius, n_candidates=50, min_distance=40):
    """
    Find potential locations to spawn new circles with high flow magnitude.
    Uses both current flow and historical flow data for spawning decisions.
    
    Args:
        flow: Current optical flow
        historical_flow_mask: Binary mask where flow was above threshold in past frames
        min_magnitude: Minimum flow magnitude to consider
        existing_circles: List of existing circles to avoid overlap
        circle_radius: Radius of circles to track flow
        exclusion_radius: Radius around existing circles where new trails cannot spawn
        n_candidates: Maximum number of candidate positions to consider
        min_distance: Minimum distance between new spawn locations
    """
    # Calculate flow magnitude for current frame
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Create a binary mask of points above threshold in current frame
    current_flow_mask = flow_magnitude > min_magnitude
    
    # Combine with historical mask to get all potential spawn areas
    # This includes areas with current high flow OR recent high flow
    combined_mask = np.logical_or(current_flow_mask, historical_flow_mask)
    
    if np.sum(combined_mask) == 0:
        return []  # No points with high enough flow anywhere
    
    # Get coordinates where flow is/was above threshold
    y_coords, x_coords = np.where(combined_mask)
    
    # If there are too many candidates, randomly sample some
    if len(y_coords) > n_candidates:
        indices = np.random.choice(len(y_coords), n_candidates, replace=False)
        y_coords = y_coords[indices]
        x_coords = x_coords[indices]
    
    # For prioritization, get current flow magnitudes at all candidate points
    mags = [flow_magnitude[y, x] if current_flow_mask[y, x] else min_magnitude * 0.5 
            for y, x in zip(y_coords, x_coords)]
    
    # Sort by flow magnitude (highest first)
    sorted_indices = np.argsort(mags)[::-1]
    
    locations = []
    min_spawn_distance = exclusion_radius  # Use exclusion radius instead of circle_radius
    
    for idx in sorted_indices:
        x, y = x_coords[idx], y_coords[idx]
        
        # Check if this location is far enough from existing spawn locations
        if all(np.sqrt((x - ex)**2 + (y - ey)**2) >= min_distance for ex, ey in locations):
            # Check if this location is far enough from existing circles
            if all(np.sqrt((x - cx)**2 + (y - cy)**2) >= min_spawn_distance 
                  for cx, cy, _, _, _, _ in existing_circles):  # Added an extra _ for age
                locations.append((x, y))
                
                # Limit the number of spawn locations
                if len(locations) >= 3:  # Maximum number of new circles per frame
                    break
    
    return locations

def generate_random_color():
    """Generate a bright random color."""
    h = random.random()  # Random hue
    s = 0.7 + random.random() * 0.3  # High saturation
    v = 0.8 + random.random() * 0.2  # High value
    
    # Convert HSV to RGB
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV

def update_circles(circles, flow, radius, min_magnitude, static_threshold, fade_rate, trail_length=10, max_age=30):
    """Update circle positions based on flow and fade out static circles."""
    updated_circles = []
    
    for x, y, alpha, color, history, age in circles:
        # Calculate flow at circle position
        avg_flow_x, avg_flow_y, avg_magnitude = calculate_circle_flow(flow, (int(x), int(y)), radius, min_magnitude)
        
        # Update position
        new_x = x + avg_flow_x
        new_y = y + avg_flow_y
        
        # Keep circles within frame bounds
        h, w = flow.shape[:2]
        new_x = max(radius, min(w - radius, new_x))
        new_y = max(radius, min(h - radius, new_y))
        
        # Update history
        new_history = history.copy()
        new_history.append((new_x, new_y))
        
        # Keep only the last N positions
        if len(new_history) > trail_length:
            new_history = new_history[-trail_length:]
        
        # Update alpha based on motion
        new_alpha = alpha
        if avg_magnitude < static_threshold:
            # Circle is static, start fading
            new_alpha = max(0, alpha - fade_rate)
        
        # Increment age counter
        new_age = age + 1
        
        # Keep the circle if it's still visible and hasn't exceeded max age
        if new_alpha > 0 and new_age <= max_age:
            updated_circles.append((new_x, new_y, new_alpha, color, new_history, new_age))
    
    return updated_circles

def draw_motion_trail(img, history, color, alpha, thickness=2, min_segments=10):
    """Draw a motion trail with transparency on the image."""
    # Only draw if we have enough history points
    if len(history) < min_segments:
        return img
    
    # Draw trail with varying thickness and alpha
    for i in range(len(history) - 1):
        # Calculate segment alpha (gets more opaque toward recent positions)
        segment_alpha = alpha * ((i + 1) / len(history))
        
        pt1 = tuple(map(int, history[i]))
        pt2 = tuple(map(int, history[i+1]))
        
        # Adjust color based on segment alpha
        segment_color = tuple([int(c * segment_alpha) for c in color])
        
        # Draw line segment
        cv2.line(img, pt1, pt2, segment_color, thickness)
    
    return img

def main():
    args = parse_args()
    circle_radius = args.circle_radius
    exclusion_radius = args.exclusion_radius if hasattr(args, 'exclusion_radius') else (circle_radius * 2)
    trail_length = args.trail_length if hasattr(args, 'trail_length') else 10
    trail_thickness = args.trail_thickness if hasattr(args, 'trail_thickness') else 2
    flow_history_frames = args.flow_history_frames if hasattr(args, 'flow_history_frames') else 30
    debug_mode = args.debug if hasattr(args, 'debug') else False
    trail_fade_steps = args.trail_fade_steps if hasattr(args, 'trail_fade_steps') else 3
    max_trail_age = args.max_trail_age if hasattr(args, 'max_trail_age') else 30
    invisible_frames = args.invisible_frames if hasattr(args, 'invisible_frames') else 10

    # Open video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input}")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Resize frame for visualization
    if args.visualization_width > 0:
        h, w = frame.shape[:2]
        scale = args.visualization_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    # Set up video writer if recording is enabled
    video_writer = None
    if args.record:
        # Get output size
        h, w = frame.shape[:2]
        fps = args.fps
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for mp4 format
        print(f"Recording output video to {args.output} ({w}x{h} @ {fps}fps)")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        
        # Check if video writer was successfully created
        if not video_writer.isOpened():
            print("Error: Could not create output video file")
            return
    
    # Create optical flow model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = of.raft_large(pretrained=True, progress=False).to(device).eval()
    
    # Initialize variables
    prev_frame = frame.copy()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize circles list - each entry is (x, y, alpha, color, history, age)
    circles = []
    
    # Initialize flow history queue
    flow_history = deque(maxlen=flow_history_frames)
    
    frame_count = 0  # To limit spawn rate
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Resize frame for visualization
        if args.visualization_width > 0:
            h, w = frame.shape[:2]
            scale = args.visualization_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        
        # Compute optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for inference if needed
        if args.inference_width > 0:
            h, w = frame.shape[:2]
            scale = args.inference_width / w
            small_prev = cv2.resize(prev_frame, (args.inference_width, int(h * scale)))
            small_curr = cv2.resize(frame, (args.inference_width, int(h * scale)))
            
            # Convert to PyTorch tensors
            img1 = torch.from_numpy(small_prev).permute(2, 0, 1).float().to(device)
            img2 = torch.from_numpy(small_curr).permute(2, 0, 1).float().to(device)
            
            # Normalize
            img1 = img1 / 255.0
            img2 = img2 / 255.0
            
            # Get flow
            with torch.no_grad():
                flow_output = model(img1.unsqueeze(0), img2.unsqueeze(0))
                
            # Get flow and upscale to original size
            if isinstance(flow_output, (list, tuple)):
                flow_tensor = flow_output[-1]
            else:
                flow_tensor = flow_output
            
            flow = flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            flow = cv2.resize(flow, (frame.shape[1], frame.shape[0]))
            flow_scale = frame.shape[1] / args.inference_width
            flow = flow * flow_scale
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate flow magnitude mask for history
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_mask = flow_magnitude > (args.min_magnitude * 1.5)  # Use slightly higher threshold for history
        
        # Add to flow history
        flow_history.append(flow_mask)
        
        # Create combined historical flow mask
        if len(flow_history) > 0:
            historical_flow_mask = np.zeros_like(flow_mask, dtype=bool)
            for past_mask in flow_history:
                historical_flow_mask = np.logical_or(historical_flow_mask, past_mask)
        else:
            historical_flow_mask = flow_mask  # First frame
        
        # Update circle positions and fade out static circles
        circles = update_circles(circles, flow, circle_radius, 
                                args.min_magnitude, args.static_threshold, 
                                args.fade_rate, trail_length, max_trail_age)
        
        # Spawn new circles if below maximum and random chance succeeds
        if len(circles) < args.max_circles and random.random() < args.spawn_rate:
            spawn_locations = find_spawn_locations(flow, historical_flow_mask, 
                                                 args.min_magnitude, circles, 
                                                 circle_radius, exclusion_radius)
            
            for spawn_x, spawn_y in spawn_locations:
                if len(circles) < args.max_circles:
                    # Generate a random color for the new circle
                    random_color = generate_random_color()
                    # Initialize with just the starting point in history and age=0
                    circles.append((spawn_x, spawn_y, 1.0, random_color, [(spawn_x, spawn_y)], 0))
        
        # Draw motion trails on the frame
        display_frame = frame.copy()
        
        # Create a transparent overlay for all trails
        overlay = np.zeros_like(display_frame)
        
        # If in debug mode, visualize the spawning zones
        if debug_mode:
            # Draw current flow mask in blue (semi-transparent)
            current_mask = flow_magnitude > args.min_magnitude
            display_frame[current_mask] = display_frame[current_mask] * 0.7 + np.array([128, 0, 0], dtype=np.uint8) * 0.3
            
            # Draw historical flow mask in green (very subtle)
            if historical_flow_mask is not None:
                display_frame[historical_flow_mask & ~current_mask] = \
                    display_frame[historical_flow_mask & ~current_mask] * 0.9 + np.array([0, 64, 0], dtype=np.uint8) * 0.1
        
        # Draw all trails on the overlay
        for x, y, alpha, color, history, age in circles:
            # Determine if this is an incomplete trail
            incomplete_trail = len(history) < trail_length
            history_len = len(history)
            
            # Skip drawing if trail is in invisible period
            if age < invisible_frames:
                # In debug mode, still show some indication
                if debug_mode:
                    # Draw just a dot at the current position
                    center_size = max(1, trail_thickness // 3)
                    cv2.circle(overlay, (int(x), int(y)), center_size, color, -1)
                    cv2.putText(overlay, f"Age: {age}/{invisible_frames}", 
                              (int(x) + circle_radius, int(y)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                continue
            
            # Calculate how much of the trail to show based on age, but only after invisibility period
            # Apply gradual appearance logic
            if trail_fade_steps > 1 and history_len > 1:
                # Calculate which appearance step we're in (after invisibility period)
                appearance_age = age - invisible_frames
                current_step = min(trail_fade_steps, appearance_age + 1)
                
                # Calculate visibility percentage (0.0 to 1.0)
                visibility = current_step / trail_fade_steps
                
                # Calculate the number of segments to show in this step - from oldest to newest
                num_segments = max(2, int(history_len * visibility))
                
                # Get segments starting from the oldest (beginning of history)
                # This makes the trail appear from tail to head
                draw_history = history[:num_segments]
                
                # In debug mode, print info about the steps periodically
                if debug_mode and age % 15 == 0 and age > invisible_frames:
                    print(f"Age: {age}, Appearance step: {current_step}/{trail_fade_steps}, " +
                          f"Showing {len(draw_history)}/{history_len} segments " +
                          f"(from oldest to newest)")
            else:
                # No gradual appearance, show all available history after invisibility period
                draw_history = history
            
            # Calculate base alpha based on age - start fading after becoming fully visible
            max_visible_age = invisible_frames + trail_fade_steps
            if age > max_visible_age:
                # Apply fading as the trail ages beyond visibility stages
                fade_progress = (age - max_visible_age) / (max_trail_age - max_visible_age)
                fade_factor = 1.0 - min(fade_progress, 1.0)
                base_alpha = alpha * fade_factor
            else:
                # Full alpha during appearance phase
                base_alpha = alpha
            
            # Draw the trail with appropriate styling
            if len(draw_history) > 1:  # Need at least 2 points to draw a line
                for i in range(len(draw_history) - 1):
                    # Calculate segment alpha (gets more opaque toward recent positions)
                    segment_alpha = base_alpha * ((i + 1) / len(draw_history))
                    
                    pt1 = tuple(map(int, draw_history[i]))
                    pt2 = tuple(map(int, draw_history[i+1]))
                    
                    # Adjust color based on segment alpha
                    segment_color = tuple([int(c * segment_alpha) for c in color])
                    
                    # If incomplete trail in debug mode, use dashed line 
                    if incomplete_trail and debug_mode:
                        # Dash pattern for incomplete trails
                        if i % 2 == 0:  # Skip every other segment for dashed effect
                            # Use thinner line for incomplete trails
                            thickness = max(1, trail_thickness // 2)
                            cv2.line(overlay, pt1, pt2, segment_color, thickness)
                    else:
                        # Normal drawing for complete trails
                        cv2.line(overlay, pt1, pt2, segment_color, trail_thickness)
            
            # If debug mode is enabled, show the circle at the current position
            if debug_mode:
                # Draw circle showing flow measurement area
                cv2.circle(overlay, (int(x), int(y)), circle_radius, color, 1)
                
                # Also draw exclusion zone with dashed line
                # Create dashed pattern for exclusion zone
                for i in range(0, 360, 30):  # Draw segments at 30 degree intervals
                    start_angle = i
                    end_angle = i + 15
                    cv2.ellipse(overlay, (int(x), int(y)), 
                               (exclusion_radius, exclusion_radius), 
                               0, start_angle, end_angle, color, 1)
                
                # Draw dot at center point
                center_size = max(1, trail_thickness // 2)
                cv2.circle(overlay, (int(x), int(y)), center_size, color, -1)
                
                # Show trail age and phase information
                if age < invisible_frames:
                    phase_text = f"Inv {age}/{invisible_frames}"
                elif age < max_visible_age:
                    appear_step = age - invisible_frames + 1
                    phase_text = f"App {appear_step}/{trail_fade_steps}"
                else:
                    fade_left = max_trail_age - age
                    phase_text = f"Fade {fade_left}"
                
                cv2.putText(overlay, phase_text, (int(x) + circle_radius, int(y) + circle_radius),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Blend overlay with original frame
        display_frame = cv2.addWeighted(display_frame, 1.0, overlay, 0.7, 0)
        
        # Add extra debug info if enabled
        if debug_mode:
            # Show threshold values
            cv2.putText(display_frame, f"Threshold: {args.min_magnitude:.2f}", 
                      (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            # Show circle radius
            cv2.putText(display_frame, f"Flow R: {circle_radius}, Exclusion R: {exclusion_radius}", 
                      (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            # Show spawn rate
            cv2.putText(display_frame, f"Spawn: {args.spawn_rate:.2f}", 
                      (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            # Show trail length requirement
            cv2.putText(display_frame, f"Trail Length: {trail_length}", 
                      (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            # Show top N flow values used
            cv2.putText(display_frame, f"Using top 5 flow values", 
                      (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            # Show trail parameters
            cv2.putText(display_frame, f"Trail appear steps: {trail_fade_steps}", 
                      (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(display_frame, f"Max trail age: {max_trail_age}", 
                      (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(display_frame, f"Invisible period: {invisible_frames} frames", 
                      (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(display_frame, f"Total lifecycle: {invisible_frames}+{trail_fade_steps}+fade", 
                      (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
            # Display stats
            cv2.putText(display_frame, f"Trails: {len(circles)}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame to output video if recording is enabled
        if args.record and video_writer is not None:
            video_writer.write(display_frame)
            
            # Show recording indicator
            if debug_mode or frame_count % 30 == 0:  # Update every second at 30fps
                print(f"Recording frame {frame_count}")
            
            # Optional: add recording indicator to the displayed frame
            cv2.putText(display_frame, "REC", (w - 70, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Flow Visualization', display_frame)
        
        # Update previous frame
        prev_frame = frame.copy()
        prev_gray = gray.copy()
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Recording completed: {args.output}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()