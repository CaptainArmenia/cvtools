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
import matplotlib.cm as cm  # Add colormap import
from scipy.ndimage import gaussian_filter
import random
import colorsys  # Add for HSV -> RGB conversion in flow line coloring

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
    parser.add_argument('--spawn_rate', type=float, default=1.0,
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
    parser.add_argument('--max_speed', type=float, default=10.0,
                        help='Maximum speed for colormap (0=auto, will use min_magnitude*5)')
    parser.add_argument('--color_scheme', type=str, default='rgb',
                        choices=['viridis', 'heat', 'neon', 'plasma', 'rgb', 'fire'],
                        help='Color scheme for speed visualization')
    parser.add_argument('--info', action='store_true',
                        help='Display industrial monitoring information about scene dynamics')
    # Add new command-line arguments for flow lines
    parser.add_argument('--show_flow_lines', action='store_true',
                        help='Show optical flow lines in active areas')
    parser.add_argument('--line_density', type=int, default=16,
                        help='Density of flow lines (higher = fewer lines)')
    parser.add_argument('--line_thickness', type=int, default=1,
                        help='Thickness of flow lines')
    return parser.parse_args()

def flow_to_color(magnitude, min_val=0, max_val=5, scheme='heat'):
    """
    Map flow magnitude to a color using the selected colormap.
    
    Args:
        magnitude: Flow magnitude
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
        scheme: Color scheme to use ('viridis', 'heat', 'neon', 'rgb', 'plasma', 'fire')
    
    Returns:
        BGR color tuple for OpenCV
    """
    # Normalize magnitude between 0 and 1
    normalized = min(1.0, max(0.0, (magnitude - min_val) / (max_val - min_val)))
    
    if scheme == 'viridis':
        # Original viridis colormap
        rgb_color = cm.viridis(normalized)[:3]
    elif scheme == 'plasma':
        # Plasma colormap - more purple/pink/yellow
        rgb_color = cm.plasma(normalized)[:3]
    elif scheme == 'heat':
        # Heat colormap - blue to red through white
        # Dark blue (slow) -> cyan -> white -> yellow -> red (fast)
        if normalized < 0.25:
            # Blue to cyan
            t = normalized / 0.25
            r, g, b = 0, t, 1
        elif normalized < 0.5:
            # Cyan to white
            t = (normalized - 0.25) / 0.25
            r, g, b = t, 1, 1
        elif normalized < 0.75:
            # White to yellow
            t = (normalized - 0.5) / 0.25
            r, g, b = 1, 1, 1 - t
        else:
            # Yellow to red
            t = (normalized - 0.75) / 0.25
            r, g, b = 1, 1 - t, 0
    elif scheme == 'neon':
        # Neon glow effect - purple to cyan to white
        if normalized < 0.5:
            # Purple to cyan
            t = normalized * 2
            r, g, b = 0.5 + 0.5 * t, t, 1
        else:
            # Cyan to white
            t = (normalized - 0.5) * 2
            r, g, b = 1, 1, 1
    elif scheme == 'rgb':
        # Full RGB spectrum in reverse (red for slow, violet for fast)
        if normalized < 0.2:
            # Red to orange (slowest)
            t = normalized / 0.2
            r, g, b = 1, t * 0.5, 0
        elif normalized < 0.4:
            # Orange to yellow
            t = (normalized - 0.2) / 0.2
            r, g, b = 1, 0.5 + t * 0.5, 0
        elif normalized < 0.6:
            # Yellow to green
            t = (normalized - 0.4) / 0.2
            r, g, b = 1 - t, 1, 0
        elif normalized < 0.8:
            # Green to blue
            t = (normalized - 0.6) / 0.2
            r, g, b = 0, 1 - t, t
        else:
            # Blue to violet (fastest)
            t = (normalized - 0.8) / 0.2
            r, g, b = t * 0.5, 0, 1
    elif scheme == 'fire':
        # Fire colormap - black to red to yellow to white
        if normalized < 0.25:
            # Black to dark red
            t = normalized * 4
            r, g, b = t, 0, 0
        elif normalized < 0.5:
            # Dark red to red
            t = (normalized - 0.25) * 4
            r, g, b = 0.25 + 0.75 * t, 0, 0
        elif normalized < 0.75:
            # Red to yellow
            t = (normalized - 0.5) * 4
            r, g, b = 1, t, 0
        else:
            # Yellow to white
            t = (normalized - 0.75) * 4
            r, g, b = 1, 1, t
    else:
        # Default to viridis if scheme is not recognized
        rgb_color = cm.viridis(normalized)[:3]
        return (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))
    
    # Convert to BGR for OpenCV (0-255 range)
    bgr_color = (int(b * 255), int(g * 255), int(r * 255))
    return bgr_color

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
                  for cx, cy, _, _, _, _, _ in existing_circles):  # Updated to unpack 7 values including speed and selected color scheme
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

def calculate_trail_length(points):
    """
    Calculate the total length of a trail in pixels.
    
    Args:
        points: List of (x, y) coordinates representing the trail path
        
    Returns:
        Total length in pixels
    """
    if len(points) < 2:
        return 0    
    
    # Sum the Euclidean distances between consecutive points
    total_length = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += segment_length
    
    return total_length

def update_circles(circles, flow, radius, min_magnitude, static_threshold, fade_rate, trail_length=10, max_age=30, max_speed=None, color_scheme='heat', invisible_frames=10):
    """Update circle positions based on flow and fade out static circles."""
    updated_circles = []
    
    # Set default max_speed if not provided
    if max_speed is None:
        max_speed = min_magnitude * 5
    
    for x, y, alpha, color, history, age, speed in circles:
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
        
        # Calculate speed only once when trail becomes visible (transitions from invisible to visible)
        new_speed = speed
        if age == invisible_frames - 1:  # About to become visible in the next frame
            # Calculate the total trail length in pixels
            total_length = calculate_trail_length(new_history)
            
            # Calculate speed as pixels per frame
            if age > 0:
                new_speed = total_length / age
            else:
                new_speed = min_magnitude * 2  # Fallback if age is 0
            
            # Ensure the speed is within reasonable bounds
            new_speed = max(min_magnitude, min(new_speed, max_speed))
        
        # Update color based on the fixed speed (only changes when speed is first calculated)
        new_color = flow_to_color(new_speed, min_val=min_magnitude, max_val=max_speed, scheme=color_scheme)    
        
        # Update alpha based on motion
        new_alpha = alpha
        if avg_magnitude < static_threshold:
            # Circle is static, start fading
            new_alpha = max(0, alpha - fade_rate)
        
        # Increment age counter
        new_age = age + 1
        
        # Keep the circle if it's still visible and hasn't exceeded max age
        if new_alpha > 0 and new_age <= max_age:
            updated_circles.append((new_x, new_y, new_alpha, new_color, new_history, new_age, new_speed))
    
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

def calculate_scene_metrics(flow, min_magnitude=1.0):
    """
    Calculate quantitative metrics about the scene dynamics.
    
    Args:
        flow: Optical flow matrix
        min_magnitude: Minimum flow magnitude to consider
    
    Returns:
        Dictionary of metrics
    """
    # Calculate flow magnitude
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Calculate basic statistics
    active_pixels = flow_magnitude > min_magnitude
    active_pixel_count = np.sum(active_pixels)
    
    # Calculate metrics only for active pixels
    if active_pixel_count > 0:
        mean_magnitude = np.mean(flow_magnitude[active_pixels])
        max_magnitude = np.max(flow_magnitude[active_pixels])
        median_magnitude = np.median(flow_magnitude[active_pixels])
        std_magnitude = np.std(flow_magnitude[active_pixels])
        
        # Calculate flow direction (in degrees, 0-360)
        flow_direction = np.zeros_like(flow_magnitude)
        valid_flow = (flow_magnitude > min_magnitude)
        if np.sum(valid_flow) > 0:
            # Calculate direction in degrees (0-360) for valid flow vectors
            flow_direction[valid_flow] = (np.degrees(np.arctan2(
                flow[..., 1][valid_flow], flow[..., 0][valid_flow])) + 360) % 360
            
            # Calculate directional statistics
            direction_histogram = np.zeros(8)  # 8 directional bins (N, NE, E, SE, S, SW, W, NW)
            for i in range(8):
                bin_min = i * 45
                bin_max = (i + 1) * 45
                bin_mask = (flow_direction >= bin_min) & (flow_direction < bin_max) & valid_flow
                direction_histogram[i] = np.sum(bin_mask)
                
            # Normalize histogram
            if np.sum(direction_histogram) > 0:
                direction_histogram = direction_histogram / np.sum(direction_histogram)
            
            # Find primary direction
            primary_dir_idx = np.argmax(direction_histogram)
            primary_dir = primary_dir_idx * 45 + 22.5  # Center of bin
            primary_dir_strength = direction_histogram[primary_dir_idx]
        else:
            direction_histogram = np.zeros(8)
            primary_dir = 0
            primary_dir_strength = 0
        
        # Calculate scene coverage
        scene_coverage = active_pixel_count / flow_magnitude.size
        
        return {
            'active_pixels': active_pixel_count,
            'scene_coverage': scene_coverage,
            'mean_magnitude': mean_magnitude,
            'median_magnitude': median_magnitude,
            'max_magnitude': max_magnitude,
            'std_magnitude': std_magnitude,
            'direction_histogram': direction_histogram,
            'primary_direction': primary_dir,
            'primary_direction_strength': primary_dir_strength
        }
    else:
        # Return default values when no active pixels
        return {
            'active_pixels': 0,
            'scene_coverage': 0,
            'mean_magnitude': 0,
            'median_magnitude': 0,
            'max_magnitude': 0, 
            'std_magnitude': 0,
            'direction_histogram': np.zeros(8),
            'primary_direction': 0,
            'primary_direction_strength': 0
        }

def draw_info_overlay(frame, metrics, history_buffer=None):
    """
    Draw industrial monitoring information overlay on the frame.
    
    Args:
        frame: Input frame to draw on
        metrics: Dictionary of scene metrics
        history_buffer: Buffer of historical metrics for trends
    
    Returns:
        Frame with info overlay
    """
    h, w = frame.shape[:2]
    
    # Create semi-transparent dark background for readability
    info_bg = frame.copy()
    cv2.rectangle(info_bg, (w - 400, 20), (w - 20, 280), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.8, info_bg, 0.2, 0)
    
    # Use an even brighter, more vivid green with better contrast
    title_color = (50, 255, 120)  # Vibrant lime green in BGR
    
    # Draw title text with bolder weight (thickness=2) instead of shadow
    title_text = "INDUSTRIAL MONITORING DATA"
    title_pos = (int(w - 380), 40)
    
    # Draw title with thicker line for better visibility
    cv2.putText(frame, title_text, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)
    
    # Display motion metrics with pure white
    text_color = (255, 255, 255)  # Pure white for better contrast
    line_height = 30
    y_pos = 70
    
    # Motion magnitude metrics - Use higher contrast white text
    cv2.putText(frame, f"Active Motion Areas: {metrics['active_pixels']:,} pixels", 
               (int(w - 380), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Scene Coverage: {metrics['scene_coverage']*100:.1f}%", 
               (int(w - 380), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Mean Motion Rate: {metrics['mean_magnitude']:.2f} px/frame", 
               (int(w - 380), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Max Motion Rate: {metrics['max_magnitude']:.2f} px/frame", 
               (int(w - 380), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Motion Variation: {metrics['std_magnitude']:.2f} px/frame", 
               (int(w - 380), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height * 1.5
    
    # Draw direction information with better contrast
    direction_text = f"Primary Direction: {metrics['primary_direction']:.1f}° ({get_direction_name(metrics['primary_direction'])})"
    direction_pos = (int(w - 380), int(y_pos))
    cv2.putText(frame, direction_text, direction_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Direction Confidence: {metrics['primary_direction_strength']*100:.1f}%", 
               (int(w - 380), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height * 1.5
    
    # Draw direction histogram in the bottom left corner
    histogram = metrics['direction_histogram']
    if np.sum(histogram) > 0:
        # Draw histogram visualization - Move to bottom left
        hist_width, hist_height = 220, 100
        margin = 20  # Margin from the edge of the frame
        hist_x, hist_y = int(margin), int(h - hist_height - margin - 25)  # 25 pixels extra for labels below
        
        # Create semi-transparent dark background for the histogram
        hist_bg = frame.copy()
        cv2.rectangle(hist_bg, (hist_x - 10, hist_y - 30), (hist_x + hist_width + 10, hist_y + hist_height + 25), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, hist_bg, 0.2, 0)
        
        # Add histogram title with same approach - bold text without shadow
        histogram_title = "Motion Direction Distribution"
        hist_title_pos = (hist_x, hist_y - 10)
        
        # Draw histogram title with thicker line
        cv2.putText(frame, histogram_title, hist_title_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, title_color, 2)
        
        # Draw histogram background
        cv2.rectangle(frame, (hist_x, hist_y), (hist_x + hist_width, hist_y + hist_height), (30, 30, 30), -1)
        cv2.rectangle(frame, (hist_x, hist_y), (hist_x + hist_width, hist_y + hist_height), (100, 100, 100), 1)
        
        # Draw histogram bars
        bar_width = hist_width // 8
        for i in range(8):
            bar_height = int(histogram[i] * hist_height)
            if bar_height > 0:
                # Fix: Ensure all coordinates are integers
                cv2.rectangle(frame, 
                             (hist_x + i * bar_width, hist_y + hist_height - bar_height),
                             (hist_x + (i + 1) * bar_width - 2, hist_y + hist_height),
                             (50, 180, 250), -1)
        
        # Draw direction labels with better contrast
        direction_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        for i, label in enumerate(direction_labels):
            # Fix: Ensure all coordinates are integers
            label_pos = (int(hist_x + i * bar_width + bar_width//2 - 5), int(hist_y + hist_height + 15))
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    return frame

def get_direction_name(angle):
    """Convert angle in degrees to cardinal direction name."""
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((angle + 22.5) % 360) / 45)
    return directions[idx]

def draw_flow_lines(frame, flow, magnitude_threshold=1.0, step=16, thickness=1, color=None, scale=1.0):
    """
    Draw optical flow lines on the frame at regular intervals.
    
    Args:
        frame: Input frame to draw on
        flow: Optical flow matrix
        magnitude_threshold: Minimum flow magnitude to draw
        step: Spacing between flow lines
        thickness: Thickness of flow lines
        color: Optional fixed color for all lines (otherwise colored by direction)
        scale: Scale factor for line length
        
    Returns:
        Frame with flow lines
    """
    h, w = frame.shape[:2]
    y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    # Calculate flow magnitude
    fx, fy = flow[..., 0], flow[..., 1]
    flow_magnitude = np.sqrt(fx**2 + fy**2)
    
    # Create a copy of the frame to draw on
    vis = frame.copy()
    
    # Draw flow lines
    lines = []
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if 0 <= y < h and 0 <= x < w:
            # Get flow at this point
            flow_x = fx[y, x]
            flow_y = fy[y, x]
            magnitude = flow_magnitude[y, x]
            
            # Only draw if above threshold
            if magnitude >= magnitude_threshold:
                # Calculate line end point
                end_x = int(x + flow_x * scale)
                end_y = int(y + flow_y * scale)
                
                # Keep end point within frame bounds
                end_x = max(0, min(w-1, end_x))
                end_y = max(0, min(h-1, end_y))
                
                # Add to list of lines to draw
                lines.append(((x, y), (end_x, end_y), magnitude))
    
    # Draw all lines
    for (x1, y1), (x2, y2), magnitude in lines:
        # Choose color based on direction if no fixed color
        if color is None:
            angle = np.arctan2(y2 - y1, x2 - x1)
            hue = (np.degrees(angle) + 180) / 360.0
            rgb_color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            line_color = (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))
        else:
            line_color = color
            
        # Draw line and circle at the end
        cv2.line(vis, (x1, y1), (x2, y2), line_color, thickness)
        cv2.circle(vis, (x2, y2), thickness+1, line_color, -1)
    
    return vis

def main():
    args = parse_args()
    circle_radius = args.circle_radius
    exclusion_radius = args.exclusion_radius if hasattr(args, 'exclusion_radius') else (circle_radius * 2)
    trail_length = args.trail_length if hasattr(args, 'trail_length') else 10
    trail_thickness = args.trail_thickness if hasattr(args, 'trail_thickness') else 2
    flow_history_frames = args.flow_history_frames if hasattr(args, 'flow_history_frames') else 30
    debug_mode = args.debug if hasattr(args, 'debug') else False
    info_mode = args.info if hasattr(args, 'info') else False
    trail_fade_steps = args.trail_fade_steps if hasattr(args, 'trail_fade_steps') else 3
    max_trail_age = args.max_trail_age if hasattr(args, 'max_trail_age') else 30
    invisible_frames = args.invisible_frames if hasattr(args, 'invisible_frames') else 10
    # Calculate max speed for colormap (if not specified, use min_magnitude * 5)
    max_speed = args.max_speed if args.max_speed > 0 else (args.min_magnitude * 5)
    # Get selected color scheme
    color_scheme = args.color_scheme if hasattr(args, 'color_scheme') else 'heat'
    
    # Extract the new parameters
    show_flow_lines = args.show_flow_lines if hasattr(args, 'show_flow_lines') else False
    line_density = args.line_density if hasattr(args, 'line_density') else 16
    line_thickness = args.line_thickness if hasattr(args, 'line_thickness') else 1
    
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
    
    # Initialize circles list - each entry is now (x, y, alpha, color, history, age, speed)
    circles = []
    
    # Initialize flow history queue
    flow_history = deque(maxlen=flow_history_frames)
    frame_count = 0  # To limit spawn rate 
    
    # Initialize metrics history for trends
    metrics_history = deque(maxlen=30)  # Store up to 30 frames of metrics
    
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
                                args.fade_rate, trail_length, max_trail_age, max_speed, color_scheme, invisible_frames)
        
        # Spawn new circles if below maximum and random chance succeeds
        if len(circles) < args.max_circles and random.random() < args.spawn_rate:
            spawn_locations = find_spawn_locations(flow, historical_flow_mask, 
                                                 args.min_magnitude, circles, 
                                                 circle_radius, exclusion_radius)
            
            for spawn_x, spawn_y in spawn_locations:
                if len(circles) < args.max_circles:
                    # Calculate initial speed at spawn location
                    x_int, y_int = int(spawn_x), int(spawn_y)
                    if 0 <= x_int < flow.shape[1] and 0 <= y_int < flow.shape[0]:
                        initial_magnitude = np.sqrt(flow[y_int, x_int, 0]**2 + flow[y_int, x_int, 1]**2)
                    else:
                        initial_magnitude = args.min_magnitude * 1.5
                    
                    # Generate color based on speed using selected color scheme
                    speed_color = flow_to_color(initial_magnitude, min_val=args.min_magnitude, max_val=max_speed, scheme=color_scheme)
                    
                    # Initialize with a provisional speed (real speed will be calculated when becoming visible)
                    initial_speed = 0.0  # Placeholder value until we calculate the actual speed
                    
                    # Generate placeholder color - will be updated when speed is calculated
                    speed_color = flow_to_color(args.min_magnitude * 1.5, min_val=args.min_magnitude, max_val=max_speed, scheme=color_scheme)
                    
                    # Initialize with just the starting point in history, age=0, and initial speed
                    circles.append((spawn_x, spawn_y, 1.0, speed_color, [(spawn_x, spawn_y)], 0, initial_speed))
        
        # Draw motion trails on the frame
        display_frame = frame.copy()
        
        # If flow lines visualization is enabled, draw them
        if show_flow_lines:
            # Draw flow lines on active areas (use same min_magnitude threshold as used for trails)
            display_frame = draw_flow_lines(
                display_frame, 
                flow, 
                magnitude_threshold=args.min_magnitude,
                step=line_density,
                thickness=line_thickness
            )
        
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
        for x, y, alpha, color, history, age, speed in circles:  # Update to unpack 7 values including speed
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
        
        # Draw industrial monitoring info if enabled
        if info_mode:
            scene_metrics = calculate_scene_metrics(flow, min_magnitude=args.min_magnitude)
            metrics_history.append(scene_metrics)
            display_frame = draw_info_overlay(display_frame, scene_metrics, metrics_history)
        
        # Add extra debug info if enabled
        if debug_mode:
            # Use higher contrast green for debug text
            debug_text_color = (0, 255, 0)  # Bright green for debug text
            
            # Show threshold values
            cv2.putText(display_frame, f"Threshold: {args.min_magnitude:.2f}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            
            # Show circle radius
            cv2.putText(display_frame, f"Flow R: {circle_radius}, Exclusion R: {exclusion_radius}", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            
            # Show spawn rate
            cv2.putText(display_frame, f"Spawn: {args.spawn_rate:.2f}", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            
            # Show trail length requirement
            cv2.putText(display_frame, f"Trail Length: {trail_length}", 
                       (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            
            # Show top N flow values used
            cv2.putText(display_frame, f"Using top 5 flow values", 
                       (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            
            # Show trail parameters
            cv2.putText(display_frame, f"Trail appear steps: {trail_fade_steps}", 
                       (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            cv2.putText(display_frame, f"Max trail age: {max_trail_age}", 
                       (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            cv2.putText(display_frame, f"Invisible period: {invisible_frames} frames", 
                       (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            cv2.putText(display_frame, f"Total lifecycle: {invisible_frames}+{trail_fade_steps}+fade", 
                       (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            cv2.putText(display_frame, f"Speed: fixed, calculated once based on trail length", 
                       (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_text_color, 1)
            
            # Display stats
            cv2.putText(display_frame, f"Trails: {len(circles)}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add colormap reference
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, f"Speed colormap ({color_scheme}):", 
                       (w - 200, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            bar_x = w - 180
            bar_y = h - 40
            bar_width = 150
            bar_height = 10
            for i in range(bar_width):
                norm_val = i / bar_width
                color = flow_to_color(args.min_magnitude + norm_val * (max_speed - args.min_magnitude), 
                                     min_val=args.min_magnitude, max_val=max_speed, scheme=color_scheme)
                cv2.line(display_frame, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height), color, 1)
            cv2.putText(display_frame, f"{args.min_magnitude:.1f}", 
                       (bar_x, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, f"{max_speed:.1f}", 
                       (bar_x + bar_width - 20, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Write frame to output video if recording is enabled
        if args.record and video_writer is not None:
            if debug_mode or frame_count % 30 == 0:  # Update every second at 30fps
                print(f"Recording frame {frame_count}")
            # Show recording indicator
            cv2.putText(display_frame, "REC", (w - 70, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Optional: add recording indicator to the displayed frame
            video_writer.write(display_frame)
        
        # Display the frame
        cv2.imshow('Flow Visualization', display_frame)
        
        # Update previous frame
        prev_gray = gray.copy()
        prev_frame = frame.copy()
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    if video_writer is not None:
        video_writer.release()
        print(f"Recording completed: {args.output}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()