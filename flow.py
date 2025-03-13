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
import matplotlib.cm as cm

# Fix potential warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optical flow visualization with temporal fading')
    parser.add_argument('--input', type=str, default='/home/andy/datasets/videos/industria/windmills.mp4',
                        help='Path to input video')
    parser.add_argument('--output', type=str, default='flow_overlay.mp4',
                        help='Path to output video')
    parser.add_argument('--inference_width', type=int, default=512,
                        help='Width for inference (0=half of original size)')
    parser.add_argument('--visualization_width', type=int, default=1200, 
                        help='Width for visualization (0=original size)')
    parser.add_argument('--min_magnitude', type=float, default=0.5,
                        help='Minimum flow magnitude to visualize')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Overlay opacity (0-1)')
    parser.add_argument('--fade_frames', type=int, default=30,
                        help='Number of frames for gradual fading')
    parser.add_argument('--smoothing', type=float, default=3.0,
                        help='Flow smoothing factor (sigma value for Gaussian blur)')
    parser.add_argument('--line_density', type=int, default=8, 
                        help='Sample every Nth pixel for flow lines (higher = less dense)')
    parser.add_argument('--line_thickness', type=int, default=1,
                        help='Thickness of flow lines')
    parser.add_argument('--info', action='store_true',
                        help='Display industrial monitoring information about scene dynamics')
    parser.add_argument('--info_position', type=str, default='top-right',
                        choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 
                                'top-center', 'bottom-center', 'left-center', 'right-center'],
                        help='Position for the information overlay')
    parser.add_argument('--histogram_position', type=str, default='bottom-left',
                        choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 
                                'top-center', 'bottom-center', 'left-center', 'right-center'],
                        help='Position for the direction histogram')
    return parser.parse_args()

def preprocess(img_tensor):
    """Preprocess image tensor for RAFT model."""
    transforms = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
    ])
    return transforms(img_tensor)

def compute_flow(model, img1, img2, device):
    """Compute optical flow between two frames."""
    with torch.no_grad():
        # Prepare inputs
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        # Run inference
        flow_predictions = model(img1, img2)
        if isinstance(flow_predictions, list):
            flow = flow_predictions[-1][0].cpu().numpy()  # Use final prediction
        else:
            flow = flow_predictions[0].cpu().numpy()
        
        # Convert from tensor format to OpenCV format (H,W,C)
        return flow.transpose(1, 2, 0)

def smooth_flow(flow, sigma=3.0):
    """Apply Gaussian smoothing to flow field while preserving structure."""
    # Apply smoothing to each channel separately
    smoothed_flow = np.zeros_like(flow)
    smoothed_flow[:,:,0] = gaussian_filter(flow[:,:,0], sigma=sigma)
    smoothed_flow[:,:,1] = gaussian_filter(flow[:,:,1], sigma=sigma)
    return smoothed_flow

def flow_to_color(flow, min_magnitude=1.0, colormap=None):
    """
    Convert flow to a smooth color representation with proper transparency handling.
    Eliminates gray artifacts by using more vibrant colors and cleaner alpha transitions.
    """
    if colormap is None:
        colormap = generate_smooth_colormap()

    flow[abs(flow) < min_magnitude] = 0
    
    # Calculate magnitude and angle
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    magnitude = np.sqrt(fx*fx + fy*fy)
    
    # Create a stronger cutoff for low magnitudes to eliminate gray areas
    magnitude_mask = magnitude > min_magnitude
    
    # Normalize magnitude with higher contrast
    normalized_magnitude = np.clip(magnitude / 15.0, 0, 1.0)  # Adjusted divisor for more vibrant colors
    
    # Create angle visualization (0-255)
    angle = np.arctan2(fy, fx) + np.pi  # Range from 0 to 2π
    angle_normalized = (angle / (2 * np.pi) * 255).astype(np.uint8)
    
    # Apply colormap to angle with higher saturation
    flow_color = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA with alpha channel
    for i in range(3):  # RGB channels
        flow_color[:,:,i] = colormap[angle_normalized, i]
    
    # Scale color by magnitude with a higher power for more contrast
    flow_color = flow_color.astype(np.float32)
    for i in range(3):
        # Apply a power curve to increase contrast and reduce weak (grayish) colors
        intensity = normalized_magnitude ** 0.8  # Power < 1 enhances mid-tones
        flow_color[:,:,i] = np.clip(flow_color[:,:,i] * intensity, 0, 255)
    
    # Create alpha channel with sharper cutoff
    alpha_falloff = 4.0  # Higher value for sharper transition
    alpha = np.clip((magnitude - min_magnitude) * alpha_falloff, 0, 1)
    
    # Hard zero for values below threshold to prevent any color bleeding
    for i in range(3):
        flow_color[:,:,i] = np.where(magnitude_mask, flow_color[:,:,i], 0)
    
    # Set alpha channel
    flow_color[:,:,3] = (alpha * 255).astype(np.uint8)
    
    return flow_color.astype(np.uint8), magnitude_mask

def overlay_flow(frame, flow_color, alpha=0.7):
    """
    Overlay flow visualization on original frame with improved alpha blending.
    Uses premultiplied alpha to prevent color artifacts.
    """
    # Create output frame
    result = frame.copy().astype(np.float32)
    
    # Extract flow color and premultiply alpha
    flow_rgb = flow_color[:,:,:3].astype(np.float32)
    flow_alpha = (flow_color[:,:,3].astype(np.float32) / 255.0) * alpha
    
    # Create mask for areas with any flow
    valid_mask = flow_alpha > 0.01  # Small threshold to completely skip near-zero alphas
    
    # Apply fast vectorized blending only where needed
    if np.any(valid_mask):
        alpha_expanded = np.expand_dims(flow_alpha, axis=2)
        
        # Only process areas with flow (more efficient)
        result[valid_mask] = (result[valid_mask] * (1 - alpha_expanded[valid_mask]) + 
                              flow_rgb[valid_mask] * alpha_expanded[valid_mask])
    
    return np.clip(result, 0, 255).astype(np.uint8)

def generate_smooth_colormap():
    """Generate a colormap with emphasis on green, purple, and blue."""
    # Create a colormap focused on the requested colors
    # Note: Colors in BGR format for OpenCV (Blue=255,0,0; Green=0,255,0; Purple=128,0,128)
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),    # Purple
        (255, 0, 0),      # Blue (repeat for continuity)
    ]
    
    # Create a colormap with 256 entries
    cmap = np.zeros((256, 3))
    n_colors = len(colors) - 1  # Subtract 1 because last color is same as first for continuity
    
    for i in range(256):
        # Map the index to a position in the color list
        pos = (i / 255) * n_colors
        idx = int(pos)
        
        # Interpolate between the colors
        if idx < n_colors:
            alpha = pos - idx
            cmap[i] = [(1-alpha) * colors[idx][j] + alpha * colors[idx+1][j] for j in range(3)]
    
    # No need to boost saturation as we're using full intensity colors
    return cmap

def post_process_visualization(flow_color, blur_size=5):
    """Apply post-processing to flow visualization for smoother appearance."""
    # Extract RGB and alpha channels
    rgb = flow_color[:,:,:3].copy()
    alpha = flow_color[:,:,3].copy()  # Just get the channel, not slice with extra dimension
    
    # Apply slight blur to RGB channels
    smoothed_rgb = cv2.GaussianBlur(rgb, (blur_size, blur_size), 0)
    
    # Apply stronger blur to alpha channel for better edge transition
    smoothed_alpha = cv2.GaussianBlur(alpha, (blur_size*2+1, blur_size*2+1), 0)
    
    # Create result array
    result = np.zeros_like(flow_color)
    result[:,:,:3] = smoothed_rgb
    result[:,:,3] = smoothed_alpha  # Direct assignment without the :4 slice
    
    return result

class FlowLine:
    """Represents a single flow line with position, color, and age information."""
    def __init__(self, start_pos, end_pos, color, magnitude, max_age=10):
        self.start_pos = start_pos  # (x, y) tuple
        self.end_pos = end_pos      # (x, y) tuple
        self.color = color          # (B, G, R) tuple
        self.magnitude = magnitude  # Flow magnitude
        self.max_age = max_age      # Maximum age before removal
        self.age = 0                # Current age (0 = newest)
        # Calculate initial alpha based on magnitude with higher base level
        self.base_opacity = min(magnitude / 8.0, 1.0)  # Increased from 10.0 to 8.0
        
    def update_age(self):
        """Increment age of the flow line."""
        self.age += 1
        # Always return True so lines are kept until they're completely invisible
        return True
    
    def get_alpha(self, base_alpha=0.7):
        """Calculate alpha based on magnitude and age with linear fade."""
        # Simple linear decay with age, but with a higher starting point
        remaining_life = 1.0 - (self.age / self.max_age)
        # Ensure negative values don't occur
        if remaining_life <= 0:
            return 0.0
        # Apply base_alpha scaling factor
        return base_alpha * remaining_life * self.base_opacity


def generate_flow_lines(flow, min_magnitude=1.0, colormap=None, density=8, max_age=20):
    """
    Generate flow lines for visualization from optical flow field.
    Only creates lines for pixels that have motion above threshold.
    
    Args:
        flow: Optical flow field (H,W,2)
        min_magnitude: Minimum flow magnitude to visualize
        colormap: Color lookup table for direction-based coloring
        density: Sample every Nth pixel (higher = less dense)
        max_age: Maximum age for flow lines before being completely transparent
        
    Returns:
        List of FlowLine objects
    """
    if colormap is None:
        colormap = generate_smooth_colormap()
    
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    magnitude = np.sqrt(fx*fx + fy*fy)
    
    # Only consider pixels with motion above threshold
    mask = magnitude > min_magnitude
    
    # Create angle visualization for coloring
    angle = np.arctan2(fy, fx) + np.pi  # Range from 0 to 2π
    angle_normalized = (angle / (2 * np.pi) * 255).astype(np.uint8)
    
    flow_lines = []
    
    # Sample points on a grid (for efficiency)
    for y in range(0, h, density):
        for x in range(0, w, density):
            if mask[y, x]:
                # Calculate displacement
                dx, dy = flow[y, x]
                
                # Calculate end point (constrain to image boundaries)
                end_x = np.clip(int(x + dx), 0, w-1)
                end_y = np.clip(int(y + dy), 0, h-1) 
                
                # Get color from angle with increased intensity
                color_idx = angle_normalized[y, x]
                color = tuple(int(min(c * 1.2, 255)) for c in colormap[color_idx])  # BGR with intensity boost
                
                # Create flow line with the specified max_age
                flow_lines.append(FlowLine(
                    start_pos=(x, y),
                    end_pos=(end_x, end_y),
                    color=color,
                    magnitude=magnitude[y, x],
                    max_age=max_age
                ))
                
    return flow_lines


def draw_flow_lines(frame, flow_lines, base_alpha=0.7, line_thickness=1):
    """
    Draw flow lines on the frame with proper alpha blending using a single overlay.
    
    Args:
        frame: Input frame
        flow_lines: List of FlowLine objects
        base_alpha: Base alpha value
        line_thickness: Thickness of lines to draw
        
    Returns:
        Frame with rendered flow lines
    """
    # Create a single transparent overlay for all lines
    overlay = np.zeros_like(frame)
    
    # Sort by age (oldest first) to ensure newer lines appear on top when drawn
    for line in sorted(flow_lines, key=lambda l: -l.age):
        # Get opacity for this specific line
        alpha = line.get_alpha(base_alpha)
        
        # Only draw lines that are still visible (alpha > threshold)
        if alpha > 0.005:  # Lower threshold to ensure gradual fade
            # Scale the color based on the alpha but with a boost
            # Use a minimum color intensity of 25% regardless of alpha to maintain visibility
            color_intensity = max(alpha * 1.8, 0.25)  # Higher boost and minimum for rainbow colors
            color = tuple([min(int(c * color_intensity), 255) for c in line.color])
            
            # Draw the line onto the overlay
            cv2.line(overlay, 
                     line.start_pos, 
                     line.end_pos,
                     color, 
                     line_thickness, 
                     cv2.LINE_AA)
    
    # Single blend of all lines with original frame
    result = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
    
    return result

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

def get_direction_name(angle):
    """Convert angle in degrees to cardinal direction name."""
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((angle + 22.5) % 360) / 45)
    return directions[idx]

def get_position_coordinates(position, frame_width, frame_height, element_width, element_height, margin=30):
    """
    Calculate coordinates for an element based on desired position on the frame.
    
    Args:
        position: String describing position ('top-left', 'bottom-right', etc.)
        frame_width: Width of the frame
        frame_height: Height of the frame
        element_width: Width of the element to position
        element_height: Height of the element
        margin: Margin from edges
        
    Returns:
        (x, y) coordinates for top-left corner of the element
    """
    if position == 'top-left':
        return margin, margin
    elif position == 'top-right':
        return frame_width - element_width - margin, margin
    elif position == 'bottom-left':
        return margin, frame_height - element_height - margin
    elif position == 'bottom-right':
        return frame_width - element_width - margin, frame_height - element_height - margin
    elif position == 'top-center':
        return (frame_width - element_width) // 2, margin
    elif position == 'bottom-center':
        return (frame_width - element_width) // 2, frame_height - element_height - margin
    elif position == 'left-center':
        return margin, (frame_height - element_height) // 2
    elif position == 'right-center':
        return frame_width - element_width - margin, (frame_height - element_height) // 2
    else:
        # Default to top-right if position not recognized
        return frame_width - element_width - margin, margin

def draw_info_overlay(frame, metrics, history_buffer=None, info_position='top-right', histogram_position='bottom-left'):
    """
    Draw industrial monitoring information overlay on the frame with customizable positions.
    
    Args:
        frame: Input frame to draw on
        metrics: Dictionary of scene metrics
        history_buffer: Buffer of historical metrics for trends
        info_position: Position for the data display
        histogram_position: Position for the direction histogram
    
    Returns:
        Frame with info overlay
    """
    h, w = frame.shape[:2]
    
    # Define common margin for all elements
    common_margin = 30
    
    # Define dimensions for info panel
    info_width = 380
    info_height = 250
    
    # Calculate position for information panel
    info_x, info_y = get_position_coordinates(
        info_position, w, h, info_width, info_height, margin=common_margin
    )
    
    # Ensure coordinates are integers
    info_x, info_y = int(info_x), int(info_y)
    
    # Create semi-transparent dark background for readability
    info_bg = frame.copy()
    cv2.rectangle(info_bg, (info_x, info_y), (info_x + info_width, info_y + info_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.8, info_bg, 0.2, 0)
    
    # Use bright green for titles with better contrast
    title_color = (50, 255, 120)  # Vibrant lime green in BGR
    
    # Draw title text with bolder weight for better visibility
    title_text = "MONITORING DATA"
    title_pos = (info_x + 20, info_y + 20)
    cv2.putText(frame, title_text, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)
    
    # Display motion metrics with pure white for better contrast
    text_color = (255, 255, 255)
    line_height = 30
    y_pos = info_y + 50
    
    cv2.putText(frame, f"Active Motion Areas: {metrics['active_pixels']:,} pixels", 
               (info_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Scene Coverage: {metrics['scene_coverage']*100:.1f}%", 
               (info_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Mean Motion Rate: {metrics['mean_magnitude']:.2f} px/frame", 
               (info_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Max Motion Rate: {metrics['max_magnitude']:.2f} px/frame", 
               (info_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Motion Variation: {metrics['std_magnitude']:.2f} px/frame", 
               (info_x + 20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height * 1.5
    
    # Draw direction information
    direction_text = f"Primary Direction: {metrics['primary_direction']:.1f}° ({get_direction_name(metrics['primary_direction'])})"
    direction_pos = (int(info_x + 20), int(y_pos))  # Ensure integers
    cv2.putText(frame, direction_text, direction_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    cv2.putText(frame, f"Direction Confidence: {metrics['primary_direction_strength']*100:.1f}%", 
               (int(info_x + 20), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Draw direction histogram with customizable position
    histogram = metrics['direction_histogram']
    if np.sum(histogram) > 0:
        # Define dimensions for histogram
        hist_width = 220
        hist_height = 100
        title_height = 30  # Space for the title above the histogram
        label_height = 30  # Space for labels below the histogram
        
        # Calculate total dimensions including title and label space
        hist_total_height = hist_height + title_height + label_height
        hist_total_width = hist_width + 20  # Add some extra width for margins
        
        # Calculate position for histogram with the same margin
        hist_x, hist_y = get_position_coordinates(
            histogram_position, w, h, hist_total_width, hist_total_height, margin=common_margin
        )
        
        # Ensure coordinates are integers
        hist_x, hist_y = int(hist_x), int(hist_y)
        
        # Adjust y position to account for the title space
        hist_content_y = hist_y + title_height
        
        # Create semi-transparent dark background for the histogram - extend background size
        hist_bg = frame.copy()
        cv2.rectangle(hist_bg, 
                     (hist_x, hist_y),  # Start from the calculated position with margin
                     (hist_x + hist_width + 20, hist_y + hist_total_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, hist_bg, 0.2, 0)
        
        # Add histogram title with proper positioning
        histogram_title = "Motion Direction Distribution"
        hist_title_pos = (hist_x + 10, hist_y + 20)  # Position title within its space
        cv2.putText(frame, histogram_title, hist_title_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, title_color, 2)
        
        # Draw histogram background - positioned below the title
        cv2.rectangle(frame, (hist_x + 5, hist_content_y), 
                     (hist_x + 5 + hist_width, hist_content_y + hist_height), 
                     (30, 30, 30), -1)
        cv2.rectangle(frame, (hist_x + 5, hist_content_y), 
                     (hist_x + 5 + hist_width, hist_content_y + hist_height), 
                     (100, 100, 100), 1)
        
        # Draw histogram bars
        bar_width = hist_width // 8
        for i in range(8):
            bar_height = int(histogram[i] * hist_height)
            if bar_height > 0:
                cv2.rectangle(frame, 
                             (hist_x + 5 + i * bar_width, hist_content_y + hist_height - bar_height),
                             (hist_x + 5 + (i + 1) * bar_width - 2, hist_content_y + hist_height),
                             (50, 180, 250), -1)
        
        # Draw direction labels
        direction_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        for i, label in enumerate(direction_labels):
            label_pos = (int(hist_x + 5 + i * bar_width + bar_width//2 - 5), 
                        int(hist_content_y + hist_height + 15))
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    return frame

def main():
    args = parse_args()
    
    # Set up the model
    print("Loading RAFT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = of.raft_large(weights='C_T_V2').to(device).eval()
    print("Model loaded")
    
    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error opening video: {args.input}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Calculate inference size
    if args.inference_width <= 0:
        inference_width = width // 2  # Default to half size
    else:
        inference_width = args.inference_width
    
    inference_scale = inference_width / width
    inference_height = int(height * inference_scale)
    inference_size = (inference_width, inference_height)
    print(f"Inference size: {inference_width}x{inference_height}")
    
    # Calculate visualization size
    if args.visualization_width <= 0:
        vis_width = width  # Default to original size
    else:
        vis_width = args.visualization_width
    
    vis_scale = vis_width / width
    vis_height = int(height * vis_scale)
    vis_size = (vis_width, vis_height)
    print(f"Visualization size: {vis_width}x{vis_height}")
    
    # Open output video with visualization dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, vis_size)
    
    # Generate smooth colormap once
    colormap = generate_smooth_colormap()
    
    # Read first frame
    ret, frame1 = cap.read()
    if not ret:
        print("Error reading first frame")
        return
    
    # Prepare first frame for processing
    frame1_small = cv2.resize(frame1, inference_size)
    frame1_tensor = torch.from_numpy(frame1_small).permute(2, 0, 1)
    frame1_tensor = preprocess(frame1_tensor).to(device)
    frame1_tensor = frame1_tensor.unsqueeze(0)
    
    # Create queue for flow lines with temporal fading
    flow_history = deque(maxlen=args.fade_frames + 10)  # Add margin to ensure lines fade fully
    
    frame_count = 0
    start_time = time.time()
    
    # Extract info flags and position settings
    info_mode = args.info if hasattr(args, 'info') else False
    info_position = args.info_position if hasattr(args, 'info_position') else 'top-right'
    histogram_position = args.histogram_position if hasattr(args, 'histogram_position') else 'bottom-left'
    
    # Initialize metrics history for trends
    metrics_history = deque(maxlen=30)  # Store up to 30 frames of metrics
    
    print("Processing frames...")
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"Processing frame {frame_count}/{total_frames} ({fps_processing:.2f} fps)")
        
        # Resize and preprocess frame for inference
        frame2_small = cv2.resize(frame2, inference_size)
        frame2_tensor = torch.from_numpy(frame2_small).permute(2, 0, 1)
        frame2_tensor = preprocess(frame2_tensor).to(device)
        frame2_tensor = frame2_tensor.unsqueeze(0)
        
        # Compute optical flow
        flow = compute_flow(model, frame1_tensor, frame2_tensor, device)
        
        # Scale flow to match visualization size
        scale_x = vis_width / inference_width
        scale_y = vis_height / inference_height
        flow = cv2.resize(flow, vis_size)
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y
        
        # Calculate average flow magnitude for pixels above threshold
        fx, fy = flow[:,:,0], flow[:,:,1]
        magnitude = np.sqrt(fx*fx + fy*fy)
        mask = magnitude > args.min_magnitude
        if np.any(mask):
            mean_flow = np.mean(magnitude[mask])
        else:
            mean_flow = 0.0
        
        # Calculate percentage of pixels with motion
        motion_percent = 100 * np.sum(mask) / mask.size
        
        # Apply smoothing to flow field
        if args.smoothing > 0:
            flow = smooth_flow(flow, sigma=args.smoothing)
        
        # Generate flow lines for this frame
        flow_lines = generate_flow_lines(
            flow, 
            args.min_magnitude, 
            colormap, 
            density=args.line_density,
            max_age=args.fade_frames
        )
        
        # Add to history
        flow_history.append(flow_lines)
        
        # Update ages of all flow lines in history
        all_active_lines = []
        for historical_lines in flow_history:
            active_lines = []
            for line in historical_lines:
                # Always update age
                line.update_age()
                # Only add lines that aren't completely transparent
                # Use a very small threshold to ensure smooth fading to nothing
                if line.get_alpha(args.alpha) > 0.001:
                    active_lines.append(line)
            all_active_lines.extend(active_lines)
        
        # Resize frame2 to visualization size if needed
        if vis_size != (width, height):
            frame2 = cv2.resize(frame2, vis_size)
        
        # Draw flow lines on the frame
        result = draw_flow_lines(
            frame2, 
            all_active_lines, 
            base_alpha=min(args.alpha * 1.2, 1.0),  # Increase alpha but cap at 1.0
            line_thickness=args.line_thickness
        )
        
        # Add industrial monitoring info overlay if enabled
        if info_mode:
            scene_metrics = calculate_scene_metrics(flow, min_magnitude=args.min_magnitude)
            metrics_history.append(scene_metrics)
            result = draw_info_overlay(result, scene_metrics, metrics_history, 
                                     info_position=info_position,
                                     histogram_position=histogram_position)
        
        # Write frame to output video (only do this once)
        out.write(result)
        
        # Display the frame (only do this once)
        cv2.imshow('Flow Visualization', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Update for next iteration
        frame1_tensor = frame2_tensor
    
    # Cleanup
    total_time = time.time() - start_time
    print(f"Processing complete!")
    print(f"Processed {frame_count} frames in {total_time:.2f}s ({frame_count/total_time:.2f} fps)")
    print(f"Output saved to {args.output}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()