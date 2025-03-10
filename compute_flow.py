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
    parser.add_argument('--min_magnitude', type=float, default=1.0,
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
        
        # Add frame counter and average flow magnitude (above threshold)
        # cv2.putText(result, f"Frame: {frame_count}", (10, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"Active Flow: {mean_flow:.2f} ({motion_percent:.1f}%)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(result)
        
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