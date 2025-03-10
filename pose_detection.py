import cv2
import argparse
from ultralytics import YOLO

def process_video(input_path: str, output_path: str):
    # Load the YOLOv8 pose estimation model.
    model = YOLO('yolo11s-pose.pt')
    
    # Open the input video file.
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    # Retrieve original video properties.
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Determine scaling factor to ensure a maximum width of 1024 while preserving aspect ratio.
    if original_width > 1024:
        scale_factor = 1024 / original_width
    else:
        scale_factor = 1.0

    # Calculate new dimensions.
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Setup the video writer with the new dimensions.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    print("Starting pose estimation. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Resize frame if needed.
        if scale_factor != 1.0:
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Run pose estimation on the (resized) frame.
        results = model(frame)
        
        # The first (and only) result contains the annotated image.
        annotated_frame = results[0].plot()
        
        # Write the annotated frame to the output video.
        out.write(annotated_frame)
        
        # Display the annotated frame live.
        cv2.imshow('Pose Estimation', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting on user request.")
            break

    # Clean up resources.
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run pose estimation on a video using Ultralytics YOLOv8, resizing frames to a maximum width of 1024, and save the annotated output."
    )
    parser.add_argument("input_video", type=str, help="Path to the input video file.")
    parser.add_argument("output_video", type=str, help="Path to save the annotated output video.")
    args = parser.parse_args()
    
    process_video(args.input_video, args.output_video)
