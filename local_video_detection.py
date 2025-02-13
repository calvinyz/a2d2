"""
Local Video Fire Detection Script
Processes video files to detect wildfires using YOLOv8 model.
Supports frame extraction and visualization of detection results.
"""

import cv2
import os
from pathlib import Path
from PIL import Image
import numpy as np

from detection.YOLOv8.model import YOLOv8
from utils.image import display_img, save_img

def extract_frames(video_path: str, output_dir: str) -> int:
    """
    Extract frames from a video file and save them as PNG images.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        
    Returns:
        int: Number of frames extracted
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process video frames
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
            
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    video.release()
    print(f"Frames extracted: {frame_count}")
    return frame_count

def process_fire_detection():
    """Run fire detection on a sample image using YOLOv8 model"""
    # Model configuration
    WEIGHTS_FILE = 'yolov8n_wildfire_local_e200.pt'
    IMAGE_PATH = ".\\images\\fire_frames\\frame_111_1.png"
    TARGET_SIZE = [1088, 1920]
    
    # Initialize detector
    detector = YOLOv8(WEIGHTS_FILE)
    
    # Load and prepare image
    fire_img = cv2.imread(IMAGE_PATH)
    fire_img_copy = fire_img.copy()
    
    # Run detection
    detections = detector.detect(fire_img, TARGET_SIZE)
    
    # Visualize results
    fire_img_with_detections = detector.visualize_detections(fire_img_copy, detections)
    display_img(fire_img_with_detections)

def main():
    """Main execution function"""
    # Video processing configuration
    VIDEO_CONFIG = {
        'video_path': ".\\fire_video.mp4",
        'output_dir': ".\\images\\fire_frames"
    }
    
    # Uncomment to extract frames from video
    # extract_frames(**VIDEO_CONFIG)
    
    # Run fire detection on sample image
    process_fire_detection()

if __name__ == "__main__":
    main()    

