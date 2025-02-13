"""
Object Detection Module
Provides functions for detecting and visualizing wildfires using YOLOv8 model.
"""

import cv2
import numpy as np
from PIL import Image
from detector.YOLOv8.model import YOLOv8

# Model configuration
MODEL_PATH = 'yolov8n_wildfire_as_e100_new.pt'
VISUALIZATION_CONFIG = {
    'box_color': (0, 255, 0),  # Green in BGR
    'box_thickness': 2,
    'text_color': (0, 255, 0),  # Green in BGR
    'text_thickness': 2,
    'font_scale': 0.5,
    'font': cv2.FONT_HERSHEY_SIMPLEX
}

def detect_objects(image: np.ndarray) -> list:
    """
    Detect wildfires in the given image using YOLOv8.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        list: Detection results containing bounding boxes, classes, and confidence scores
    """
    detector = YOLOv8(MODEL_PATH)
    return detector.detect(image)

def visualize_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw detection results on the image.
    
    Args:
        image: Input image as numpy array
        detections: Detection results from YOLOv8
        
    Returns:
        np.ndarray: Image with visualized detections
    """
    boxes = detections[0].boxes.xyxy
    class_names = detections[0].boxes.cls
    confidences = detections[0].boxes.conf

    # Draw each detection
    for box, class_name, confidence in zip(boxes, class_names, confidences):
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            VISUALIZATION_CONFIG['box_color'],
            VISUALIZATION_CONFIG['box_thickness']
        )

        # Create and draw label
        label = f"{class_name.item()}: {confidence.item():.2f}"
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            VISUALIZATION_CONFIG['font'],
            VISUALIZATION_CONFIG['font_scale'],
            VISUALIZATION_CONFIG['text_color'],
            VISUALIZATION_CONFIG['text_thickness']
        )

    return image