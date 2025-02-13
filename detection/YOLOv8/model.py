"""
YOLOv8 Model Wrapper
Provides a simplified interface for training and using YOLOv8 models
for wildfire detection with visualization capabilities.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from os.path import join, dirname, abspath
from typing import Union, List, Optional

# Visualization configuration
VISUALIZATION_CONFIG = {
    'box_color': (0, 0, 255),  # Red in BGR
    'text_color': (0, 0, 255),  # Red in BGR
    'box_thickness': 1,
    'text_thickness': 1,
    'font_scale': 0.5,
    'font': cv2.FONT_HERSHEY_SIMPLEX
}

class YOLOv8:
    """YOLOv8 model wrapper for wildfire detection."""
    
    def __init__(self, weights_file: str):
        """
        Initialize YOLOv8 model.
        
        Args:
            weights_file: Path to model weights file
        """
        model_path = join(dirname(abspath(__file__)), weights_file)
        self.model = YOLO(model_path)

    def train(
        self,
        data_pathfile: str,
        epochs: int = 100,
        img_size: int = 640,
        batch: int = 8,
        name: str = 'yolov8_wildfire_e100.pt'
    ) -> dict:
        """
        Train the YOLOv8 model.
        
        Args:
            data_pathfile: Path to data configuration file
            epochs: Number of training epochs
            img_size: Input image size
            batch: Batch size
            name: Output model name
            
        Returns:
            dict: Training results
        """
        return self.model.train(
            data=data_pathfile,
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            name=name
        )
    
    def detect(
        self,
        img: np.ndarray,
        img_size: Union[int, List[int]] = 640
    ) -> List:
        """
        Perform object detection on an image.
        
        Args:
            img: Input image as numpy array
            img_size: Target image size(s)
            
        Returns:
            List: Detection results
        """
        return self.model.predict(img, imgsz=img_size)
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List
    ) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Args:
            image: Input image
            detections: Detection results from model
            
        Returns:
            np.ndarray: Image with visualized detections
        """
        if not detections or len(detections) == 0 or len(detections[0].boxes.xyxy) == 0:
            return image
        boxes = detections[0].boxes.xyxy
        class_name = detections[0].boxes.cls
        confidence = detections[0].boxes.conf

        for box, cls, conf in zip(boxes, class_name, confidence):
            # Draw bounding box
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)

            # Display class name and confidence
            label = f"A2D2 Alert: {self.model.names[int(cls)]}: {conf.item():.1f}"
            cv2.putText(image, label, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return image    
    

if __name__ == '__main__':
    cur_dir = dirname(abspath(__file__))
    weights_file = 'yolov8s_wildfire_as_e100_new_2.pt'
    weights_path = join(cur_dir, weights_file)
    detector = YOLOv8(weights_path)

    img_file = 'Drone_Wildfire_Detection.png'
    img_path = join(cur_dir, img_file)
    img = cv2.imread(img_path)

    res = detector.detect(img, [545,959])
    img_res = detector.visualize_detections(img, res)
    img_res_path = img_path[:-4] + '-res.png'
    cv2.imwrite(img_res_path, img_res)
