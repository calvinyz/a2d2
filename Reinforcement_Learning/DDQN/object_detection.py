import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def detect_objects(image):
    """ Use YOLOv8 with trained weights to detect fire """
    image = Image.fromarray(image).resize((640, 640))
    image = np.array(image)
    model = YOLO('.\\yolov8_weights.pt')
    results = model(image)

    return results

def visualize_detections(image, detections):
    """ Draw image with YOLOv8 detections """
    boxes = detections[0].boxes.xyxy
    class_name = detections[0].boxes.cls
    confidence = detections[0].boxes.conf

    for box, cls, conf in zip(boxes, class_name, confidence):
        # Draw bounding box
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # Display class name and confidence
        label = f"{cls.item()}: {conf.item():.2f}"
        cv2.putText(image, label, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image