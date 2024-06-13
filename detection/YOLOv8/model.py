from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from os.path import join, dirname, abspath

class YOLOv8:
    def __init__(self, weights_file):
        cur_dir = dirname(abspath(__file__))
        self.model = YOLO(join(cur_dir, weights_file))

    def train(self, data_pathfile, epochs=100, img_size=640, batch=8, name='yolov8_wildfire_e100.pt'):
        return self.model.train(data=data_pathfile, 
                                epochs=epochs, 
                                imgsz=img_size, 
                                batch=batch, 
                                name=name)
    
    def detect(self, img, img_size=640):
        return self.model.predict(img, imgsz=img_size)
    
    def visualize_detections(self, image, detections):
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
