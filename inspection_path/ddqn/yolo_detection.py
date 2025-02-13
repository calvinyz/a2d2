"""
YOLO Detection Module
Provides image processing and object detection functionality using YOLOv8
for drone-captured images in AirSim environment.
"""

import numpy as np
from PIL import Image
import airsim
from detector.YOLOv8.model import YOLOv8

# Configuration constants
MODEL_CONFIG = {
    'weights_path': 'yolov8n_wildfire_as_e100_new.pt',
    'target_size': (640, 640)  # YOLO input size
}

def process_image(client: airsim.MultirotorClient) -> list:
    """
    Process drone camera image and perform object detection.
    
    Args:
        client: AirSim client instance for capturing images
        
    Returns:
        list: Detection results from YOLOv8 model
    """
    # Capture image from drone camera
    responses = client.simGetImages([
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)
    ])
    response = responses[0]
    
    # Convert image data to numpy array
    img_1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    image = img_1d.reshape(response.height, response.width, 3)
    
    # Preprocess image for YOLO
    processed_image = preprocess_image(image)
    
    # Initialize and run YOLOv8 model
    model = YOLOv8(MODEL_CONFIG['weights_path'])
    return model.detect(processed_image)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for YOLO model input.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Convert to PIL Image for resizing
    pil_image = Image.fromarray(image)
    
    # Resize to YOLO input dimensions
    resized_image = pil_image.resize(MODEL_CONFIG['target_size'])
    
    # Convert back to numpy array
    return np.array(resized_image)