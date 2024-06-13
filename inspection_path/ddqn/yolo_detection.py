import numpy as np
from detector.YOLOv8.model import YOLOv8
import airsim
from PIL import Image

def process_image(client):
    responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    image = img1d.reshape(response.height, response.width, 3)

    # Resize the image to 640 by 640 for YOLO
    image = Image.fromarray(image)
    image = image.resize((640, 640))

    # Convert the image back to a NumPy array
    image = np.array(image)

    # Load a pretrained YOLOv8n model
    model = YOLOv8('yolov8n_wildfire_as_e100_new.pt')

    # Run inference on the resized image
    results = model.detect(image)

    return results