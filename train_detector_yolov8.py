"""
YOLOv8 Training Script
Handles training of YOLOv8 model on wildfire dataset.
"""

from ultralytics import YOLO
from os.path import join, abspath, dirname
import argparse
import yaml
import logging
from pathlib import Path

# Default training parameters
DEFAULT_MODEL = 'yolov8s.pt'
DEFAULT_EPOCHS = 200
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 8

def setup_logging():
    """Configure logging for the training process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_yolo(
    data_yaml: str,
    model_name: str = 'yolov8n.pt',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = 'cuda:0'
):
    """Train YOLOv8 model."""
    model = YOLO(model_name)
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device
    )
    
    # Validate the model
    val_results = model.val()
    
    return train_results, val_results

def main():
    """Main function to handle training workflow"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for wildfire detection')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                      help='Path to training configuration file')
    args = parser.parse_args()

    setup_logging()
    
    try:
        config = load_config(args.config)
        train_yolo(
            config.get('data_yaml'),
            config.get('model_name', DEFAULT_MODEL),
            config.get('epochs', DEFAULT_EPOCHS),
            config.get('batch_size', DEFAULT_BATCH_SIZE),
            config.get('image_size', DEFAULT_IMG_SIZE),
            config.get('device', 'cuda:0')
        )
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
