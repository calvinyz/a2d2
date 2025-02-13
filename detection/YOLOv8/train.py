"""
YOLOv8 Training Script
Handles training of YOLOv8 models for wildfire detection with configurable parameters.
"""

from ultralytics import YOLO
from os.path import join, abspath, dirname
from typing import Optional

# Training configuration
TRAINING_CONFIG = {
    'model_name': 'yolov8n.pt',
    'epochs': 200,
    'image_size': 640,
    'batch_size': 8
}

def train_model(
    model_name: str,
    data_config: str,
    epochs: int,
    image_size: int,
    batch_size: int
) -> tuple[dict, dict]:
    """
    Train a YOLOv8 model for wildfire detection.
    
    Args:
        model_name: Base model name (e.g., 'yolov8n.pt')
        data_config: Path to data configuration YAML file
        epochs: Number of training epochs
        image_size: Input image size
        batch_size: Training batch size
        
    Returns:
        tuple: (training_results, validation_results)
    """
    # Initialize model
    model = YOLO(model_name)
    
    # Generate output model name
    output_name = f'{model_name[:-3]}_wildfire_e{epochs}_local'
    
    # Train model
    training_results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        name=output_name
    )
    
    # Validate model
    validation_results = model.val()
    
    return training_results, validation_results

def main():
    """Main training execution."""
    # Setup paths
    current_dir = dirname(abspath(__file__))
    data_config_path = join(current_dir, 'wildfire_local_yolov8/data.yaml')
    
    # Run training
    train_model(
        model_name=TRAINING_CONFIG['model_name'],
        data_config=data_config_path,
        epochs=TRAINING_CONFIG['epochs'],
        image_size=TRAINING_CONFIG['image_size'],
        batch_size=TRAINING_CONFIG['batch_size']
    )

if __name__ == '__main__':
    main()
