"""
Empty Bounding Box Generator
Creates YOLO format annotation files with full-image bounding boxes
for images without smoke, indicating background/negative samples.
"""

import os
from pathlib import Path
from typing import Set

# Configuration
CONFIG = {
    'paths': {
        'image_dir': './datasets/unannotated_hpwren/no_smoke/images',
        'label_dir': './datasets/unannotated_hpwren/no_smoke/labels'
    },
    'annotation': {
        'class_id': 0,  # Class ID for no smoke
        'bbox': '0.0 0.0 1.0 1.0'  # Full image bounding box
    },
    'image_extensions': {'.jpg', '.jpeg', '.png'}
}

def is_image_file(filename: str, valid_extensions: Set[str]) -> bool:
    """
    Check if file is an image based on extension.
    
    Args:
        filename: Name of file to check
        valid_extensions: Set of valid image extensions
        
    Returns:
        bool: True if file is an image, False otherwise
    """
    return Path(filename).suffix.lower() in valid_extensions

def create_empty_annotation(
    image_path: str,
    output_path: str,
    class_id: int,
    bbox: str
):
    """
    Create YOLO format annotation file for an image.
    
    Args:
        image_path: Path to source image
        output_path: Path for output annotation file
        class_id: Class ID for annotation
        bbox: Bounding box coordinates in YOLO format
    """
    annotation = f'{class_id} {bbox}'
    with open(output_path, 'w') as f:
        f.write(annotation)

def generate_empty_annotations(
    image_dir: str,
    label_dir: str,
    class_id: int = 0,
    bbox: str = '0.0 0.0 1.0 1.0'
):
    """
    Generate empty annotations for all images in directory.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory for output annotation files
        class_id: Class ID to use in annotations
        bbox: Bounding box coordinates in YOLO format
    """
    # Create output directory if it doesn't exist
    os.makedirs(label_dir, exist_ok=True)
    
    # Process each image file
    for filename in os.listdir(image_dir):
        if not is_image_file(filename, CONFIG['image_extensions']):
            continue
            
        # Generate paths
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(
            label_dir,
            f"{Path(filename).stem}.txt"
        )
        
        # Create annotation file
        create_empty_annotation(
            image_path,
            label_path,
            class_id,
            bbox
        )

def main():
    """Generate empty annotations for all images."""
    generate_empty_annotations(
        image_dir=CONFIG['paths']['image_dir'],
        label_dir=CONFIG['paths']['label_dir'],
        class_id=CONFIG['annotation']['class_id'],
        bbox=CONFIG['annotation']['bbox']
    )

if __name__ == "__main__":
    main()
