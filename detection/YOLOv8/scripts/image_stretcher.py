"""
Image Stretcher
Stretches images to a target size without preserving aspect ratio,
useful for preparing images for neural network input.
"""

from PIL import Image
import os
from pathlib import Path
from typing import Set, Tuple

# Configuration
CONFIG = {
    'paths': {
        'input': './datasets/flame_aerial_dataset/resized',
        'output': './datasets/flame_aerial_dataset/stretched'
    },
    'image': {
        'target_size': (640, 640),
        'resampling': Image.ANTIALIAS
    },
    'extensions': {'.jpg', '.jpeg', '.png'}
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

def stretch_single_image(
    input_path: str,
    output_path: str,
    target_size: Tuple[int, int]
):
    """
    Stretch a single image to target size.
    
    Args:
        input_path: Path to input image
        output_path: Path for output image
        target_size: Desired (width, height)
    """
    try:
        with Image.open(input_path) as img:
            stretched_img = img.resize(
                target_size,
                CONFIG['image']['resampling']
            )
            stretched_img.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_directory(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int]
):
    """
    Process all images in directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        target_size: Target size for stretching
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for filename in os.listdir(input_dir):
        if not is_image_file(filename, CONFIG['extensions']):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        stretch_single_image(
            input_path,
            output_path,
            target_size
        )

def main():
    """Execute image stretching process."""
    process_directory(
        input_dir=CONFIG['paths']['input'],
        output_dir=CONFIG['paths']['output'],
        target_size=CONFIG['image']['target_size']
    )

if __name__ == "__main__":
    main()
