"""
Image Resizer
Resizes images in a directory while maintaining aspect ratio,
with configurable target width and sampling frequency.
"""

from PIL import Image
import os
from pathlib import Path
from typing import Set, Tuple

# Configuration
CONFIG = {
    'paths': {
        'input': './datasets/flame_aerial_dataset/images',
        'output': './datasets/flame_aerial_dataset/resized'
    },
    'resize': {
        'target_width': 640,
        'sample_frequency': 10  # Process every Nth image
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

def calculate_new_dimensions(
    original_size: Tuple[int, int],
    target_width: int
) -> Tuple[int, int]:
    """
    Calculate new dimensions maintaining aspect ratio.
    
    Args:
        original_size: Original (width, height)
        target_width: Desired width
        
    Returns:
        tuple: New (width, height)
    """
    width, height = original_size
    scale_factor = target_width / width
    new_height = int(height * scale_factor)
    return (target_width, new_height)

def resize_single_image(
    input_path: str,
    output_path: str,
    target_width: int
):
    """
    Resize a single image maintaining aspect ratio.
    
    Args:
        input_path: Path to input image
        output_path: Path for output image
        target_width: Desired width in pixels
    """
    try:
        with Image.open(input_path) as img:
            new_size = calculate_new_dimensions(img.size, target_width)
            resized_img = img.resize(new_size, Image.ANTIALIAS)
            resized_img.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_directory(
    input_dir: str,
    output_dir: str,
    target_width: int,
    sample_frequency: int
):
    """
    Process images in directory with sampling.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        target_width: Target width for resizing
        sample_frequency: Process every Nth image
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    for idx, filename in enumerate(os.listdir(input_dir)):
        if not is_image_file(filename, CONFIG['image_extensions']):
            continue
            
        # Skip files based on sampling frequency
        if idx % sample_frequency != 0:
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        resize_single_image(input_path, output_path, target_width)

def main():
    """Execute image resizing process."""
    process_directory(
        input_dir=CONFIG['paths']['input'],
        output_dir=CONFIG['paths']['output'],
        target_width=CONFIG['resize']['target_width'],
        sample_frequency=CONFIG['resize']['sample_frequency']
    )

if __name__ == "__main__":
    main()
