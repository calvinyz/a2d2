"""
Image Mask Overlayer
Overlays mask images on white backgrounds with configurable transparency
for visualization and dataset preparation.
"""

from PIL import Image
import os
from pathlib import Path
from typing import Tuple

# Configuration
CONFIG = {
    'paths': {
        'images': './datasets/flame_aerial_dataset/images/',
        'masks': './datasets/flame_aerial_dataset/masks/',
        'output': './datasets/flame_aerial_dataset/overlayed_images/'
    },
    'image': {
        'size': (640, 480),
        'alpha': 0.5,
        'background': 'white',
        'format': 'PNG'
    },
    'extensions': {
        'input': '.png'
    }
}

def create_base_image(size: Tuple[int, int], color: str) -> Image.Image:
    """
    Create a base image with specified size and color.
    
    Args:
        size: Tuple of (width, height)
        color: Background color name
        
    Returns:
        Image: New base image
    """
    return Image.new('RGB', size, color)

def overlay_mask(
    mask_path: str,
    output_path: str,
    image_size: Tuple[int, int],
    alpha: float = 0.5
):
    """
    Overlay a mask on a white background with transparency.
    
    Args:
        mask_path: Path to mask image
        output_path: Path for output image
        image_size: Target image size (width, height)
        alpha: Transparency level (0-1)
    """
    # Create white background
    base_image = create_base_image(image_size, CONFIG['image']['background'])
    
    # Load and resize mask
    try:
        mask = Image.open(mask_path)
        mask = mask.resize(image_size, Image.ANTIALIAS)
        
        # Blend images
        blended = Image.blend(base_image, mask, alpha=alpha)
        
        # Save result
        blended.save(
            output_path,
            format=CONFIG['image']['format']
        )
    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")

def process_mask_folder(
    mask_dir: str,
    output_dir: str,
    image_size: Tuple[int, int],
    alpha: float
):
    """
    Process all mask images in a directory.
    
    Args:
        mask_dir: Directory containing mask images
        output_dir: Directory for output images
        image_size: Target image size (width, height)
        alpha: Transparency level (0-1)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each mask file
    for filename in os.listdir(mask_dir):
        if not filename.lower().endswith(CONFIG['extensions']['input']):
            continue
            
        mask_path = os.path.join(mask_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        overlay_mask(
            mask_path,
            output_path,
            image_size,
            alpha
        )

def main():
    """Execute mask overlay process."""
    process_mask_folder(
        mask_dir=CONFIG['paths']['masks'],
        output_dir=CONFIG['paths']['output'],
        image_size=CONFIG['image']['size'],
        alpha=CONFIG['image']['alpha']
    )

if __name__ == "__main__":
    main()
