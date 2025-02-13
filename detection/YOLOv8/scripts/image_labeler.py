"""
XML to YOLO Label Converter
Converts XML annotation files to YOLO format text files with normalized coordinates.
Handles single-class object detection with class ID 0.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Dict

# Configuration
CONFIG = {
    'paths': {
        'input': './datasets/annotated_hpwren_small/full/xmls/',
        'output': './datasets/annotated_hpwren_small/full/labels/'
    },
    'class_id': 0,  # Single class for smoke detection
    'extensions': {
        'input': '.xml',
        'output': '.txt'
    }
}

def get_image_dimensions(root: ET.Element) -> Tuple[float, float]:
    """
    Extract image dimensions from XML root.
    
    Args:
        root: XML root element
        
    Returns:
        tuple: (width, height) of image
    """
    size_elem = root.find('size')
    if size_elem is None:
        raise ValueError("XML missing size element")
        
    width = float(size_elem.find('width').text)
    height = float(size_elem.find('height').text)
    return width, height

def get_bounding_box(
    root: ET.Element,
    width: float,
    height: float
) -> Tuple[float, float, float, float]:
    """
    Extract and normalize bounding box coordinates.
    
    Args:
        root: XML root element
        width: Image width for normalization
        height: Image height for normalization
        
    Returns:
        tuple: Normalized (xmin, ymin, xmax, ymax)
    """
    bbox_elem = root.find('.//bndbox')
    if bbox_elem is None:
        raise ValueError("XML missing bounding box element")
        
    # Extract and normalize coordinates
    coords = {
        'xmin': float(bbox_elem.find('xmin').text) / width,
        'ymin': float(bbox_elem.find('ymin').text) / height,
        'xmax': float(bbox_elem.find('xmax').text) / width,
        'ymax': float(bbox_elem.find('ymax').text) / height
    }
    
    return (coords['xmin'], coords['ymin'], coords['xmax'], coords['ymax'])

def convert_annotation(xml_path: str, output_dir: str):
    """
    Convert single XML annotation to YOLO format.
    
    Args:
        xml_path: Path to input XML file
        output_dir: Directory for output text file
    """
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions and bounding box
    width, height = get_image_dimensions(root)
    xmin, ymin, xmax, ymax = get_bounding_box(root, width, height)
    
    # Create output filename
    output_name = Path(xml_path).stem + CONFIG['extensions']['output']
    output_path = os.path.join(output_dir, output_name)
    
    # Write YOLO format annotation
    with open(output_path, 'w') as f:
        f.write(f'{CONFIG["class_id"]} {xmin} {ymin} {xmax} {ymax}')

def process_annotations():
    """Process all XML annotations in directory."""
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG['paths']['output'], exist_ok=True)
    
    # Process each XML file
    for filename in os.listdir(CONFIG['paths']['input']):
        if not filename.endswith(CONFIG['extensions']['input']):
            continue
            
        xml_path = os.path.join(CONFIG['paths']['input'], filename)
        try:
            convert_annotation(xml_path, CONFIG['paths']['output'])
        except (ValueError, ET.ParseError) as e:
            print(f"Error processing {filename}: {str(e)}")

def main():
    """Execute XML to YOLO label conversion."""
    process_annotations()

if __name__ == "__main__":
    main()
    