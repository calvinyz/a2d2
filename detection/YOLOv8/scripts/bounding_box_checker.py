"""
Bounding Box Visualization Tool
Visualizes XML-defined bounding boxes on images to verify annotation accuracy.
"""

import os
import xml.etree.ElementTree as ET
import cv2
from typing import Tuple
from pathlib import Path

# Configuration
CONFIG = {
    'paths': {
        'xml_dir': './datasets/annotated_hpwren_large/full/annotations/xmls',
        'image_dir': './datasets/annotated_hpwren_large/full/images',
        'output_dir': './datasets/annotated_hpwren_large/full/annotated/'
    },
    'visualization': {
        'box_color': (255, 0, 0),  # Blue in BGR
        'box_thickness': 2
    }
}

def parse_xml_dimensions(root: ET.Element) -> Tuple[float, float]:
    """
    Extract image dimensions from XML.
    
    Args:
        root: XML root element
        
    Returns:
        tuple: (width, height) of image
    """
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    return width, height

def parse_bounding_box(coordinates: ET.Element, width: float, height: float) -> Tuple[int, int, int, int]:
    """
    Parse and normalize bounding box coordinates.
    
    Args:
        coordinates: XML bounding box element
        width: Image width
        height: Image height
        
    Returns:
        tuple: (xmin, ymin, xmax, ymax) in pixel coordinates
    """
    # Get normalized coordinates
    xmin = float(coordinates.find('xmin').text) / width
    ymin = float(coordinates.find('ymin').text) / height
    xmax = float(coordinates.find('xmax').text) / width
    ymax = float(coordinates.find('ymax').text) / height
    
    # Convert to pixel coordinates
    return (
        int(xmin * width),
        int(ymin * height),
        int(xmax * width),
        int(ymax * height)
    )

def process_annotation(xml_path: str):
    """
    Process a single annotation file and visualize bounding box.
    
    Args:
        xml_path: Path to XML annotation file
    """
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    width, height = parse_xml_dimensions(root)
    
    # Get bounding box coordinates
    bbox_elem = root.find('.//bndbox')
    if bbox_elem is None:
        print(f"Warning: No bounding box found in {xml_path}")
        return
        
    xmin, ymin, xmax, ymax = parse_bounding_box(bbox_elem, width, height)
    
    # Load corresponding image
    image_filename = Path(xml_path).stem + '.jpeg'
    image_path = os.path.join(CONFIG['paths']['image_dir'], image_filename)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
        
    # Draw bounding box
    cv2.rectangle(
        image,
        (xmin, ymin),
        (xmax, ymax),
        CONFIG['visualization']['box_color'],
        CONFIG['visualization']['box_thickness']
    )
    
    # Save annotated image
    output_filename = f"{Path(xml_path).stem}_annotated.jpeg"
    output_path = os.path.join(CONFIG['paths']['output_dir'], output_filename)
    cv2.imwrite(output_path, image)

def main():
    """Process all XML annotations in directory."""
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG['paths']['output_dir'], exist_ok=True)
    
    # Process each XML file
    for xml_file in os.listdir(CONFIG['paths']['xml_dir']):
        if not xml_file.endswith('.xml'):
            continue
            
        xml_path = os.path.join(CONFIG['paths']['xml_dir'], xml_file)
        process_annotation(xml_path)

if __name__ == "__main__":
    main()
