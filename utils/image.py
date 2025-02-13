"""
Image Utility Functions
Provides functions for displaying and saving images with support for different color modes.
"""

from PIL import Image
from typing import Union
import numpy as np

def display_img(img_data: np.ndarray, mode: str = 'RGB') -> None:
    """
    Display an image from numpy array data.
    
    Args:
        img_data: Image data as numpy array
        mode: Color mode of the image ('RGB' or 'BGR')
    """
    img = Image.fromarray(img_data, mode)
    if mode == 'BGR':
        # Convert BGR to RGB by swapping channels
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
    img.show()

def display_img_1d(
    img: Union[bytes, np.ndarray], 
    width: int, 
    height: int, 
    mode: str = 'RGB'
) -> None:
    """
    Display a 1D image array by reshaping it to 2D.
    
    Args:
        img: Raw image data as bytes or 1D array
        width: Image width in pixels
        height: Image height in pixels
        mode: Color mode of the image ('RGB' or 'BGR')
    """
    img = Image.frombytes('RGB', (width, height), img, 'raw')
    if mode == 'BGR':
        # Convert BGR to RGB by swapping channels
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
    img.show()

def save_img(img_data: np.ndarray, img_path: str, mode: str = 'RGB') -> None:
    """
    Save an image from numpy array data to a file.
    
    Args:
        img_data: Image data as numpy array
        img_path: Path where the image will be saved
        mode: Color mode of the image ('RGB' or 'BGR')
    """
    img = Image.fromarray(img_data, mode)
    if mode == 'BGR':
        # Convert BGR to RGB by swapping channels
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
    img.save(img_path)
