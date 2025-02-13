"""
Dataset Splitter
Splits image and label datasets into training, testing, and validation sets
while maintaining corresponding pairs of files.
"""

import os
import random
import shutil
from typing import List, Tuple
from pathlib import Path

# Configuration
CONFIG = {
    'paths': {
        'source': {
            'images': './datasets/smoke_and_no_smoke_hpwren/full/images/',
            'labels': './datasets/smoke_and_no_smoke_hpwren/full/labels/'
        },
        'output': {
            'training': {
                'images': './datasets/smoke_and_no_smoke_hpwren/training/images/',
                'labels': './datasets/smoke_and_no_smoke_hpwren/training/labels/'
            },
            'testing': {
                'images': './datasets/smoke_and_no_smoke_hpwren/testing/images/',
                'labels': './datasets/smoke_and_no_smoke_hpwren/testing/labels/'
            },
            'validation': {
                'images': './datasets/smoke_and_no_smoke_hpwren/validation/images/',
                'labels': './datasets/smoke_and_no_smoke_hpwren/validation/labels/'
            }
        }
    },
    'split_ratios': [0.8, 0.1, 0.1],  # training, testing, validation
    'image_extensions': ['.jpg', '.jpeg']
}

def get_image_files(
    source_dir: str,
    valid_extensions: List[str]
) -> List[str]:
    """
    Get list of image files without extensions.
    
    Args:
        source_dir: Source directory path
        valid_extensions: List of valid file extensions
        
    Returns:
        list: Base filenames without extensions
    """
    return [
        Path(f).stem for f in os.listdir(source_dir)
        if f.lower().endswith(tuple(valid_extensions))
    ]

def split_dataset(
    files: List[str],
    split_ratios: List[float]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split list of files into three sets based on ratios.
    
    Args:
        files: List of files to split
        split_ratios: List of ratios [train, test, val]
        
    Returns:
        tuple: (training_files, testing_files, validation_files)
    """
    random.shuffle(files)
    
    train_split = int(len(files) * split_ratios[0])
    test_split = int(len(files) * (split_ratios[0] + split_ratios[1]))
    
    return (
        files[:train_split],
        files[train_split:test_split],
        files[test_split:]
    )

def copy_files(
    file_list: List[str],
    source_dir: str,
    dest_dir: str,
    extensions: List[str]
):
    """
    Copy files with matching extensions to destination.
    
    Args:
        file_list: List of base filenames
        source_dir: Source directory
        dest_dir: Destination directory
        extensions: List of valid file extensions
    """
    for filename in file_list:
        for ext in extensions:
            source_path = os.path.join(source_dir, filename + ext)
            if os.path.exists(source_path):
                dest_path = os.path.join(dest_dir, filename + ext)
                shutil.copyfile(source_path, dest_path)
                break  # Use first matching extension

def create_directories():
    """Create all necessary output directories."""
    for split_type in ['training', 'testing', 'validation']:
        for data_type in ['images', 'labels']:
            os.makedirs(
                CONFIG['paths']['output'][split_type][data_type],
                exist_ok=True
            )

def split_data():
    """Split dataset into training, testing, and validation sets."""
    # Get list of image files
    image_files = get_image_files(
        CONFIG['paths']['source']['images'],
        CONFIG['image_extensions']
    )
    
    # Split into sets
    train_files, test_files, val_files = split_dataset(
        image_files,
        CONFIG['split_ratios']
    )
    
    # Create output directories
    create_directories()
    
    # Copy image files
    for files, split_type in [
        (train_files, 'training'),
        (test_files, 'testing'),
        (val_files, 'validation')
    ]:
        # Copy images
        copy_files(
            files,
            CONFIG['paths']['source']['images'],
            CONFIG['paths']['output'][split_type]['images'],
            CONFIG['image_extensions']
        )
        
        # Copy corresponding labels
        copy_files(
            files,
            CONFIG['paths']['source']['labels'],
            CONFIG['paths']['output'][split_type]['labels'],
            ['.txt']
        )

def main():
    """Execute dataset splitting process."""
    split_data()

if __name__ == "__main__":
    main()
