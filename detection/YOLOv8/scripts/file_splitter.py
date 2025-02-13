"""
Dataset Splitter for JPEG Images
Splits JPEG images and their corresponding label files into training,
testing, and validation sets while maintaining paired relationships.
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
            'images': './datasets/annotated_hpwren_small/full/images/',
            'labels': './datasets/annotated_hpwren_small/full/labels/'
        },
        'output': {
            'training': {
                'images': './datasets/annotated_hpwren_small/training/images/',
                'labels': './datasets/annotated_hpwren_small/training/labels/'
            },
            'testing': {
                'images': './datasets/annotated_hpwren_small/testing/images/',
                'labels': './datasets/annotated_hpwren_small/testing/labels/'
            },
            'validation': {
                'images': './datasets/annotated_hpwren_small/validation/images/',
                'labels': './datasets/annotated_hpwren_small/validation/labels/'
            }
        }
    },
    'split_ratios': [0.8, 0.1, 0.1],  # training, testing, validation
    'extensions': {
        'images': '.jpeg',
        'labels': '.txt'
    }
}

def get_image_files(source_dir: str) -> List[str]:
    """
    Get list of JPEG image files without extensions.
    
    Args:
        source_dir: Source directory path
        
    Returns:
        list: Base filenames without extensions
    """
    return [
        f[:-5] for f in os.listdir(source_dir)
        if f.endswith(CONFIG['extensions']['images'])
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
    extension: str
):
    """
    Copy files with specified extension to destination.
    
    Args:
        file_list: List of base filenames
        source_dir: Source directory
        dest_dir: Destination directory
        extension: File extension to use
    """
    for filename in file_list:
        source_path = os.path.join(source_dir, filename + extension)
        dest_path = os.path.join(dest_dir, filename + extension)
        shutil.copyfile(source_path, dest_path)

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
    image_files = get_image_files(CONFIG['paths']['source']['images'])
    
    # Split into sets
    train_files, test_files, val_files = split_dataset(
        image_files,
        CONFIG['split_ratios']
    )
    
    # Create output directories
    create_directories()
    
    # Copy files for each split
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
            CONFIG['extensions']['images']
        )
        
        # Copy corresponding labels
        copy_files(
            files,
            CONFIG['paths']['source']['labels'],
            CONFIG['paths']['output'][split_type]['labels'],
            CONFIG['extensions']['labels']
        )

def main():
    """Execute dataset splitting process."""
    split_data()

if __name__ == "__main__":
    main()
