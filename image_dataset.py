"""
Image Dataset Collection Script
Collects drone images at various altitudes and camera angles for training the fire detection model.
"""

from os.path import join, dirname, abspath
from collector.image_data import ImageDataCollector
from drone.airsim_drone import AirsimDrone

# Data collection parameters
ALTITUDE_CONFIG = {
    'min': 20,
    'max': 125,
    'step': 10
}

CAMERA_ANGLE_CONFIG = {
    'min': -90,
    'max': -10,
    'step': 15
}

FIRE_GRID_CONFIG = {
    'n_x': 3,  # Number of fire positions in x direction
    'n_y': 3   # Number of fire positions in y direction
}

def setup_data_collection() -> tuple[ImageDataCollector, str]:
    """
    Initialize the data collector and prepare output directory.
    
    Returns:
        tuple: (ImageDataCollector instance, output directory path)
    """
    # Initialize drone
    drone = AirsimDrone()
    
    # Create data collector
    collector = ImageDataCollector(
        altitude_range=[ALTITUDE_CONFIG['min'], ALTITUDE_CONFIG['max']],
        altitude_step=ALTITUDE_CONFIG['step'],
        camera_angle_range=[CAMERA_ANGLE_CONFIG['min'], CAMERA_ANGLE_CONFIG['max']],
        camera_angle_step=CAMERA_ANGLE_CONFIG['step'],
        fire_n_x=FIRE_GRID_CONFIG['n_x'],
        fire_n_y=FIRE_GRID_CONFIG['n_y'],
        drone=drone
    )
    
    # Set output directory
    output_dir = join(dirname(abspath(__file__)), 'images')
    
    return collector, output_dir

def main():
    """Main execution function"""
    collector, output_dir = setup_data_collection()
    collector.collect_imgs(output_dir)

if __name__ == '__main__':
    main()