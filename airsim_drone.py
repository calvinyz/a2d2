"""
AirSim Drone Control Script
Test script for drone control and fire detection using AirSim simulator.
"""

from drone.airsim_drone import AirsimDrone
from utils.image import display_img
from detection.YOLOv8.model import YOLOv8
from time import sleep
import math
import numpy as np

# Configuration constants
WEIGHTS_FILE = 'yolov8n_wildfire_as_e100_new_2.pt'
DRONE_CONFIG = {
    'initial_position': {'x': 0, 'y': 0, 'z': -30},
    'camera_angle': -90,
    'target_size': [736, 1280]
}

def setup_environment(drone: AirsimDrone):
    """
    Initialize drone and environment settings.
    
    Args:
        drone: AirsimDrone instance to configure
    """
    # Position drone and set camera angle
    drone.move_to(**DRONE_CONFIG['initial_position'])
    drone.set_cam_pose(pitch=DRONE_CONFIG['camera_angle'])
    
    # Move fire object to ground level
    fire_x, fire_y, _ = drone.get_obj_pos('FiremeshBP')
    drone.move_obj_to('FiremeshBP', fire_x, fire_y, 0)
    sleep(2)  # Allow time for object to settle

def run_fire_detection(drone: AirsimDrone, detector: YOLOv8):
    """
    Capture image and run fire detection.
    
    Args:
        drone: AirsimDrone instance to capture image from
        detector: YOLOv8 model for fire detection
    """
    # Capture and process image
    drone_img = drone.capture_cam_img()
    drone_img_copy = drone_img.copy()
    
    # Run detection
    detections = detector.detect(drone_img, DRONE_CONFIG['target_size'])
    
    # Visualize results
    drone_img_with_detections = detector.visualize_detections(drone_img_copy, detections)
    display_img(drone_img_with_detections)

def main():
    """Main execution function"""
    # Initialize detector and drone
    detector = YOLOv8(WEIGHTS_FILE)
    drone = AirsimDrone()
    
    # Setup environment
    setup_environment(drone)
    
    # Run detection
    run_fire_detection(drone, detector)

if __name__ == '__main__':
    main()

"""
Additional functionality:

# Drone movement calculations
# drone_height = 50
# cell_w = 2 * drone_height * math.tan(np.deg2rad(abs(drone_cam_angle)/2)) 
# cell_h = cell_w * drone.img_height / drone.img_width
# print(cell_w, cell_h)    
# drone_x, drone_y, drone_z, drone_pitch, drone_yaw, _ = drone.get_pose()

# Movement commands
# drone.move_to(drone_x, drone_y + cell_w/3, drone_z, 20)
# drone.set_cam_orientation(drone_cam_angle, 90, 0)
# drone.move_to(drone_x - cell_h/3, drone_y + cell_w/3, drone_z, 20)
# drone.move_to(drone_x, drone_y + drone_height * 2/3, drone_z, 20)

# Camera orientation tests
# drone.set_cam_pose(-90, -90, 0)
# drone.set_cam_pose(-90, 90, 0)
"""