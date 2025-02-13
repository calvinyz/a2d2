"""
Grid Search Optimizer
Performs grid search to find optimal drone height and camera angle for fire detection.
"""

import math
import numpy as np
from time import sleep
from os.path import join, dirname, abspath
from drone.airsim_drone import AirsimDrone
from detection.YOLOv8.model import YOLOv8
from utils.image import save_img

# Configuration constants
CAMERA_CONSTRAINTS = {
    'max_angle': -60,
    'min_height': 85
}

class GridSearchOptimizer:
    """Optimizes drone parameters through grid search."""
    
    def __init__(
        self, 
        max_height: int,
        min_cam_angle: int,
        n_cell_x: int,
        n_cell_y: int,
        yolo_weights: str,
        detect: bool = False
    ):
        """
        Initialize the grid search optimizer.
        
        Args:
            max_height: Maximum drone height to test
            min_cam_angle: Minimum camera angle to test
            n_cell_x: Number of grid cells in x direction
            n_cell_y: Number of grid cells in y direction
            yolo_weights: Path to YOLO model weights
            detect: Whether to run detection during search
        """
        # Set camera angle constraints
        self.max_cam_angle = CAMERA_CONSTRAINTS['max_angle']
        self.min_cam_angle = (CAMERA_CONSTRAINTS['max_angle'] + 5 
                            if min_cam_angle <= CAMERA_CONSTRAINTS['max_angle']
                            else min_cam_angle)
        
        # Set height constraints
        self.max_height = max_height
        self.min_height = CAMERA_CONSTRAINTS['min_height']
        
        # Grid configuration
        self.n_cell_x = n_cell_x
        self.n_cell_y = n_cell_y
        
        # Step sizes
        self.height_step = -5
        self.cam_angle_step = 5
        
        # Initialize components
        self.drone = AirsimDrone()
        self.detector = YOLOv8(yolo_weights)
        self.detect = detect
        
        # Add result tracking
        self.best_height = None
        self.best_angle = None
        self.best_detection_score = float('-inf')

    def set_height_step(self, step: int):
        """Set the step size for height search."""
        self.height_step = step
    
    def set_cam_angle_step(self, step: int):
        """Set the step size for camera angle search."""
        self.cam_angle_step = step

    def grid_search(self):
        """
        Perform grid search over height and camera angle combinations.
        Captures images at each position and optionally runs detection.
        """
        cam_fov = self.drone.get_cam_fov()
        
        for height in range(self.max_height, self.min_height, self.height_step):
            # Calculate grid cell dimensions
            gca_x = 2 * height  # Horizontal FOV of 90 degrees
            gca_y = self.drone.img_height / self.drone.img_width * gca_x
            
            # Calculate half dimensions
            gca_x_half = gca_x / 2
            gca_y_half = gca_y / 2
            
            # Calculate cell sizes
            cell_x = gca_x / self.n_cell_x
            cell_y = gca_y / self.n_cell_y
            cell_x_half = cell_x / 2
            cell_y_half = cell_y / 2
            
            for pitch in range(self.max_cam_angle, self.min_cam_angle, self.cam_angle_step):
                # Set camera pitch
                self.drone.set_cam_pitch(pitch)
                
                # Calculate x offset based on pitch
                offset_x = (height / math.tan((pitch - cam_fov/2) * math.pi / 180) - gca_y_half 
                          if pitch != -90 else 0)
                
                # Move drone to position
                self.drone.move_to([offset_x, 1, -height])
                
                # Test each grid cell
                self._test_grid_cells(height, pitch, gca_x_half, gca_y_half, 
                                    cell_x_half, cell_y_half)

    def _test_grid_cells(self, height, pitch, gca_x_half, gca_y_half, 
                        cell_x_half, cell_y_half):
        """Test fire detection in each grid cell."""
        for j in range(self.n_cell_y):
            for i in range(self.n_cell_x):
                # Calculate cell center
                c_x = gca_x_half - ((self.n_cell_x - 1 - i) * 2 + 1) * cell_x_half
                c_y = gca_y_half - (j * 2 + 1) * cell_y_half
                
                # Move fire object
                self.drone.move_obj_to('FiremeshBP', [c_y, c_x, -10])
                
                # Wait for object to settle
                sleep(6 if not i and not j else 2.5)
                
                # Capture and save image
                self._capture_and_process_image(height, pitch, i, j)

    def _capture_and_process_image(self, height, pitch, i, j):
        """Capture, save, and optionally process image with detection."""
        # Capture image
        cam_img = self.drone.capture_cam_img()
        
        # Generate filename
        img_filename = f'fire_{height}_{pitch}_{i + j * self.n_cell_x}.png'
        img_path = join(dirname(abspath(__file__)), 'images', img_filename)
        
        # Save original image
        save_img(cam_img, img_path)
        
        # Run detection if enabled
        if self.detect:
            cam_img_copy = np.copy(cam_img)
            detections = self.detector.detect(cam_img, [736, 1280])
            
            # Track best parameters based on detection confidence
            if detections and len(detections[0].boxes.conf) > 0:
                detection_score = float(detections[0].boxes.conf.mean())
                if detection_score > self.best_detection_score:
                    self.best_detection_score = detection_score
                    self.best_height = height
                    self.best_angle = pitch
            
            cam_img_with_detections = self.detector.visualize_detections(cam_img_copy, detections)
            
            # Save detection results
            detection_path = join(dirname(abspath(__file__)), 
                                'images_detections', 
                                f'{img_filename[:-4]}-res.png')
            save_img(cam_img_with_detections, detection_path)

def main():
    """Main execution function."""
    # Configuration
    yolo_weights = 'yolov8n_wildfire_as_e100_new.pt'
    weights_path = join(dirname(abspath(__file__)), yolo_weights)
    
    # Initialize and run optimizer
    optimizer = GridSearchOptimizer(
        max_height=120,
        min_cam_angle=-55,
        n_cell_x=5,
        n_cell_y=5,
        yolo_weights=weights_path,
        detect=True
    )
    
    # Run grid search
    params = optimizer.grid_search()
    print(params)

if __name__ == '__main__':
    main()

