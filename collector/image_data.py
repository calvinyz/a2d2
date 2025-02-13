"""
Image Data Collector with Real-time Detection
Collects drone camera images and performs real-time smoke detection.
"""

import math
import os
from utils.image import save_img, display_img
from os.path import join
from datetime import datetime
from time import sleep
from typing import Tuple, List, Optional
from alert.alert_manager import AlertManager, AlertPriority, AlertType
from detection.YOLOv8.model import YOLOv8
from drone.airsim_drone import AirsimDrone  # Import AirsimDrone directly

class ImageDataCollector:
    """
    Systematic image data collection from drone camera.
    
    Collects images across a grid of positions while varying altitude
    and camera angle to build a comprehensive dataset.
    """

    def __init__(
        self,
        alt_range: Tuple[int, int],
        alt_step: int,
        cam_angle_range: Tuple[int, int],
        cam_angle_step: int,
        n_x: int,
        n_y: int,
        drone: AirsimDrone,  # Change type hint to AirsimDrone
        alert_manager: AlertManager,
        detector: Optional[YOLOv8] = None
    ):
        """
        Initialize data collector.
        
        Args:
            alt_range: (min_altitude, max_altitude) in meters
            alt_step: Altitude increment step
            cam_angle_range: (min_angle, max_angle) in degrees
            cam_angle_step: Camera angle increment step
            n_x: Number of grid points in x direction
            n_y: Number of grid points in y direction
            drone: Drone control object
            alert_manager: Alert manager object
            detector: Optional YOLOv8 detector
        """
        self.alt_range = alt_range
        self.alt_step = alt_step
        self.ca_range = cam_angle_range
        self.ca_step = cam_angle_step
        self.n_x = n_x
        self.n_y = n_y
        self.drone = drone
        self.alert_manager = alert_manager
        self.detector = detector

    def collect_imgs(self, img_dir: str = 'images', detect_smoke: bool = False):
        """
        Collect images with optional smoke detection.
        
        Args:
            img_dir: Base directory for saving images
            detect_smoke: Whether to run smoke detection
        """
        try:
            # Create output directory
            timestamp = datetime.now().strftime('%H%M%S')
            output_dir = join(img_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)

            for altitude in range(self.alt_range[0], self.alt_range[1], self.alt_step):
                # Check drone status directly
                status = self.drone.get_status()
                if status["status"] == "ERROR":  # Use string directly
                    raise RuntimeError(f"Drone in unsafe state: {status['status']}")

                for angle in range(self.ca_range[0], self.ca_range[1], self.ca_step):
                    if not self.drone.set_cam_pose(0, angle, 0):
                        continue

                    # Calculate parameters
                    gcc = self._calculate_ground_coverage(altitude)
                    offsets = self._calculate_offsets(altitude, angle)
                    
                    # Collect grid images with detection
                    self._collect_grid_images(
                        altitude,
                        angle,
                        gcc,
                        offsets,
                        output_dir,
                        timestamp,
                        detect_smoke
                    )

        except Exception as e:
            self.drone.alert_manager.send_alert(
                f"Image collection failed: {str(e)}",
                AlertPriority.HIGH,
                AlertType.ALL
            )
            raise

    def _calculate_ground_coverage(self, altitude: int) -> dict:
        """
        Calculate ground coverage parameters.
        
        Args:
            altitude: Current altitude
            
        Returns:
            dict: Ground coverage parameters
        """
        gcc_x = 2 * altitude
        gcc_y = gcc_x * self.drone.img_height / self.drone.img_width
        
        return {
            'x': gcc_x,
            'y': gcc_y,
            'x_half': gcc_x / 2,
            'y_half': gcc_y / 2,
            'cell_x': gcc_x / self.n_x,
            'cell_y': gcc_y / self.n_y
        }

    def _calculate_offsets(self, altitude: int, angle: int) -> dict:
        """
        Calculate camera view offsets.
        
        Args:
            altitude: Current altitude
            angle: Current camera angle
            
        Returns:
            dict: Offset values
        """
        return {
            'x': (altitude * math.tan(
                math.radians(90 - abs(angle) - self.drone.fov_h/2)
            )),
            'y': (altitude * math.tan(
                math.radians(90 - abs(angle) - self.drone.fov_v/2)
            ))
        }

    def _collect_grid_images(
        self,
        altitude: int,
        angle: int,
        gcc: dict,
        offsets: dict,
        output_dir: str,
        timestamp: str,
        detect_smoke: bool = False
    ):
        """Collect grid images with optional detection."""
        cell_x_half = gcc['cell_x'] / 2
        cell_y_half = gcc['cell_y'] / 2
        
        for j in range(self.n_y):
            for i in range(self.n_x):
                # Calculate position
                pos_x = (i * gcc['cell_x'] + cell_x_half - gcc['x_half'])
                pos_y = (j * gcc['cell_y'] + cell_y_half - gcc['y_half'] 
                        - gcc['y_half'] - offsets['y'])
                
                # Move drone
                if not self.drone.move_to(pos_y, pos_x, -altitude):
                    continue

                # Capture image
                image = self.drone.capture_cam_img()
                
                # Run detection if enabled
                if detect_smoke and self.detector:
                    detections = self.detector.detect(
                        image,
                        [self.drone.img_height, self.drone.img_width]
                    )
                    if detections and len(detections[0].boxes.xyxy) > 0:
                        # Visualize and save detection
                        image_with_dets = image.copy()
                        image_with_dets = self.detector.visualize_detections(
                            image_with_dets,
                            detections
                        )
                        
                        # Alert about detection
                        self.alert_manager.send_alert(
                            f"Smoke detected at altitude={altitude}m, angle={angle}°",
                            AlertPriority.HIGH,
                            AlertType.ALL
                        )
                        
                        # Save both original and annotated images
                        det_filename = f'det-{altitude}_{-angle}_{j}_{i}-{timestamp}.png'
                        save_img(image_with_dets, join(output_dir, det_filename))

                # Save original image
                filename = f'img-{altitude}_{-angle}_{j}_{i}-{timestamp}.png'
                save_img(image, join(output_dir, filename))
