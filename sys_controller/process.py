"""
System Process Controller
Manages the complete workflow of the A2D2 system, including data collection,
model training, path planning, and fire detection.
"""

import threading
from time import sleep
from typing import Optional
from datetime import datetime
from pathlib import Path

from optimization.grid_search import GridSearchOptimizer, CAMERA_CONSTRAINTS
from coverage_path.e_spiral.model import ESpiralCPP
from detection.YOLOv8.model import YOLOv8
from drone.airsim_drone import AirsimDrone
from alert.alert_manager import AlertManager, AlertPriority, AlertType
from ipp_dqn import DQNController
from utils.image import display_img
from collector.image_data import ImageDataCollector

class ProcessController:
    def __init__(self, config: dict):
        """Initialize process controller with configuration."""
        # Store config
        self.config = config
        
        # Initialize alert manager
        self.alert_manager = AlertManager(config.get('alerts', {}))
        
        # Initialize drone with alert manager
        self.drone = AirsimDrone(
            start_pos=config['drone']['start_pos'],
            cam_angle=config['drone']['camera_angle'],
            max_distance=config['drone']['max_distance'],
            alert_manager=self.alert_manager
        )
        
        # Initialize components with direct drone reference
        self.detector = YOLOv8(config['detector']['weights_file'])
        
        self.cpp = ESpiralCPP(
            self.drone,  # Pass AirsimDrone directly
            config['patrol']['altitude'],
            config['patrol']['camera_angle']
        )
        
        # Initialize inspection controller with DQN
        self.ipp = DQNController(
            self.drone,
            self.detector,
            config['inspection']
        )
        
        # Initialize grid search with existing parameters
        self.optimizer = GridSearchOptimizer(
            max_height=config['optimization']['max_height'],
            min_cam_angle=config['optimization']['min_cam_angle'],
            n_cell_x=config['optimization']['grid_cells_x'],
            n_cell_y=config['optimization']['grid_cells_y'],
            yolo_weights=config['detector']['weights_file'],
            detect=True
        )
        
        # Add optimization state
        self.optimized_params = None
        
        # State tracking
        self.mission_active = False
        self.detection_count = 0
        self.detection_threshold = config.get('detection_threshold', 3)
        
        # Initialize image collector
        self.collector = ImageDataCollector(
            alt_range=config['collection']['altitude_range'],
            alt_step=config['collection']['altitude_step'],
            cam_angle_range=config['collection']['angle_range'],
            cam_angle_step=config['collection']['angle_step'],
            n_x=config['collection']['grid_x'],
            n_y=config['collection']['grid_y'],
            drone=self.drone,
            alert_manager=self.alert_manager,
            detector=self.detector
        )
        
    def optimize_parameters(self):
        """Run grid search to optimize drone parameters."""
        try:
            print("Starting parameter optimization...")
            
            # Configure step sizes if provided
            if 'height_step' in self.config['optimization']:
                self.optimizer.set_height_step(self.config['optimization']['height_step'])
            if 'angle_step' in self.config['optimization']:
                self.optimizer.set_cam_angle_step(self.config['optimization']['angle_step'])
            
            # Run grid search
            self.optimizer.grid_search()
            
            # Store results
            self.optimized_params = {
                'altitude': self.optimizer.best_height,
                'camera_angle': self.optimizer.best_angle
            }
            
            # Update system with optimized parameters
            self._apply_optimized_params()
            
            self.alert_manager.send_alert(
                f"Parameter optimization complete: {self.optimized_params}",
                AlertPriority.LOW
            )
            
        except Exception as e:
            self.alert_manager.send_alert(
                f"Parameter optimization failed: {str(e)}",
                AlertPriority.HIGH
            )
            raise
            
    def _apply_optimized_params(self):
        """Apply optimized parameters to system components."""
        if not self.optimized_params:
            return
            
        # Update drone parameters
        self.drone.set_altitude(self.optimized_params['altitude'])
        self.drone.set_camera_angle(self.optimized_params['camera_angle'])
        
        # Update path planner
        self.cpp = ESpiralCPP(
            self.drone,
            self.optimized_params['altitude'],
            self.optimized_params['camera_angle']
        )
        
    def run_mission(self):
        """Execute complete mission sequence."""
        try:
            # Run optimization if not done
            if not self.optimized_params:
                self.optimize_parameters()
            
            self.mission_active = True
            
            # 1. Take off and setup
            self._setup_mission()
            
            # 2. Start patrol with spiral coverage
            waypoints = self.cpp.calculate_waypoints()
            print(f'Max distance: {self.drone.max_distance}, Waypoints: {waypoints}')
            
            # Start patrol execution
            self.drone.move_on_path_async(waypoints, 10)
            
            # 3. Start detection thread
            detection_thread = threading.Thread(target=self._monitor_detection)
            detection_thread.daemon = True
            detection_thread.start()
            
            # 4. Monitor mission
            while self.mission_active:
                sleep(1)
                if not detection_thread.is_alive():
                    break
                    
            # 5. Return home
            self._complete_mission()
            
        except Exception as e:
            self.alert_manager.send_alert(
                f"Mission failed: {str(e)}",
                AlertPriority.CRITICAL,
                AlertType.ALL
            )
            self._handle_emergency()
            
    def _setup_mission(self):
        """Prepare for mission execution."""
        try:
            # Remove monitor thread - no longer needed
            self.drone.move_obj_to("FiremeshBP", 0, 0, 20)
        except Exception as e:
            self.alert_manager.send_alert(
                f"Mission setup failed: {str(e)}",
                AlertPriority.HIGH
            )
            raise
            
    def _monitor_detection(self):
        """Monitor for smoke detection."""
        while self.mission_active:
            try:
                # Capture and process image
                image = self.drone.capture_cam_img()
                detections = self.detector.detect(
                    image,
                    [self.drone.img_height, self.drone.img_width]
                )
                
                # Check for valid detections
                if detections and len(detections[0].boxes.xyxy) > 0:
                    self.detection_count += 1
                    print(f"Found {len(detections[0].boxes.xyxy)} detections")
                    
                    # Handle confirmed detection
                    if self.detection_count >= self.detection_threshold:
                        self._handle_confirmed_detection(image, detections)
                        break
                else:
                    self.detection_count = 0
                    
                sleep(1)
                
            except Exception as e:
                self.alert_manager.send_alert(
                    f"Detection error: {str(e)}",
                    AlertPriority.HIGH
                )
                
    def _handle_confirmed_detection(self, image, detections):
        """Handle confirmed smoke detection."""
        try:
            self.alert_manager.send_alert(
                "Smoke detection confirmed - starting inspection",
                AlertPriority.HIGH,
                AlertType.ALL
            )
            
            # Pause patrol
            self.drone.cancel_last_task()
            self.drone.hover()
            
            # Save detection images using collector
            timestamp = datetime.now().strftime('%H%M%S')
            detection_dir = f'data/detections/{timestamp}'
            
            # Collect detailed images of detection area
            self.collector.collect_imgs(
                img_dir=detection_dir,
                detect_smoke=True
            )
            
            # Display initial detection
            image_with_dets = image.copy()
            image_with_dets = self.detector.visualize_detections(
                image_with_dets,
                detections
            )
            display_img(image_with_dets)
            
            # Start DQN-based inspection
            inspection_result = self.ipp.run_inspection(
                max_steps=self.config['inspection']['max_steps'],
                epsilon=self.config['inspection'].get('epsilon', 0.1),
                learning_rate=self.config['inspection'].get('learning_rate', 0.001),
                batch_size=self.config['inspection'].get('batch_size', 32),
                gamma=self.config['inspection'].get('gamma', 0.99)
            )
            
            # Handle inspection result
            if inspection_result['fire_confirmed']:
                self.alert_manager.send_alert(
                    f"Fire confirmed at location {inspection_result['fire_location']}",
                    AlertPriority.CRITICAL,
                    AlertType.ALL
                )
            else:
                self.alert_manager.send_alert(
                    "Inspection complete - no fire confirmed",
                    AlertPriority.MEDIUM
                )
                self.drone.resume_task()
            
        except Exception as e:
            self.alert_manager.send_alert(
                f"Inspection failed: {str(e)}",
                AlertPriority.HIGH
            )
            
    def _complete_mission(self):
        """Complete mission and return home."""
        try:
            self.mission_active = False
            self.drone.return_to_home()
            
            self.alert_manager.send_alert(
                "Mission complete - returned home",
                AlertPriority.LOW
            )
            
        except Exception as e:
            self.alert_manager.send_alert(
                f"Return failed: {str(e)}",
                AlertPriority.HIGH
            )
            
    def _handle_emergency(self):
        """Handle emergency situation."""
        try:
            self.mission_active = False
            self.drone.hover()
            self.drone.emergency_land()
            
        except Exception as e:
            self.alert_manager.send_alert(
                f"Emergency handling failed: {str(e)}",
                AlertPriority.CRITICAL,
                AlertType.ALL
            )

    def collect_training_data(self, output_dir: str = 'data/images'):
        """Collect training data using grid-based collection."""
        try:
            print("Starting data collection...")
            
            # Run collection with detection
            self.collector.collect_imgs(
                img_dir=output_dir,
                detect_smoke=True  # Enable real-time detection
            )
            
            self.alert_manager.send_alert(
                f"Data collection complete in {output_dir}",
                AlertPriority.LOW
            )
            
        except Exception as e:
            self.alert_manager.send_alert(
                f"Data collection failed: {str(e)}",
                AlertPriority.HIGH,
                AlertType.ALL
            )
            raise

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'drone'):
                self.drone.emergency_land()
                self.drone.client.enableApiControl(False)
        except:
            pass

