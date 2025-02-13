"""
E-Spiral Coverage Path Planning with Wildfire Detection
Implements continuous wildfire detection during spiral coverage path execution.
"""

import threading
from time import sleep

from coverage_path.e_spiral.model import ESpiralCPP
from drone.airsim_drone import AirsimDrone
from detection.YOLOv8.model import YOLOv8
from utils.image import display_img
from alert.alert_manager import AlertManager, AlertPriority, AlertType

def detect_wildfire(drone: AirsimDrone, detector: YOLOv8, alert_manager: AlertManager):
    """
    Monitor for wildfire detection.
    
    Args:
        drone: AirSim drone instance
        detector: YOLOv8 detector instance
        alert_manager: Alert manager instance
    """
    print("Starting wildfire detection")
    consecutive_detections = 0
    detection_threshold = 3
    
    while consecutive_detections < detection_threshold:
        sleep(1)
        
        # Capture and process image
        img = drone.capture_cam_img()
        detections = detector.detect(img, [drone.img_height, drone.img_width])
        
        # Check for valid detections
        if detections and len(detections[0].boxes.xyxy) > 0:
            consecutive_detections += 1
            print(f"Found {len(detections[0].boxes.xyxy)} detections")
            
            # Handle confirmed detection
            if consecutive_detections >= detection_threshold:
                alert_manager.send_alert(
                    "Wildfire confirmed - stopping patrol",
                    AlertPriority.CRITICAL,
                    AlertType.ALL
                )
                
                # Stop patrol and hover
                drone.cancel_last_task()
                drone.hover()
                
                # Display detection
                img_dets = img.copy()
                img_dets = detector.visualize_detections(img_dets, detections)
                display_img(img_dets)
                break
        else:
            consecutive_detections = 0

if __name__ == '__main__':
    # Initialize components
    alert_manager = AlertManager({})
    weights_file = 'yolov8n_wildfire_as_e100_new_2.pt'
    yolo_detector = YOLOv8(weights_file)
    
    # Setup drone
    altitude = 70
    camera_angle = -45
    
    airsim_drone = AirsimDrone(
        start_pos=[0, 0, -altitude],
        cam_angle=camera_angle,
        max_distance=1000,
        alert_manager=alert_manager
    )
    
    # Move fire mesh to initial position
    airsim_drone.move_obj_to("FiremeshBP", 0, 0, 20)
    
    try:
        # Calculate patrol path
        cpp = ESpiralCPP(airsim_drone, altitude, camera_angle)
        waypoints = cpp.calculate_waypoints()
        print(f'Max distance: {airsim_drone.max_distance}, Waypoints: {waypoints}')
        
        # Start patrol
        airsim_drone.move_on_path_async(waypoints, 10)
        
        # Start detection thread
        detection_thread = threading.Thread(
            target=detect_wildfire,
            args=(airsim_drone, yolo_detector, alert_manager)
        )
        detection_thread.start()
        
        # Simulate fire movement
        sleep(10)
        airsim_drone.move_obj_to("FiremeshBP", 0, 0, -2)
        
    except Exception as e:
        alert_manager.send_alert(
            f"Mission failed: {str(e)}",
            AlertPriority.CRITICAL,
            AlertType.ALL
        )
        airsim_drone.emergency_land()

    

