import threading
from time import sleep

from pathplanner.coverage.ESpiral.model import ESpiralCPP
from drone.airsim_drone import AirsimDrone
from detector.YOLOv8.model import YOLOv8
from utils.image import display_img

def detect_wildfire(drone, detector):
    print("Detecting wildfire")
    dets = 0
    while dets < 3:
        sleep(1)
        img = drone.capture_cam_img()
        detections = detector.detect(img, [drone.img_height, drone.img_width])
        if detections and len(detections) > 0 and len(detections[0].boxes.xyxy) > 0:
            # setattr(t_detect, "detected", True)
            print(f"Found {len(detections[0].boxes.xyxy)} detections. Cancel task")
            dets += 1
            if dets >= 3:
                print("Detected 3 times. Stop patrolling.")
                drone.cancel_last_task()
                drone.hover()
                img_dets = img.copy()
                img_dets = detector.visualize_detections(img_dets, detections)
                display_img(img_dets)                
                break
        else:
            dets = 0


if __name__ == '__main__':
    weights_file = 'yolov8n_wildfire_as_e100_new_2.pt'
    yolo_detector = YOLOv8(weights_file)
    
    altitude = 70
    camera_angle = -45

    airsim_drone = AirsimDrone(start_pos=[0, 0, -altitude], cam_angle=camera_angle, max_distance=1000)
    airsim_drone.move_obj_to("FiremeshBP", 0, 0, 20)

    cpp = ESpiralCPP(airsim_drone, altitude, camera_angle)
    cpp_wps = cpp.calculate_waypoints()
    print(f'Max distance: {airsim_drone.max_distance}, Waypoints: {cpp_wps}')

    # print(f'Waypoints steps: {cpp_wps_steps}')
    airsim_drone.move_on_path_async(cpp_wps, 10)

    t_detect_wildfire = threading.Thread(target=detect_wildfire, args=(airsim_drone, yolo_detector))
    t_detect_wildfire.start()

    sleep(10)
    airsim_drone.move_obj_to("FiremeshBP", 0, 0, -2)

    

