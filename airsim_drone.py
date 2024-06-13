from drone.airsim_drone import AirsimDrone
from utils.image import display_img
from time import sleep
import math
import numpy as np
from detection.YOLOv8.model import YOLOv8

if __name__ == '__main__':
    weights_file = 'yolov8n_wildfire_as_e100_new_2.pt'

    detector = YOLOv8(weights_file)

    drone = AirsimDrone()
    # # drone.reset()
    # drone.set_velocity(50)
    # drone_height = 50
    drone_cam_angle = -90
    drone.move_to(0, 0, -30)
    drone.set_cam_pose(pitch=drone_cam_angle)
    # cell_w = 2 * drone_height * math.tan(np.deg2rad(abs(drone_cam_angle)/2)) 
    # cell_h = cell_w * drone.img_height / drone.img_width
    # print(cell_w, cell_h)    
    # drone_x, drone_y, drone_z, drone_pitch, drone_yaw, _ = drone.get_pose()

    fire_x, fire_y, fire_z = drone.get_obj_pos('FiremeshBP')

    drone.move_obj_to('FiremeshBP', fire_x, fire_y, 0)
    sleep(2)
    # fire_x, fire_y, fire_z = drone.get_obj_pos('FiremeshBP')
    # print(fire_x, fire_y, fire_z)

    drone_img = drone.capture_cam_img()
    # display_img(drone_img)
    
    # drone.move_to(drone_x, drone_y + cell_w/3, drone_z, 20)
    # # drone.set_cam_orientation(-45, 0, 0)
    # drone.set_cam_orientation(drone_cam_angle, 90, 0)

    # drone_img = drone.capture_cam_img()
    # display_img(drone_img)
    
    # drone.move_to(drone_x - cell_h/3, drone_y + cell_w/3, drone_z, 20)

    # drone_img = drone.capture_cam_img()
    # display_img(drone_img)  

    # drone.move_to(drone_x, drone_y + drone_height * 2/3, drone_z, 20)
    # # drone.set_cam_orientation(-90, 0, 0)
    # sleep(5)
    # drone_img = drone.capture_cam_img()
    # display_img(drone_img)

    # print(drone.get_obj_pos('FiremeshBP'))

    # res = drone.move_obj_to('FiremeshBP', [20, 20, -2])  
    # # sleep(3)
    # drone_img = drone.capture_cam_img()
    # display_img(drone_img)

    # # res = drone.move_obj_to('FiremeshBP', [30, 30, -2])
    # # sleep(3)
    # drone.set_cam_pose(-90, -90, 0)
    # drone_img = drone.capture_cam_img()
    # display_img(drone_img)

    # drone.set_cam_pose(-90, 90, 0)
    # drone_img = drone.capture_cam_img()
    # display_img(drone_img)

    drone_img_copy = drone_img.copy()

    res = detector.detect(drone_img, [736, 1280])
    drone_img_res = detector.visualize_detections(drone_img_copy, res)
    display_img(drone_img_res)