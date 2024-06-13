import math
from time import sleep
from drone.airsim_drone import AirsimDrone
from detection.YOLOv8.model import YOLOv8
from utils.image import save_img
from os.path import join, dirname, abspath
import numpy as np

MAX_CAM_ANGLE = -60
MIN_HEIGHT = 85

class GridSearchOptimizer:
    def __init__(self, max_height, min_cam_angle, n_cell_x, n_cell_y, yolo_weights, detect=False):
        self.max_height = max_height
        self.min_height = MIN_HEIGHT
        if min_cam_angle <= MAX_CAM_ANGLE:
            self.max_cam_angle = MAX_CAM_ANGLE
            self.min_cam_angle = MAX_CAM_ANGLE + 5
        else:
            self.max_cam_angle = MAX_CAM_ANGLE 
            self.min_cam_angle = min_cam_angle
        self.n_cell_x = n_cell_x
        self.n_cell_y = n_cell_y
        self.drone = AirsimDrone()        
        self.detector = YOLOv8(yolo_weights)
        self.detect = detect
        self.height_step = -5
        self.cam_angle_step = 5

    def set_height_step(self, step):
        self.height_step = step
    
    def set_cam_angle_step(self, step):
        self.cam_angle_step = step    

    def grid_search(self):

        cam_fov = self.drone.get_cam_fov()
        for h in range(self.max_height, self.min_height, self.height_step):
            gca_x = 2 * h # Horizontal FOV of 90 degrees
            gca_y = self.drone.img_height /self.drone.img_width * gca_x
            gca_x_h = gca_x / 2
            gca_y_h = gca_y / 2
            cell_x = gca_x / self.n_cell_x
            cell_y = gca_y / self.n_cell_y
            cell_x_h = cell_x / 2
            cell_y_h = cell_y / 2
            for p in range(self.max_cam_angle, self.min_cam_angle, self.cam_angle_step):
                self.drone.set_cam_pitch(p)
                offset_x = h/math.tan((p  - cam_fov/2)* math.pi / 180) - gca_y_h if p != -90 else 0
                self.drone.move_to([offset_x, 1, -h]) # Check offset sign/direction
                for j in range(self.n_cell_y):
                    for i in range(self.n_cell_x):
                        c_x = gca_x_h - ((self.n_cell_x - 1 - i) * 2 + 1) * cell_x_h
                        c_y = gca_y_h - (j * 2 + 1) * cell_y_h
                        self.drone.move_obj_to('FiremeshBP', [c_y, c_x, -10])
                        if not i and not j:
                            sleep(6)
                        else:
                            sleep(2.5)
                        cam_img = self.drone.capture_cam_img()
                        cam_img_file = f'fire_{h}_{p}_{i + j * self.n_cell_x}.png'
                        save_img(cam_img, join(dirname(abspath(__file__)), 'images\\', cam_img_file))
                        cam_img_copy = np.copy(cam_img)

                        if self.detect:
                            res = self.detector.detect(cam_img, [736, 1280])
                            cam_img_res = self.detector.visualize_detections(cam_img_copy, res)
                            save_img(cam_img_res, join(dirname(abspath(__file__)), 'images_detections\\', cam_img_file[:-4] + '-res.png'))

                        # Add results to a list

        # Return the best h, p with most detections
        # Create a composite image with all results visualized with green found and red not found
                        
    # Retrain yolo with the new dataset with smaller fire images


if __name__ == '__main__':
    yolo_weights = 'yolov8n_wildfire_as_e100_new.pt'
    yolo_weights_path = join(dirname(abspath(__file__)), yolo_weights)
    params_opt = GridSearchOptimizer(120, -55, 5, 5, yolo_weights_path, True)
    params = params_opt.grid_search()
    print(params)

