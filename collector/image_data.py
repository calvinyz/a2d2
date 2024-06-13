import math
import os
from utils.image import save_img
from os.path import join
from datetime import datetime
from time import sleep

class ImageDataCollector:

    def __init__(self, alt_range, alt_step, 
                 cam_angle_range, cam_angle_step, 
                 n_x, n_y, drone):        
        self.alt_range = alt_range
        self.alt_step = alt_step
        self.ca_range = cam_angle_range
        self.ca_step = cam_angle_step
        self.n_x = n_x
        self.n_y = n_y
        self.drone = drone

    def collect_imgs(self, img_dir='images'):
        dt = datetime.now().strftime('%H%M%S')
        img_dir = join(img_dir, f'{dt}')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)

        for t in range(self.alt_range[0], self.alt_range[1], self.alt_step):
            gcc_x = 2 * t
            gcc_y = gcc_x * self.drone.img_height /self.drone.img_width
            gcc_x_hf, gcc_y_hf = gcc_x / 2, gcc_y / 2
            cell_x, cell_y = gcc_x / self.n_x, gcc_y / self.n_y
            cell_x_hf, cell_y_hf = cell_x / 2, cell_y / 2
            for a in range(self.ca_range[0], self.ca_range[1], self.ca_step):
                self.drone.set_cam_pose(0, 0, 0.5, pitch=a)
                offset_x = gcc_x_hf + t * math.tan(math.radians(90 - abs(a) - self.drone.fov_h/2))
                offset_y = gcc_y_hf + t * math.tan(math.radians(90 - abs(a) - self.drone.fov_v/2))
                for j in range(self.n_y):
                    for i in range(self.n_x):
                        pos_x = i * cell_x + cell_x_hf - gcc_x_hf
                        pos_y = j * cell_y + cell_y_hf - gcc_y_hf - offset_y
                        self.drone.move_to(pos_y, pos_x, -t)
                        drone_img = self.drone.capture_cam_img()                        
                        # Make sure image folder is available
                        img_filename = f'img-{t}_{-a}_{j}_{i}-{dt}.png'
                        print(f'Saving {img_filename}')
                        save_img(drone_img, join(img_dir, img_filename))
