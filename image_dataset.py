from os.path import join, dirname, abspath

from collector.image_data import ImageDataCollector
from drone.airsim_drone import AirsimDrone

if __name__ == '__main__':
    altitude_min = 20
    altitude_max = 125
    altitude_step = 10
    camera_angle_min = -90
    camera_angle_max = -10
    camera_angle_step = 15
    fire_n_x = 3
    fire_n_y = 3
    drone = AirsimDrone()

    collector = ImageDataCollector([altitude_min, altitude_max], 
                                    altitude_step,
                                    [camera_angle_min, camera_angle_max], 
                                    camera_angle_step, 
                                    fire_n_x, fire_n_y, drone)
    collector.collect_imgs(join(dirname(abspath(__file__)), 'images'))