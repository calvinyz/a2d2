from collector.image_data import ImageDataCollector
from drone.airsim_drone import AirsimDrone
from detector.YOLOv8.model import YOLOv8
from optimization.grid_search import GridSearchOptimizer
from pathplanner.coverage.ESpiral.model import ESpiralCPP
from pathplanner.inspection.DDQN.model import DDQN_Agent
from alert.email import AlertNotifier
from utils.image import display_img

class SysProcController:
    def __init__(self, drone='Airsim',
                 detector='YOLOv8', 
                 optimizer='GridSearch', 
                 cpp='ESpiral', 
                 ipp='DDQN',
                 alert='a2d2alert@gmail.com'):
        if (drone != 'Airsim' or detector != 'YOLOv8' or 
            optimizer != 'GridSearch' or cpp != 'ESpiral' or ipp != 'DDQN'):
            raise NotImplementedError('Only YOLOv8|GridSearch|ESpiral|DDQN'
                                      'implemented')
        self.collector = ImageDataCollector()
        self.drone = AirsimDrone(max_distance=500)
        # self.optimizer = GridSearchOptimizer()
        self.detector = YOLOv8('yolov8n_wildfire_as_e100_new.pt')
        self.cpp = ESpiralCPP(-50, -45,
                              self.drone.max_distance, 
                              self.drone.fov_h, 
                              self.drone.img_width, 
                              self.drone.img_height) 
        # self.ipp = DDQN_Agent()
        self.alert = AlertNotifier(alert)
        self.init_modules()

    def init_modules(self):
        # Data collection for training detector
        self.collector.collect_imgs()
        
        # Train detector 
        # Train function input should be annotated dataset
        # Fix fire location and move drone around to generate dataset
        self.detector.train()

        # Gird search to find optimal drone height and camera angle
        # Fix fire location and move drone around to run grid search
        # Run input should be detector and drone
        self.optimizer.run()
        self.drone_height, self.cam_angle = self.optimizer.get_best_params()

        # Train inspection path planner (DRL)
        # Fix fire location and move drone start location for training
        # Repeat training so that the agent can learn to fly closer to the fire
        self.ipp.train()

        # Compute coverage path waypoints
        self.cpp_path = self.cpp.calculate_waypoints()

    def run(self):
        # Run the system
        # self.drone.reset()

        # Get current drone location
        # cur_loc = base_loc = self.drone.get_loc()

        while self.cpp.has_next_tp() and dist(cur_loc, self.next_tp) > self.drone.get_flyable_dist() - dist(self.cpp.next_tp, self.base_loc):
            # Get next turning point
            self.drone.move_to(next_tp)
            while self.drone.is_moving():
                detections = self.detetor.detect()
                if detections and detections[0].boxes.conf > 0.5:
                    # Run ipp to fly closer for inspection
                    next_ipp_loc = self.ipp.get_next_loc(detections)
                    while self.drone.flyable_dist > dist(cur_loc, next_ipp_loc) + dist(next_ipp_loc, self.drone.base_loc):
                        self.drone.move_to(next_ipp_loc)
                        cur_loc = next_ipp_loc
                        detections = self.detetor.detect()
                        if detections and detections[0].boxes.conf > 0.5:
                            self.alert.send_alert()
                            break
                        else:
                            next_ipp_loc = self.ipp.get_next_loc(detections)
    

    def e2e_test(self):
        frame_count = 0 
        while(frame_count < 1):
            drone_img = self.drone.capture_cam_img()
            # display_img(drone_img)
            detector = self.detector
            rets = detector.detect(drone_img, 
                [self.drone.img_height, self.drone.img_width])
            conf = rets[0].boxes.conf
            if conf.numel() and list(conf.size())[0] > 0:
                print('Fire detected')
                drone_img_dets = drone_img.copy()
                drone_img_dets_bbxs = self.detector.visualize_detections(drone_img_dets, rets)
                display_img(drone_img_dets_bbxs)
                self.alert.send_alert()

            frame_count += 1

