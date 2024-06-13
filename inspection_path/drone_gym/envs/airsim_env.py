from PIL import Image
import numpy as np
import gymnasium as gym
from gymnasium import spaces


OUT_OF_VIEW_COUNT_LIMIT = 2
BBOX_SIZE_THRESHOLD = 0.02
ACTION = {0: 'forward', 
          1: 'left', 
          2: 'right',
          3: 'down',
          4: 'left_forward',
          5: 'right_forward',
          6: 'left_down',
          7: 'right_down',
          8: 'forward_down'}

def is_detected(dets):
    return False if not dets or len(dets) == 0 or \
        len(dets[0].boxes.xyxy) == 0 else True

class DroneAirSimEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, drone, detector, step_length=2, velocity_duration=1.0, max_steps_episode=50, 
                 img_shape=(64, 64, 1)):
        self.viewer = None

        self.drone = drone
        self.detector = detector
        self.step_length = step_length
        self.velocity_duration = velocity_duration
        self.max_steps_episode = max_steps_episode

        self.img_width = drone.img_width
        self.img_height = drone.img_height
        self.img_size = self.img_width * self.img_height
        self.img_width_hlf = int(self.img_width / 2)
        self.img_height_hlf = int(self.img_height / 2)

        self.obs_width = int(drone.img_width / 4)
        self.obs_height = int(drone.img_height / 4)

        self.img_shape = img_shape

        self.observation_space = spaces.Box(0, 255, 
                                            shape=img_shape,
                                            dtype=np.uint8)
        self.action_space = spaces.Discrete(9)
        self._setup_flight()

    def render(self):
        return self._get_obs()

    def step(self, action):
        self._do_action(action)
        obs, info = self._get_obs()
        reward, done, truncated = self._compute_reward()
        
        return obs, reward, done, truncated, info

    def reset(self, seed=None): # Important: Add seed=None for gymnasium compatibility
        self._setup_flight()
        return self._get_obs()
    
    def close(self):
        # self.drone.reset()
        pass

    def _get_obs(self):
        img = self.drone.capture_cam_img()

        dets = self.detector.detect(img, [self.img_height, self.img_width]) if img is not None else []

        self.state['collision'] = self.drone.check_collision()
        self.state['steps_count'] += 1
        self.state['detections'] = dets

        if img is None:
            obs = np.zeros(list(self.img_shape))
        else:
            img_dets = img.copy()
            img_dets = self.detector.visualize_detections(img_dets, dets)
            obs = np.array(Image.fromarray(img_dets, 'RGB').resize((self.img_shape[0], self.img_shape[1])).convert("L"))
            obs = obs.reshape(list(self.img_shape))

        return obs, self.state
    
    def _do_action(self, action):
        # self.drone.hover_async()

        print(f'Action: {action} - {ACTION[action]}')

        # if action <= 3:
            # measure time taken to move
            
        self.drone.move(ACTION[action], 
                        self.step_length, 
                        self.velocity_duration)
            
        # else:
        #     self.drone.rotate(ACTION[action], 
        #                       self.step_length, 
        #                       self.velocity_duration)

    def _compute_reward(self):

        reward = 0
        done = False
        truncated = False

        if self.state['steps_count'] >= self.max_steps_episode:
            self.state['steps_count'] = 0
            truncated = True

        if self.state['collision']:
            reward = -100
            truncated = True
            print(f'Reward: {reward}. Truncated: {truncated} - Collision detected.')
        else:
            if is_detected(self.state['detections']):
                bboxs = self.state['detections'][0].boxes.xyxy
                bbox = bboxs[0]
                bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                bbox_size_ratio = bbox_size / self.img_size
                bbox_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                bbox_x_offset_ratio = 1 - abs((bbox_center[0] - self.img_width_hlf)/self.img_width_hlf)
                bbox_y_offset_ratio = 1 - abs((bbox_center[1] - self.img_height_hlf)/self.img_height_hlf)
        
                if bbox_size_ratio >= BBOX_SIZE_THRESHOLD: # cannot be too close to fire
                    done = True

                reward = bbox_size_ratio * 10_000 + bbox_x_offset_ratio * 50 + bbox_y_offset_ratio * 50
                print(f'Reward: {reward}, s: {bbox_size_ratio:.5f}, x: {bbox_x_offset_ratio:.2f}, y: {bbox_y_offset_ratio:.2f}. Done: {done}')
            else:
                reward = 0
                self.state['out_of_view_count'] += 1
                if self.state['out_of_view_count'] >= OUT_OF_VIEW_COUNT_LIMIT:
                    truncated = True
                print(f'Reward: {reward}. Truncated: {truncated} - Out of view count: {self.state["out_of_view_count"]}')

        return reward, done, truncated
    
    def _setup_flight(self):
        self.drone.reset()
        self.state = {"collision": False, 
                      "steps_count": 0,
                      "detections": [],
                      "out_of_view_count": 0}
