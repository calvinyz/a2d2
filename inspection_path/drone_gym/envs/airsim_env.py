"""
AirSim Drone Environment
Custom Gymnasium environment for drone control in AirSim simulator with fire detection capabilities.
"""

from PIL import Image
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Environment constants
ENVIRONMENT_CONFIG = {
    'out_of_view_limit': 2,
    'bbox_size_threshold': 0.02
}

# Action mapping
ACTIONS = {
    0: 'forward',
    1: 'left',
    2: 'right',
    3: 'down',
    4: 'left_forward',
    5: 'right_forward',
    6: 'left_down',
    7: 'right_down',
    8: 'forward_down'
}

def has_valid_detection(detections) -> bool:
    """Check if there are valid fire detections."""
    return not (not detections or len(detections) == 0 or 
               len(detections[0].boxes.xyxy) == 0)

class DroneAirSimEnv(gym.Env):
    """Gymnasium environment for drone control in AirSim."""
    
    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        drone,
        detector,
        step_length: int = 2,
        velocity_duration: float = 1.0,
        max_steps_episode: int = 50,
        img_shape: tuple = (64, 64, 1)
    ):
        """
        Initialize the drone environment.
        
        Args:
            drone: Drone controller instance
            detector: Object detection model
            step_length: Length of each movement step
            velocity_duration: Duration of velocity commands
            max_steps_episode: Maximum steps per episode
            img_shape: Shape of observation images
        """
        # Core components
        self.drone = drone
        self.detector = detector
        
        # Movement parameters
        self.step_length = step_length
        self.velocity_duration = velocity_duration
        self.max_steps_episode = max_steps_episode
        
        # Image dimensions
        self.img_width = drone.img_width
        self.img_height = drone.img_height
        self.img_size = self.img_width * self.img_height
        self.img_width_half = self.img_width // 2
        self.img_height_half = self.img_height // 2
        
        # Observation parameters
        self.obs_width = self.img_width // 4
        self.obs_height = self.img_height // 4
        self.img_shape = img_shape
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=img_shape,
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        
        # Initialize state
        self._setup_flight()

    def step(self, action):
        """Execute action and return new state."""
        self._do_action(action)
        obs, info = self._get_obs()
        reward, done, truncated = self._compute_reward()
        return obs, reward, done, truncated, info

    def reset(self, seed=None):
        """Reset environment to initial state."""
        self._setup_flight()
        return self._get_obs()

    def render(self):
        """Render current state."""
        return self._get_obs()

    def _get_obs(self):
        """Get current observation and state."""
        # Capture image
        img = self.drone.capture_cam_img()
        
        # Run detection if image is valid
        detections = (self.detector.detect(img, [self.img_height, self.img_width]) 
                     if img is not None else [])
        
        # Update state
        self.state.update({
            'collision': self.drone.check_collision(),
            'steps_count': self.state['steps_count'] + 1,
            'detections': detections
        })
        
        # Process observation image
        if img is None:
            obs = np.zeros(list(self.img_shape))
        else:
            img_with_detections = self.detector.visualize_detections(img.copy(), detections)
            obs = np.array(Image.fromarray(img_with_detections, 'RGB')
                         .resize((self.img_shape[0], self.img_shape[1]))
                         .convert("L"))
            obs = obs.reshape(list(self.img_shape))
        
        return obs, self.state

    def _do_action(self, action):
        """Execute the specified action."""
        print(f'Action: {action} - {ACTIONS[action]}')
        self.drone.move(
            ACTIONS[action],
            self.step_length,
            self.velocity_duration
        )

    def _compute_reward(self):
        """Compute reward based on current state."""
        reward = 0
        done = False
        truncated = False
        
        # Check step limit
        if self.state['steps_count'] >= self.max_steps_episode:
            self.state['steps_count'] = 0
            truncated = True
        
        # Check collision
        if self.state['collision']:
            reward = -100
            truncated = True
            print(f'Reward: {reward}. Truncated: {truncated} - Collision detected.')
            return reward, done, truncated
        
        # Process detections
        if has_valid_detection(self.state['detections']):
            reward, done = self._calculate_detection_reward()
        else:
            self.state['out_of_view_count'] += 1
            if self.state['out_of_view_count'] >= ENVIRONMENT_CONFIG['out_of_view_limit']:
                truncated = True
            print(f'Reward: {reward}. Truncated: {truncated} - '
                  f'Out of view count: {self.state["out_of_view_count"]}')
        
        return reward, done, truncated

    def _calculate_detection_reward(self):
        """Calculate reward based on detection quality."""
        bbox = self.state['detections'][0].boxes.xyxy[0]
        
        # Calculate bounding box metrics
        bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        bbox_size_ratio = bbox_size / self.img_size
        
        bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        bbox_x_offset_ratio = 1 - abs((bbox_center[0] - self.img_width_half) / self.img_width_half)
        bbox_y_offset_ratio = 1 - abs((bbox_center[1] - self.img_height_half) / self.img_height_half)
        
        # Determine completion
        done = bbox_size_ratio >= ENVIRONMENT_CONFIG['bbox_size_threshold']
        
        # Calculate reward
        reward = (bbox_size_ratio * 10_000 + 
                 bbox_x_offset_ratio * 50 + 
                 bbox_y_offset_ratio * 50)
        
        print(f'Reward: {reward}, s: {bbox_size_ratio:.5f}, '
              f'x: {bbox_x_offset_ratio:.2f}, y: {bbox_y_offset_ratio:.2f}. '
              f'Done: {done}')
        
        return reward, done

    def _setup_flight(self):
        """Initialize flight state."""
        self.drone.reset()
        self.state = {
            "collision": False,
            "steps_count": 0,
            "detections": [],
            "out_of_view_count": 0
        }
