"""
DDQN Drone Environment
Custom environment for drone control in AirSim simulator with Double DQN implementation.
"""

import time
import numpy as np
from PIL import Image
import airsim
import pathplanner.inspection.DDQN.movements as movements
import pathplanner.inspection.DDQN.object_detection as object_detection

# Environment constants
MOVEMENT_CONFIG = {
    'interval': 1,
    'velocity_scale': 0.3,
    'rotation_scale': 0.2,
    'duration': 2
}

# Action mapping
ACTIONS = {
    0: 'hover',
    1: 'yaw_right',
    2: 'yaw_left',
    3: 'move_forward',
    4: 'move_right',
    5: 'move_left',
    6: 'move_up',
    7: 'move_down'
}

class DroneEnv:
    """Environment for drone control in AirSim."""

    def __init__(self):
        """Initialize drone environment and establish AirSim connection."""
        self.client = airsim.MultirotorClient()
        self._setup_drone()

    def _setup_drone(self):
        """Setup initial drone configuration."""
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

    def step(self, action: int) -> tuple:
        """
        Execute one environment step.
        
        Args:
            action: Integer representing the action to take
            
        Returns:
            tuple: (state, reward, done, image)
        """
        # Execute action
        self.move(action)
        
        # Get updated state
        collision = self.client.simGetCollisionInfo().has_collided
        state, image, detections = self.get_obs()
        reward, done = self.compute_reward(collision, detections)
        
        return state, reward, done, image

    def reset(self) -> tuple:
        """
        Reset environment to initial state.
        
        Returns:
            tuple: (observation, image)
        """
        self.client.reset()
        self._setup_drone()
        obs, image, _ = self.get_obs()
        return obs, image

    def get_obs(self) -> tuple:
        """
        Get current observation from environment.
        
        Returns:
            tuple: (processed_observation, raw_image, detections)
        """
        # Capture image
        responses = self.client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)
        ])
        response = responses[0]
        
        # Process image
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        image = img1d.reshape(response.height, response.width, 3)
        
        # Perform object detection
        detections = object_detection.detect_objects(image)
        image_with_detections = object_detection.visualize_detections(image, detections)
        
        # Process observation
        processed_obs = np.array(
            Image.fromarray(image_with_detections)
            .resize((84, 84))
            .convert("L")
        )
        
        return processed_obs, image_with_detections, detections

    def compute_reward(self, collision: bool, detections: list) -> tuple[float, bool]:
        """
        Compute reward based on current state.
        
        Args:
            collision: Whether drone has collided
            detections: List of object detections
            
        Returns:
            tuple: (reward, done)
        """
        if collision:
            return -50, True
        
        # Get highest confidence detection
        boxes_conf = detections[0].boxes.conf
        max_conf = max(boxes_conf.tolist()) if boxes_conf.numel() > 0 else 0
        print(f'Confidence: {max_conf}')
        
        # Calculate reward
        reward = round(max_conf * 100)
        done = reward < 0 or reward > 80
        
        print(f'Reward: {reward}')
        return reward, done

    def move(self, action: int):
        """
        Execute movement action.
        
        Args:
            action: Integer representing the action to take
        """
        self.client.hoverAsync()
        
        if action == 0:  # Hover
            self.client.moveByVelocityAsync(0, 0, 0, MOVEMENT_CONFIG['interval'])
            self.client.rotateByYawRateAsync(0, MOVEMENT_CONFIG['interval'])
            
        elif action == 1:  # Yaw right
            movements.yaw_right(
                self.client, 
                MOVEMENT_CONFIG['duration'],
                MOVEMENT_CONFIG['rotation_scale']
            )
            
        elif action == 2:  # Yaw left
            movements.yaw_left(
                self.client,
                MOVEMENT_CONFIG['duration'],
                MOVEMENT_CONFIG['rotation_scale']
            )
            
        elif action == 3:  # Forward
            movements.straight(
                self.client,
                MOVEMENT_CONFIG['duration'],
                MOVEMENT_CONFIG['velocity_scale'],
                "straight",
                -2
            )
            
        elif action == 4:  # Right
            movements.straight(
                self.client,
                MOVEMENT_CONFIG['duration'],
                MOVEMENT_CONFIG['velocity_scale'],
                "right",
                -2
            )
            
        elif action == 5:  # Left
            movements.straight(
                self.client,
                MOVEMENT_CONFIG['duration'],
                MOVEMENT_CONFIG['velocity_scale'],
                "left",
                -2
            )
            
        elif action == 6:  # Up
            movements.up(
                self.client,
                MOVEMENT_CONFIG['duration'],
                MOVEMENT_CONFIG['rotation_scale']
            )
            
        elif action == 7:  # Down
            movements.down(
                self.client,
                MOVEMENT_CONFIG['duration'],
                MOVEMENT_CONFIG['rotation_scale']
            )