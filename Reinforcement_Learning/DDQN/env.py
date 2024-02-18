from PIL import Image
import numpy as np
import airsim
import movements
import object_detection
import torch

MOVEMENT_INTERVAL = 1

class DroneEnv(object):

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

    def step(self, action):
        """ Step """
        self.move(action)
       
        collision = self.client.simGetCollisionInfo().has_collided
        state, image, detections = self.get_obs()
        result, done = self.compute_reward(collision, detections)
        
        return state, result, done, image

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
       
        obs, image, _ = self.get_obs()

        return obs, image

    def get_obs(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        image = img1d.reshape(response.height, response.width, 3)

        # Perform object detection
        detections = object_detection.detect_objects(image)

        # Visualize detections
        image_with_detections = object_detection.visualize_detections(image, detections)
        image_array = Image.fromarray(image_with_detections).resize((84, 84)).convert("L")
        obs = np.array(image_array)

        return obs, image_with_detections, detections

    def compute_reward(self, collision, detections):
        """ Compute reward based on quadcopter state, collision, and object detections """
        reward = 0
        done = 0

        if collision:
            reward = -1
            done = 1
        else:
            if not detections or len(detections) == 0 or len(detections[0].boxes.xyxy) == 0:
                reward = 0
            else:            
                bboxs = detections[0].boxes.xyxy
                bbox = bboxs[0]
                bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                bbox_size_ratio = bbox_size / 409600
                bbox_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                bbox_x_offset_ratio = 1 - abs((bbox_center[0] - 320)/320)
                bbox_y_offset_ratio = 1 - abs((bbox_center[1] - 320)/320)
        

                if reward > 60 or bbox_size_ratio > 0.125: # cannot be too close to fire)
                    done = 1

                reward = int(bbox_size_ratio * 200 + bbox_x_offset_ratio * 25 + bbox_y_offset_ratio * 25)
                print(f'Reward: {reward}. s: {bbox_size_ratio:.2f}, x: {bbox_x_offset_ratio:.2f}, y: {bbox_y_offset_ratio:.2f}. Done: {done}')

        return reward, done

    def move(self, action):
        """ Take a move in the world """
        
        self.client.hoverAsync()

        # Do nothing
        if action == 0:
            self.client.moveByVelocityAsync(0, 0, 0, 1)
            self.client.rotateByYawRateAsync(0, 1)

        # Orient right
        if action == 1:
            movements.yaw_right(self.client, 2, 0.2)

        # Orient left
        if action == 2:
            movements.yaw_left(self.client, 2, 0.2)

        # Go straight
        if action == 3:
            movements.straight(self.client, 2, 0.3, "straight", -2)

        # Go right
        if action == 4:
            movements.straight(self.client, 2, 0.3, "right", -2)

        # Go left
        if action == 5:
            movements.straight(self.client, 2, 0.3, "left", -2)

        # Go up
        if action == 6:
            movements.up(self.client, 2, 0.2)

        # Go down
        elif action == 7:
            movements.down(self.client, 2, 0.2)