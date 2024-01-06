import csv
import math
import pprint
import time
import torch
from PIL import Image
import numpy as np
import airsim
import movements

MOVEMENT_INTERVAL = 1

class DroneEnv(object):

    def __init__(self, useDepth=False):
        self.client = airsim.MultirotorClient()
        self.last_dist = self.get_distance(self.client.getMultirotorState().kinematics_estimated.position)
        self.useDepth = useDepth
        
        client = airsim.MultirotorClient()

        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()

    def step(self, action):
        """Step"""
        #print("new step ------------------------------")

        self.move(action)
        #print("quad_offset: ", self.quad_offset)

        # quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
       
        collision = self.client.simGetCollisionInfo().has_collided

        time.sleep(0.5)
        quad_state = self.client.simGetVehiclePose().position

        result, done = self.compute_reward(quad_state, collision)
        state, image = self.get_obs()

        return state, result, done, image

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
 
        obs, image = self.get_obs()

        return obs, image

    def get_obs(self):
        if self.useDepth:
            # get depth image
            responses = self.client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, pixels_as_float=True)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            image = np.reshape(img1d, (responses[0].height, responses[0].width))
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")
        else:
            # Get rgb image
            responses = self.client.simGetImages(
                [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
            )
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, 3)
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")

        obs = np.array(image_array)

        return obs, image

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        pts = np.array([5800, -18930, 15560])
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def compute_reward(self, quad_state, collision):
        """Compute reward"""

        reward = -1

        if collision:
            reward = -50
        else:
            dist = self.get_distance(quad_state)
            diff = self.last_dist - dist

            if dist < 10:
                reward = 500
            else:
                reward += diff

            self.last_dist = dist

        done = 0
        if reward <= -10:
            done = 1
            time.sleep(1)
        elif reward > 499:
            done = 1
            time.sleep(1)

        return reward, done

    """ Take a move in the world """
    def move(self, action):

        self.client.hoverAsync()

        # Do nothing
        if action == 0:
            self.client.moveByVelocityAsync(0, 0, 0, 1)
            self.client.rotateByYawRateAsync(0, 1)

        # Orient Right
        if action == 1:
            movements.yaw_right(self.client, 50, 0.2)

        # Orient Left
        if action == 2:
            movements.yaw_left(self.client, 50, 0.2)

        # Go straight
        if action == 3:
            movements.straight(self.client, 6, 0.3, "straight", -2)

        # Go right
        if action == 4:
            movements.straight(self.client, 6, 0.3, "right", -2)

        # Go left
        if action == 5:
            movements.straight(self.client, 6, 0.3, "left", -2)

        # Go up
        if action == 6:
            movements.up(self.client, 20, 0.2)

        # Go down
        elif action == 7:
            movements.down(self.client, 20, 0.2)