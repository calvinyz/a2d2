import airsim
import math
import numpy as np
import json
import time
from pyquaternion import Quaternion

DRONE_TIMEOUT = 30
MAX_RETRIES = 5

class AirsimDrone:

    def __init__(self, start_pos = [0, 0, 0], cam_angle=-90, velocity=5, max_distance = 10000, ip_addr='127.0.0.1'):
        self.client = airsim.MultirotorClient(ip=ip_addr)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.start_pos = start_pos
        self.altitude = abs(start_pos[2])
        self.cam_angle = cam_angle
        self.velocity = velocity
        self._img_reqest = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        as_settings = json.loads(self.client.getSettingsString())

        self.img_width = as_settings['CameraDefaults']['CaptureSettings'][0]['Width']
        self.img_height = as_settings['CameraDefaults']['CaptureSettings'][0]['Height']
        self.fov_h = as_settings['CameraDefaults']['CaptureSettings'][0]['FOV_Degrees']
        self.fov_v = math.degrees(2 * math.atan(math.tan(math.radians(self.fov_h/2)) * self.img_height / self.img_width)) 
        self.img_channels = 3
        self.max_distance = max_distance

        self.reset()

    def set_start_pos(self, start_x, start_y, start_z):
        self.start_pos = [start_x, start_y, start_z]

    def set_cam_angle(self, cam_angle):
        self.cam_angle = cam_angle

    def set_velocity(self, vx, vy, vz, duration=1):
        self.velocity = [vx, vy, vz, duration]

    def set_max_distance(self, distance):
        self.max_distance = distance
    
    def reset(self):
        # self.client.reset()
        offset_y = self.altitude * self.img_height / self.img_width + self.altitude * math.tan(
            math.radians(90 - abs(self.cam_angle) - self.fov_v / 2))
        self.set_pose(self.start_pos[0] - offset_y, self.start_pos[1], self.start_pos[2])
        self.set_cam_pose(z=1, pitch=self.cam_angle)
        self.takeoff_asnc()
        self.hover()

    def move_to(self, x, y, z, velocity=5):
        self.client.moveToPositionAsync(x, y, z, velocity, DRONE_TIMEOUT, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
    
    def hover(self):
        self.client.hoverAsync().join()

    def hover_async(self):
        self.client.hoverAsync()

    def takeoff(self):
        self.client.takeoffAsync().join()

    def takeoff_asnc(self):
        self.client.takeoffAsync()

    def cancel_last_task(self):
        self.client.cancelLastTask()

    def move_on_path(self, path, velocity=5):
        if path and not type(path[0]) is airsim.Vector3r:
            path = [airsim.Vector3r(p[0], p[1], p[2]) for p in path]
        self.client.moveOnPathAsync(path, velocity, DRONE_TIMEOUT, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()

    def move_by_velocity(self, vx, vy, vz, duration):
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()

    def get_obj_pos(self, obj_name):
        obj_pos = self.client.simGetObjectPose(obj_name)
        return obj_pos.position.x_val, obj_pos.position.y_val, obj_pos.position.z_val

    def move_obj_to(self, obj_name, x, y, z):
        obj_pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
        return self.client.simSetObjectPose(obj_name, obj_pose)
    
    def get_cam_pose(self):
        cam_pose = self.client.simGetCameraInfo("0").pose
        x, y, z = cam_pose.position.x_val, cam_pose.position.y_val, cam_pose.position.z_val
        pitch, yaw, roll = airsim.to_eularian_angles(cam_pose.orientation)
        return x, y, z, pitch, yaw, roll

    def set_cam_pose(self, x=0, y=0, z=0, pitch=0, yaw=0, roll=0):
        cam_pose = airsim.Pose(airsim.Vector3r(x, y, z), 
                               airsim.to_quaternion(math.radians(pitch), 
                                                    math.radians(yaw), 
                                                    math.radians(roll)))
        self.client.simSetCameraPose("0", cam_pose)

    def get_pose(self):
        pose = self.client.simGetVehiclePose()
        x, y, z = pose.position.x_val, pose.position.y_val, pose.position.z_val       
        pitch, yaw, roll = airsim.to_eularian_angles(pose.orientation)
        return x, y, z, pitch, yaw, roll
    
    def set_pose(self, x, y, z, pitch=0, roll=0, yaw=0):
        orient = airsim.to_quaternion(math.radians(pitch), 
                                      math.radians(yaw), 
                                      math.radians(roll))
        pose = airsim.Pose(airsim.Vector3r(x, y, z), orient)
        self.client.simSetVehiclePose(pose, True)

    def check_collision(self):
        return self.client.simGetCollisionInfo().has_collided
    
    def get_linear_velocity(self):
        return self.client.getMultirotorState().kinematics_estimated.linear_velocity

    def capture_cam_img(self):
        # Capture image from the drone
        responses = self.client.simGetImages([self._img_reqest])
        # Retry if responses invalid
        retry = 0
        while (not responses) and len(responses) == 0 and len(responses[0])  == 0 and retry < MAX_RETRIES:
            responses = self.client.simGetImages([self._img_reqest])
            retry += 1
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_bgr = img1d.reshape(self.img_height, self.img_width, self.img_channels)
        img = img_bgr[:, :, ::-1] # Convert BGR to RGB
        return img
    
    def rotate(self, direction, speed, duration):
        orient = self.client.simGetVehiclePose().orientation
        pitch, yaw, roll = airsim.to_eularian_angles(orient)
        print(f'Before orient: {yaw}, {pitch}, {roll}')
        if direction == "turn_left":
            speed = -1*speed
        self.client.rotateByYawRateAsync(speed, duration).join()
        orient = self.client.simGetVehiclePose().orientation
        pitch, yaw, roll = airsim.to_eularian_angles(orient)
        print(f'After orient: {yaw}, {pitch}, {roll}')

    def move(self, direction, speed, duration):
        pose = self.client.simGetVehiclePose()
        orient = pose.orientation
        orient_q = Quaternion(w_val=orient.w_val,
                              x_val=orient.x_val,
                              y_val= orient.y_val,
                              z_val=orient.z_val)
        pos = pose.position
        print(f'Before Position: {pos.x_val}, {pos.y_val}, {pos.z_val}')

        ang_vel = self.client.getMultirotorState().\
            kinematics_estimated.angular_velocity
        lin_vel = self.client.getMultirotorState().\
            kinematics_estimated.linear_velocity

        if direction == "left":
            drtn = [0, -1*speed, 0]
        elif direction == "right":
            drtn = [0, speed, 0]
        elif direction == "forward":
            drtn = [speed, 0, 0]
        elif direction == "backward":
            drtn = [-1*speed, 0, 0]
        elif direction == "up":
            drtn = [0, 0, -1*speed]
        elif direction == "down":
            drtn = [0, 0, speed]
        elif direction == "left_forward":
            drtn = [speed, -1*speed, 0]
        elif direction == "right_forward":
            drtn = [speed, speed, 0]
        elif direction == "left_down":
            drtn = [0, -1*speed, speed]
        elif direction == "right_down":
            drtn = [0, speed, speed]
        elif direction == "forward_down":
            drtn = [speed, 0, speed]
        else:
            drtn = [0, 0, 0]

        # if direction in ['up', 'down']: 
        #     self.client.moveByVelocityAsync(vx=0, 
        #                                     vy=0, 
        #                                     vz=lin_vel.z_val + drtn[2], 
        #                                     duration=duration).join()#,
        #                                     # drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        #                                     # yaw_mode=airsim.YawMode(True, 0)).join()
        # else:
        #     orient_q_rtt = orient_q.rotate(drtn)
        #     self.client.moveByVelocityZAsync(vx=2*(ang_vel.x_val + orient_q_rtt[0]),
        #                                      vy=2*(ang_vel.y_val + orient_q_rtt[1]), 
        #                                      z = pos.z_val, 
        #                                      duration=duration).join()

        self.client.moveByVelocityAsync(vx=drtn[0], 
                                        vy=drtn[1], 
                                        vz=drtn[2], 
                                        duration=duration).join()

        time.sleep(duration)
            
        pos = self.client.simGetVehiclePose().position
        print(f'After Position: {pos.x_val}, {pos.y_val}, {pos.z_val}')
