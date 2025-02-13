"""
AirSim Drone Controller
Main drone control implementation for AirSim simulator with comprehensive movement,
camera control, and state management functionality.
"""

import airsim
import math
import numpy as np
import json
import time
from pyquaternion import Quaternion

# Control constants
CONTROL_CONFIG = {
    'timeout': 30,
    'max_retries': 5
}

class AirsimDrone:
    """Main drone controller for AirSim simulator."""

    def __init__(
        self,
        start_pos: list = [0, 0, 0],
        cam_angle: float = -90,
        velocity: float = 5,
        max_distance: float = 10000,
        ip_addr: str = '127.0.0.1'
    ):
        """
        Initialize drone controller.
        
        Args:
            start_pos: Initial position [x, y, z]
            cam_angle: Initial camera angle in degrees
            velocity: Default movement velocity
            max_distance: Maximum allowed travel distance
            ip_addr: AirSim server IP address
        """
        # Initialize AirSim connection
        self.client = airsim.MultirotorClient(ip=ip_addr)
        self._setup_connection()
        
        # Store initial parameters
        self.start_pos = start_pos
        self.altitude = abs(start_pos[2])
        self.cam_angle = cam_angle
        self.velocity = velocity
        self.max_distance = max_distance
        
        # Setup camera configuration
        self._img_request = airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        self._setup_camera()
        
        # Initialize drone state
        self.reset()

    def _setup_connection(self):
        """Establish and confirm AirSim connection."""
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def _setup_camera(self):
        """Configure camera settings from AirSim."""
        as_settings = json.loads(self.client.getSettingsString())
        camera_settings = as_settings['CameraDefaults']['CaptureSettings'][0]
        
        # Store camera parameters
        self.img_width = camera_settings['Width']
        self.img_height = camera_settings['Height']
        self.fov_h = camera_settings['FOV_Degrees']
        self.fov_v = self._calculate_vertical_fov()
        self.img_channels = 3

    def _calculate_vertical_fov(self) -> float:
        """Calculate vertical field of view based on horizontal FOV and aspect ratio."""
        return math.degrees(2 * math.atan(
            math.tan(math.radians(self.fov_h/2)) * self.img_height / self.img_width
        ))

    # Position and Movement Methods
    def set_start_pos(self, start_x: float, start_y: float, start_z: float):
        """Set drone start position."""
        self.start_pos = [start_x, start_y, start_z]

    def move_to(self, x: float, y: float, z: float, velocity: float = 5):
        """Move drone to specific position."""
        self.client.moveToPositionAsync(
            x, y, z, velocity,
            CONTROL_CONFIG['timeout'],
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0)
        ).join()

    def move_on_path(self, path: list, velocity: float = 5):
        """Move drone along specified path."""
        if path and not isinstance(path[0], airsim.Vector3r):
            path = [airsim.Vector3r(p[0], p[1], p[2]) for p in path]
        
        self.client.moveOnPathAsync(
            path,
            velocity,
            CONTROL_CONFIG['timeout'],
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0)
        ).join()

    def move(self, direction: str, speed: float, duration: float):
        """
        Move drone in specified direction.
        
        Args:
            direction: Movement direction ('left', 'right', 'forward', etc.)
            speed: Movement speed
            duration: Movement duration
        """
        # Movement direction vectors
        direction_vectors = {
            "left": [0, -speed, 0],
            "right": [0, speed, 0],
            "forward": [speed, 0, 0],
            "backward": [-speed, 0, 0],
            "up": [0, 0, -speed],
            "down": [0, 0, speed],
            "left_forward": [speed, -speed, 0],
            "right_forward": [speed, speed, 0],
            "left_down": [0, -speed, speed],
            "right_down": [0, speed, speed],
            "forward_down": [speed, 0, speed]
        }
        
        direction_vector = direction_vectors.get(direction, [0, 0, 0])
        
        # Execute movement
        self.client.moveByVelocityAsync(
            vx=direction_vector[0],
            vy=direction_vector[1],
            vz=direction_vector[2],
            duration=duration
        ).join()
        
        time.sleep(duration)
        self._log_position_change()

    # Camera Control Methods
    def set_cam_angle(self, cam_angle: float):
        """Set camera angle in degrees."""
        self.cam_angle = cam_angle

    def set_cam_pose(self, x: float = 0, y: float = 0, z: float = 0,
                    pitch: float = 0, yaw: float = 0, roll: float = 0):
        """Set camera pose with position and orientation."""
        cam_pose = airsim.Pose(
            airsim.Vector3r(x, y, z),
            airsim.to_quaternion(
                math.radians(pitch),
                math.radians(yaw),
                math.radians(roll)
            )
        )
        self.client.simSetCameraPose("0", cam_pose)

    def capture_cam_img(self) -> np.ndarray:
        """
        Capture image from drone camera.
        
        Returns:
            np.ndarray: RGB image array
        """
        # Capture image with retries
        responses = self.client.simGetImages([self._img_request])
        retry_count = 0
        
        while (not responses or len(responses) == 0 or 
               len(responses[0]) == 0) and retry_count < CONTROL_CONFIG['max_retries']:
            responses = self.client.simGetImages([self._img_request])
            retry_count += 1
            
        # Process image
        response = responses[0]
        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_bgr = img_1d.reshape(self.img_height, self.img_width, self.img_channels)
        return img_bgr[:, :, ::-1]  # Convert BGR to RGB

    # State Management Methods
    def reset(self):
        """Reset drone to initial state."""
        offset_y = self.altitude * self.img_height / self.img_width + self.altitude * math.tan(
            math.radians(90 - abs(self.cam_angle) - self.fov_v / 2))
        self.set_pose(self.start_pos[0] - offset_y, self.start_pos[1], self.start_pos[2])
        self.set_cam_pose(z=1, pitch=self.cam_angle)
        self.takeoff_asnc()
        self.hover()

    def _log_position_change(self):
        """Log position change for debugging."""
        pos = self.client.simGetVehiclePose().position
        print(f'After Position: {pos.x_val}, {pos.y_val}, {pos.z_val}')

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
