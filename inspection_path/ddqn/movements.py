"""
Drone Movement Controls
Provides movement control functions for the AirSim drone using quaternion-based transformations.
"""

import airsim
from pyquaternion import Quaternion

# Movement configuration
DRIVETRAIN = airsim.DrivetrainType.MaxDegreeOfFreedom

def straight(client: airsim.MultirotorClient, speed: float, duration: float, 
            direction: str, z: float):
    """
    Move the drone in a straight line along specified direction.
    
    Args:
        client: AirSim client instance
        speed: Movement speed
        duration: Duration of movement
        direction: Direction of movement ('right', 'left', or 'straight')
        z: Target altitude
    """
    # Define movement vector based on direction
    action_vectors = {
        "right": [0, speed, 0],
        "left": [0, -speed, 0],
        "straight": [speed, 0, 0]
    }
    action = action_vectors[direction]

    # Get current orientation and convert to quaternion
    current_orientation = client.simGetVehiclePose().orientation
    orientation_quaternion = Quaternion(
        w_val=current_orientation.w_val,
        x_val=current_orientation.x_val,
        y_val=current_orientation.y_val,
        z_val=current_orientation.z_val
    )
    
    # Apply rotation to movement vector
    rotated_movement = orientation_quaternion.rotate(action)
    
    # Get current velocities
    current_velocities = client.getMultirotorState().kinematics_estimated.angular_velocity
    drone_velocities = [current_velocities.x_val, current_velocities.y_val]
    
    # Execute movement
    client.moveByVelocityZAsync(
        vx=drone_velocities[0] + rotated_movement[0],
        vy=drone_velocities[1] + rotated_movement[1],
        z=z,
        duration=duration,
        drivetrain=DRIVETRAIN,
        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0)
    ).join()

def yaw_right(client: airsim.MultirotorClient, speed: float, duration: float):
    """
    Rotate the drone right.
    
    Args:
        client: AirSim client instance
        speed: Rotation speed
        duration: Duration of rotation
    """
    client.rotateByYawRateAsync(speed, duration).join()

def yaw_left(client: airsim.MultirotorClient, speed: float, duration: float):
    """
    Rotate the drone left.
    
    Args:
        client: AirSim client instance
        speed: Rotation speed
        duration: Duration of rotation
    """
    client.rotateByYawRateAsync(-speed, duration).join()

def up(client: airsim.MultirotorClient, speed: float, duration: float):
    """
    Move the drone upward.
    
    Args:
        client: AirSim client instance
        speed: Vertical speed
        duration: Duration of movement
    """
    client.moveByVelocityZAsync(0, 0, -speed, duration).join()

def down(client: airsim.MultirotorClient, speed: float, duration: float):
    """
    Move the drone downward.
    
    Args:
        client: AirSim client instance
        speed: Vertical speed
        duration: Duration of movement
    """
    client.moveByVelocityZAsync(0, 0, speed, duration).join()
