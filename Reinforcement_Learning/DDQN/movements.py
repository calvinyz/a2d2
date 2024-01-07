import airsim
from pyquaternion import Quaternion

""" Go straight movement for the drone """
def straight(client, speed, duration, direction, z):
    
    typeDrivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
    if direction == "right":
        action = [0, speed, 0]
    elif direction == "left":
        action = [0, -speed, 0]
    elif direction == "straight":
        action = [speed, 0, 0]

    # Get current motorstate and transform them to quaternion
    q                   = client.simGetVehiclePose().orientation 
    my_quaternion       = Quaternion(w_val=q.w_val,x_val=q.x_val,y_val= q.y_val,z_val=q.z_val)
    mvm                 = my_quaternion.rotate(action)
    velocities          = client.getMultirotorState().kinematics_estimated.angular_velocity
    donre_vel_rota      = [velocities.x_val , velocities.y_val]

    # Perform the movement
    client.moveByVelocityZAsync(vx         = donre_vel_rota[0] + mvm[0],
                                    vy          = donre_vel_rota[1] + mvm[1],
                                    z           = z,
                                    duration    = duration,
                                    drivetrain  = typeDrivetrain,
                                    yaw_mode    = airsim.YawMode(is_rate = True, yaw_or_rate = 0)).join()
    
""" Orient right for the drone """
def yaw_right(client, speed, duration):
    client.rotateByYawRateAsync(speed, duration).join()

""" Orient left for the drone """
def yaw_left(client, speed, duration):
    client.rotateByYawRateAsync(-1*speed, duration).join()

""" Go up movement for the drone """
def up(client, speed, duration):
    client.moveByVelocityZAsync(0, 0, -1*speed, duration).join()

""" Go down movement for the drone """
def down(client, speed, duration):
    client.moveByVelocityZAsync(0, 0, speed, duration).join()
