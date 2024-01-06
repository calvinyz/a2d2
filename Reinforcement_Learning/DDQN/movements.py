import airsim
from pyquaternion import Quaternion

""" Go Straight Movement for the Drone """
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
    client.moveByVelocityZAsync(vx         = donre_vel_rota[0] + mvm[0], #the already existing speed + the one the agent wants to add, smoother drive?
                                    vy          = donre_vel_rota[1] + mvm[1],
                                    z           = z,
                                    duration    = duration, #will last x secondes or will be stoped by a new command (put a time.sleep(0.5) next to it)
                                    drivetrain  = typeDrivetrain, #the camera is indepedant of the movement, but the movement is w.r.t the cam orientation
                                    yaw_mode    = airsim.YawMode(is_rate = True, yaw_or_rate = 0)).join() # True means that yaw_or_rate is seen as a degrees/sec

""" Orient Right for the Drone """
def yaw_right(client, speed, duration):
    client.rotateByYawRateAsync(speed, duration).join()

""" Orient Left for the Drone """
def yaw_left(client, speed, duration):
    client.rotateByYawRateAsync(-1*speed, duration).join()

""" Go Up Movement for the Drone """
def up(client, speed, duration):
    client.moveByVelocityZAsync(0, 0, -1*speed, duration).join()

""" Go Down Movement for the Drone """
def down(client, speed, duration):
    client.moveByVelocityZAsync(0, 0, speed, duration).join()
