from datetime import datetime
from inspection_path.a2c.model import A2C_Agent
from drone.airsim_drone import AirsimDrone
from detection.YOLOv8.model import YOLOv8

if __name__ == '__main__':
    
    retrain = True

    train_steps = 10_000
    detector_model_name = 'yolov8n_wildfire_as_e100_new_2.pt'
    ipp_model_name = f'ipp_model_a2c-{train_steps}-{datetime.now().strftime("%Y%m%d%H%M%S")}'

    altitude = 50
    cam_angle = -60

    start_x = 0
    start_y = 0

    drone = AirsimDrone(start_pos=[-40, 0, -altitude], cam_angle=cam_angle)

    yolov8detector = YOLOv8(detector_model_name)

    # TODO: Train drl with different start positions?
    a2c = A2C_Agent(drone, 
                      yolov8detector, 
                      step_length=2, 
                      velo_duration=1,
                      max_steps_episode=25)

    if retrain:
        a2c.train(10_000)
        a2c.evaluate()
        a2c.save(ipp_model_name)
    else:
        a2c.load(ipp_model_name)

    completed = False
    obs = a2c.env.reset()
    while not completed:
        action, _states = a2c.predict(obs, deterministic=True)
        print(action, _states)
        obs, reward, completed, truncated = a2c.env.step(action)
        
        if truncated:
            # Restore drone to initial pose when start inspection
            print('Truncated. Restoring drone to initial pose')
    if completed:
        print('Inspection completed')
        # Send alerts