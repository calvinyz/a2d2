from datetime import datetime
from inspection_path.ppo.model import PPO_Agent
from drone.airsim_drone import AirsimDrone
from detection.YOLOv8.model import YOLOv8

if __name__ == '__main__':
    
    train = False

    detector_model_name = 'yolov8n_wildfire_as_e100_new_2.pt'
    train_steps = 5000
    model_save_steps = 1000
    cur_steps = 0
    model_run = f'{datetime.now().strftime("%Y%m%d%H%M")}'

    ipp_model_name = f'ipp_model_ppo-{model_run}' if train else \
        'ipp_model_ppo-202404131835-6000'

    # Starting pos and cam angles should be updated for retraining to cover different cases
    # Each retrained model should continue with the last trained model

    altitude = 50
    cam_angle = -60
    start_x = -40
    start_y = 40

    drone = AirsimDrone(start_pos=[start_x, start_y, -altitude], cam_angle=cam_angle)

    yolov8detector = YOLOv8(detector_model_name)

    ppo = PPO_Agent(drone,
                      yolov8detector, 
                      step_length=20, 
                      velo_duration=2,
                      max_steps_episode=25)

    if train:
        while (cur_steps <= train_steps):
            ppo.train(total_steps=model_save_steps,
                    reset_num_timesteps=True if cur_steps == 0 else False, 
                    tb_log_name=ipp_model_name)
            ppo.evaluate()
            cur_steps += model_save_steps
            steps_model_name = f'{ipp_model_name}-{cur_steps}'
            ppo.save(steps_model_name)
    else:
        ppo.load(ipp_model_name)
        completed = False
        obs = ppo.env.reset()
        while not completed:
            action, _states = ppo.predict(obs)
            print(action, _states)
            obs, reward, completed, info = ppo.env.step(action)
            print(f'reward: {reward}, completed: {completed}')
            
        if completed:
            print('Inspection completed')
            # Send alerts