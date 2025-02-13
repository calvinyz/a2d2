"""
Inspection Path Planning with DQN
Trains and evaluates a DQN (Deep Q-Network) agent for drone inspection path planning.
"""

from datetime import datetime
from inspection_path.dqn.model import DQN_Agent
from drone.airsim_drone import AirsimDrone
from detection.YOLOv8.model import YOLOv8

# Training configuration
TRAINING_CONFIG = {
    'train_mode': False,
    'total_steps': 5000,
    'save_interval': 1000,
    'detector_model': 'yolov8n_wildfire_as_e100_new_2.pt',
    'model_timestamp': datetime.now().strftime("%Y%m%d%H%M")
}

# Model naming
MODEL_CONFIG = {
    'base_name': 'ipp_model_dqn',
    'load_model': 'ipp_model_dqn-202404131853-6000',  # Used when train_mode is False
}

# Drone configuration
DRONE_CONFIG = {
    'altitude': 50,
    'cam_angle': -60,
    'start_position': [-40, 40, -50]  # [x, y, z]
}

# DQN agent parameters
AGENT_CONFIG = {
    'step_length': 20,
    'velocity_duration': 2,
    'max_steps_per_episode': 25
}

def setup_environment():
    """
    Initialize the drone, detector, and DQN agent.
    
    Returns:
        tuple: (DQN_Agent, str) - agent and model name
    """
    # Initialize drone with configuration
    drone = AirsimDrone(
        start_pos=[*DRONE_CONFIG['start_position'][:2], -DRONE_CONFIG['altitude']],
        cam_angle=DRONE_CONFIG['cam_angle']
    )
    
    # Initialize fire detector
    detector = YOLOv8(TRAINING_CONFIG['detector_model'])
    
    # Initialize DQN agent
    agent = DQN_Agent(
        drone=drone,
        detector=detector,
        step_length=AGENT_CONFIG['step_length'],
        velo_duration=AGENT_CONFIG['velocity_duration'],
        max_steps_episode=AGENT_CONFIG['max_steps_per_episode']
    )
    
    # Set model name based on mode
    model_name = (f"{MODEL_CONFIG['base_name']}-{TRAINING_CONFIG['model_timestamp']}" 
                 if TRAINING_CONFIG['train_mode'] else MODEL_CONFIG['load_model'])
    
    return agent, model_name

def train_agent(agent: DQN_Agent, model_name: str):
    """
    Train the DQN agent with periodic saving.
    
    Args:
        agent: DQN_Agent instance to train
        model_name: Base name for saving model checkpoints
    """
    current_steps = 0
    while current_steps <= TRAINING_CONFIG['total_steps']:
        # Train for save_interval steps
        agent.train(
            total_steps=TRAINING_CONFIG['save_interval'],
            reset_num_timesteps=(current_steps == 0),
            tb_log_name=model_name
        )
        
        # Evaluate and save checkpoint
        agent.evaluate()
        current_steps += TRAINING_CONFIG['save_interval']
        checkpoint_name = f'{model_name}-{current_steps}'
        agent.save(checkpoint_name)

def run_inspection(agent: DQN_Agent):
    """
    Run the inspection process using the trained agent.
    
    Args:
        agent: Trained DQN_Agent instance
    """
    completed = False
    obs = agent.env.reset()
    
    while not completed:
        # Get and execute action
        action, states = agent.predict(obs)
        print(f"Action: {action}, States: {states}")
        obs, reward, completed, info = agent.env.step(action)
        print(f'Reward: {reward}, Completed: {completed}')
        
    if completed:
        print('Inspection completed')

def main():
    """Main execution function"""
    # Setup environment and agent
    agent, model_name = setup_environment()
    
    # Train or load based on configuration
    if TRAINING_CONFIG['train_mode']:
        train_agent(agent, model_name)
    else:
        agent.load(model_name)
        run_inspection(agent)

if __name__ == '__main__':
    main()