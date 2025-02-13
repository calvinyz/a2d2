"""
Inspection Path Planning with A2C
Trains and evaluates an A2C (Advantage Actor-Critic) agent for drone inspection path planning.
"""

from datetime import datetime
from inspection_path.a2c.model import A2C_Agent
from drone.airsim_drone import AirsimDrone
from detection.YOLOv8.model import YOLOv8

# Training configuration
TRAINING_CONFIG = {
    'steps': 10_000,
    'retrain': True,
    'detector_model': 'yolov8n_wildfire_as_e100_new_2.pt'
}

# Drone configuration
DRONE_CONFIG = {
    'altitude': 50,
    'cam_angle': -60,
    'start_position': [-40, 0, -50],  # [x, y, z]
}

# A2C agent parameters
AGENT_CONFIG = {
    'step_length': 2,
    'velocity_duration': 1,
    'max_steps_per_episode': 25
}

def setup_environment():
    """
    Initialize the drone, detector, and A2C agent.
    
    Returns:
        tuple: (A2C_Agent, str) - trained agent and model name
    """
    # Initialize drone with configuration
    drone = AirsimDrone(
        start_pos=DRONE_CONFIG['start_position'],
        cam_angle=DRONE_CONFIG['cam_angle']
    )
    
    # Initialize fire detector
    detector = YOLOv8(TRAINING_CONFIG['detector_model'])
    
    # Create unique model name with timestamp
    model_name = f'ipp_model_a2c-{TRAINING_CONFIG["steps"]}-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    
    # Initialize A2C agent
    agent = A2C_Agent(
        drone=drone,
        detector=detector,
        step_length=AGENT_CONFIG['step_length'],
        velo_duration=AGENT_CONFIG['velocity_duration'],
        max_steps_episode=AGENT_CONFIG['max_steps_per_episode']
    )
    
    return agent, model_name

def train_or_load_agent(agent: A2C_Agent, model_name: str):
    """
    Train a new agent or load an existing one based on configuration.
    
    Args:
        agent: A2C_Agent instance to train or load
        model_name: Name of the model file
    """
    if TRAINING_CONFIG['retrain']:
        agent.train(TRAINING_CONFIG['steps'])
        agent.evaluate()
        agent.save(model_name)
    else:
        agent.load(model_name)

def run_inspection(agent: A2C_Agent):
    """
    Run the inspection process using the trained agent.
    
    Args:
        agent: Trained A2C_Agent instance
    """
    completed = False
    obs = agent.env.reset()
    
    while not completed:
        # Get and execute action
        action, _states = agent.predict(obs, deterministic=True)
        print(f"Action: {action}, States: {_states}")
        obs, reward, completed, truncated = agent.env.step(action)
        
        if truncated:
            print('Truncated. Restoring drone to initial pose')
            
    if completed:
        print('Inspection completed')

def main():
    """Main execution function"""
    # Setup environment and agent
    agent, model_name = setup_environment()
    
    # Train or load the agent
    train_or_load_agent(agent, model_name)
    
    # Run inspection
    run_inspection(agent)

if __name__ == '__main__':
    main()