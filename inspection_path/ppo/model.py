"""
PPO Agent for Drone Inspection Path Planning
Implements a Proximal Policy Optimization (PPO) agent for controlling drone inspection paths
using stable-baselines3 with CNN policy for image-based observations.
"""

import os
import torch
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Register the custom drone environment
gym.envs.registration.register(
    id="drone-airsim-env-v0",
    entry_point="inspection_path.drone_gym.envs:DroneAirSimEnv",
)

class PPO_Agent:
    """PPO agent for drone control using image-based observations."""
    
    def __init__(
        self,
        drone,
        detector,
        step_length: int = 5,
        velo_duration: int = 1,
        max_steps_episode: int = 50,
        tb_log_path: str = "./tb_logs/"
    ):
        """
        Initialize the PPO agent.
        
        Args:
            drone: Drone instance for environment interaction
            detector: Object detection model
            step_length: Length of each step action
            velo_duration: Duration of velocity commands
            max_steps_episode: Maximum steps per episode
            tb_log_path: Path for tensorboard logs
        """
        # Initialize environment with monitoring and preprocessing
        self.env = self._setup_environment(
            drone, detector, step_length, velo_duration, max_steps_episode
        )
        
        # Initialize PPO model
        self.model = self._setup_model(tb_log_path)

    def _setup_environment(self, drone, detector, step_length, velo_duration, max_steps_episode):
        """Setup and wrap the drone environment."""
        base_env = gym.make(
            "drone-airsim-env-v0",
            drone=drone,
            detector=detector,
            step_length=step_length,
            velocity_duration=velo_duration,
            max_steps_episode=max_steps_episode
        )
        
        # Add monitoring and vectorize
        monitored_env = Monitor(base_env)
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        # Add image transposition for CNN
        return VecTransposeImage(vec_env)

    def _setup_model(self, tb_log_path: str):
        """Setup the PPO model with CNN policy."""
        return PPO(
            "CnnPolicy",
            self.env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=tb_log_path
        )

    def train(
        self,
        total_steps: int = 1000,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "ppo_drone_airsim_run"
    ):
        """
        Train the PPO agent.
        
        Args:
            total_steps: Total timesteps for training
            reset_num_timesteps: Whether to reset timestep counter
            tb_log_name: Name for tensorboard logging
        """
        # Setup evaluation callback
        eval_callback = EvalCallback(
            self.env,
            callback_on_new_best=None,
            n_eval_episodes=5,
            best_model_save_path=".\\best_model\\ppo",
            log_path=".\\best_model_eval_logs\\ppo",
            eval_freq=100,
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            callback=[eval_callback]
        )

    def save(self, model_name: str = "ppo_airsim_drone_policy"):
        """Save the trained model."""
        self.model.save(model_name)
        print(f"Model {model_name} saved")

    def load(self, model_name: str):
        """Load a trained model."""
        model_path = f'{model_name}.zip'
        if os.path.exists(model_path):
            if self.model is not None:
                del self.model
            self.model = PPO.load(model_name, env=self.env)
            print(f"Model {model_name} loaded")
        else:
            print(f"Model {model_name} does not exist")

    def evaluate(self, n_eval_episodes: int = 5) -> tuple[float, float]:
        """
        Evaluate the trained model.
        
        Args:
            n_eval_episodes: Number of episodes for evaluation
            
        Returns:
            tuple: (mean_reward, std_reward)
        """
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_eval_episodes
        )
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        return mean_reward, std_reward

    def predict(self, obs, deterministic: bool = True) -> tuple:
        """
        Get action prediction from the model.
        
        Args:
            obs: Environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            tuple: (action, states)
        """
        return self.model.predict(obs, deterministic=deterministic)

    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)
