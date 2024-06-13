import gymnasium as gym
from datetime import datetime
import torch
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from gymnasium.envs.registration import register

register(
    id="drone-airsim-env-v0", 
    entry_point="inspection_path.drone_gym.envs:DroneAirSimEnv",
)


class PPO_Agent:
    def __init__(self, drone, detector, step_length=5, velo_duration=1, max_steps_episode=50, tb_log_path="./tb_logs/"):
        self.env = VecTransposeImage(
            DummyVecEnv(
                [lambda: Monitor(
                    gym.make("drone-airsim-env-v0",
                             drone=drone, 
                             detector=detector,
                             step_length=step_length,
                             velocity_duration=velo_duration,
                             max_steps_episode=max_steps_episode))]))
        
        self.model = PPO(
            "CnnPolicy",
            self.env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=tb_log_path)
        
    def train(self, total_steps=1000, reset_num_timesteps=True,
              tb_log_name="ppo_drone_airsim_run"):
        # Create an evaluation callback with the same env, called every 10000 iterations
        callbacks = []
        eval_callback = EvalCallback(
            self.env,
            callback_on_new_best=None,
            n_eval_episodes=5,
            best_model_save_path=".\\best_model\\ppo",
            log_path=".\\best_model_eval_logs\\ppo",
            eval_freq=100,
        )
        callbacks.append(eval_callback)

        # Train for a certain number of timesteps
        self.model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            callback=callbacks)
        
    def step(self, action):
        return self.env.step(action)

    def save(self, model_name="ppo_airsim_drone_policy"):
        self.model.save(model_name)
        print(f"Model {model_name} saved")

    def load(self, model_name):
        if os.path.exists(f'{model_name}.zip'):
            if self.model is not None:
                del self.model            
            self.model = PPO.load(model_name, env=self.env)
            print(f"Model {model_name} loaded")
        else:
            print(f"Model {model_name} does not exist")

    def evaluate(self, n_eval_episodes=5):
        mean_reward, std_reward = evaluate_policy(
            self.model, self.env, n_eval_episodes=n_eval_episodes)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
        return mean_reward, std_reward

    def predict(self, obs, deterministic=True):
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action, _states
