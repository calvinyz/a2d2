"""
Double DQN Model Implementation
Implements a Double Deep Q-Network (DDQN) with prioritized experience replay
for drone control in the AirSim environment.
"""

import math
import random
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from setuptools import glob
from torch.utils.tensorboard import SummaryWriter
from pathplanner.inspection.drone_gym.envs.airsim_env import Env
from pathplanner.inspection.DDQN.prioritized_memory import Memory

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Initialize tensorboard writer and device
writer = SummaryWriter()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, in_channels: int = 1, num_actions: int = 8):
        """
        Initialize the DQN model.
        
        Args:
            in_channels: Number of input channels
            num_actions: Number of possible actions
        """
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 84, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(84, 42, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(42, 21, kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc4 = nn.Linear(21 * 4 * 4, 168)
        self.fc5 = nn.Linear(168, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return self.fc5(x)

class DDQN_Agent:
    """Double DQN Agent with prioritized experience replay."""
    
    def __init__(self, drone, detector):
        """Initialize the DDQN agent."""
        # Training hyperparameters
        self.training_config = {
            'eps_start': 0.9,
            'eps_end': 0.05,
            'eps_decay': 30000,
            'gamma': 0.8,
            'learning_rate': 0.001,
            'batch_size': 128,
            'max_episodes': 10000,
            'max_steps': 200
        }
        
        # Training intervals
        self.intervals = {
            'save': 2,
            'test': 10,
            'network_update': 10
        }
        
        # Initialize training state
        self.training_state = {
            'episode': -1,
            'steps_done': 0,
            'eps_threshold': self.training_config['eps_start']
        }
        
        # Initialize components
        self._setup_networks()
        self._setup_environment(drone, detector)
        self._setup_memory()
        self._setup_directories()
        self._load_checkpoint()
        
        # Initialize optimizer and logging
        self.optimizer = optim.Adam(self.policy.parameters(), 
                                  self.training_config['learning_rate'])
        self._initialize_logging()

    def _setup_networks(self):
        """Initialize neural networks."""
        self.policy = DQN().to(device)
        self.target = DQN().to(device)
        self.test_network = DQN().to(device)
        
        self.target.eval()
        self.test_network.eval()
        self.updateNetworks()

    def _setup_environment(self, drone, detector):
        """Setup the training environment."""
        self.env = Env(drone, detector)

    def _setup_memory(self):
        """Initialize experience replay memory."""
        self.memory = Memory(10000)

    def _setup_directories(self):
        """Create necessary directories."""
        cwd = os.getcwd()
        self.save_dir = os.path.join(cwd, "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(cwd, "train_videos"), exist_ok=True)
        os.makedirs(os.path.join(cwd, "test_videos"), exist_ok=True)

    def _load_checkpoint(self):
        """Load checkpoint if available."""
        files = glob.glob(self.save_dir + '\\*.pt')
        if len(files) > 0:
            files.sort(key=os.path.getmtime)
            file = files[-1]
            checkpoint = torch.load(file)
            self.policy.load_state_dict(checkpoint['state_dict'])
            self.training_state['episode'] = checkpoint['episode']
            self.training_state['steps_done'] = checkpoint['steps_done']
            self.updateNetworks()
            print("Saved parameters loaded"
                  "\nModel: ", file,
                  "\nSteps done: ", self.training_state['steps_done'],
                  "\nEpisode: ", self.training_state['episode'])

        else:
            if os.path.exists("log.txt"):
                open('log.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('last_episode.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('saved_model_params.txt', 'w').close()

    def _initialize_logging(self):
        """Initialize logging for tensorboard."""
        obs, _ = self.env.reset()
        tensor = self.transformToTensor(obs)
        writer.add_graph(self.policy, tensor)

    def updateNetworks(self):
        """Update target network."""
        self.target.load_state_dict(self.policy.state_dict())

    def transformToTensor(self, img):
        """Transform image to tensor."""
        tensor = torch.FloatTensor(img).to(device)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        return tensor

    def convert_size(self, size_bytes):
        """Convert size in bytes to a human-readable format."""
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    def act(self, state):
        """Select an action based on the current state."""
        self.training_state['eps_threshold'] = self.training_config['eps_end'] + (self.training_config['eps_start'] - self.training_config['eps_end']) * math.exp(
            -1.0 * self.training_state['steps_done'] / self.training_config['eps_decay']
        )
        self.training_state['steps_done'] += 1
        if random.random() > self.training_state['eps_threshold']:
            if torch.cuda.is_available():
                action = np.argmax(self.policy(state).cpu().data.squeeze().numpy())
            else:
                action = np.argmax(self.policy(state).data.squeeze().numpy())
        else:
            action = random.randrange(0, 8)
        return int(action)

    def append_sample(self, state, action, reward, next_state):
        """Append a sample to the experience replay memory."""
        next_state = self.transformToTensor(next_state)

        current_q = self.policy(state).squeeze().cpu().detach().numpy()[action]
        next_q = self.target(next_state).squeeze().cpu().detach().numpy()[action]
        expected_q = reward + (self.training_config['gamma'] * next_q)

        error = abs(current_q - expected_q),

        self.memory.add(error, state, action, reward, next_state)

    def learn(self):
        """Learn from a batch of experiences."""
        if self.memory.tree.n_entries < self.training_config['batch_size']:
            return

        states, actions, rewards, next_states, idxs, _ = self.memory.sample(self.training_config['batch_size'])

        states = tuple(states)
        next_states = tuple(next_states)

        states = torch.cat(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = torch.cat(next_states)

        current_q = self.policy(states)[range(self.training_config['batch_size']), actions]
        next_q =self.target(next_states).cpu().detach().numpy()[[range(0, self.training_config['batch_size'])], [actions]]
        expected_q = torch.FloatTensor(rewards + (self.training_config['gamma'] * next_q)).to(device)

        errors = torch.abs(current_q.squeeze() - expected_q.squeeze()).cpu().detach().numpy()

        # Update priority
        for i in range(self.training_config['batch_size']):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        loss = F.smooth_l1_loss(current_q.squeeze(), expected_q.squeeze())
        loss_val = loss.item()
        writer.add_scalar('loss', loss_val, self.training_state['episode'])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        """Train the DDQN agent."""
        print("Starting...")

        score_history = []
        reward_history = []
        image_array = []

        if self.training_state['episode'] == -1:
            self.training_state['episode'] = 1

        for e in range(1, self.training_config['max_episodes'] + 1):
            start = time.time()
            state, next_state_image = self.env.reset()
            image_array.append(next_state_image)
            steps = 0
            score = 0
            
            while True:
                state = self.transformToTensor(state)

                action = self.act(state)
                next_state, reward, done, next_state_image = self.env.step(action)
                print(f"Epside: {self.training_state['episode']}: Step {steps}")
                image_array.append(next_state_image)

                if steps == self.training_config['max_steps']:
                    done = 1

                self.append_sample(state, action, reward, next_state)
                self.learn()

                state = next_state
                steps += 1
                score += reward
                if done:
                    print("----------------------------------------------------------------------------------------")
                    if self.memory.tree.n_entries < self.training_config['batch_size']:
                        print("Training will start after ", self.training_config['batch_size'] - self.memory.tree.n_entries, " steps.")
                        break

                    print(
                        "episode: {0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}".format(
                            self.training_state['episode'], reward, round(score / steps, 2), score, self.training_state['eps_threshold'], self.training_state['steps_done']))
                    score_history.append(score)
                    reward_history.append(reward)
                    with open('log.txt', 'a') as file:
                        file.write(
                            "episode: {0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}\n".format(
                                self.training_state['episode'], reward, round(score / steps, 2), score, self.training_state['eps_threshold'],
                                self.training_state['steps_done']))

                    if torch.cuda.is_available():
                        print('Total Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory))
                        print('Allocated Memory:', self.convert_size(torch.cuda.memory_allocated(0)))
                        print('Cached Memory:', self.convert_size(torch.cuda.memory_reserved(0)))
                        print('Free Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory - (
                                torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved())))

                        memory_usage_allocated = np.float64(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
                        memory_usage_cached = np.float64(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))

                        writer.add_scalar("memory_usage_allocated", memory_usage_allocated, self.training_state['episode'])
                        writer.add_scalar("memory_usage_cached", memory_usage_cached, self.training_state['episode'])

                    writer.add_scalar('epsilon_value', self.training_state['eps_threshold'], self.training_state['episode'])
                    writer.add_scalar('score_history', score, self.training_state['episode'])
                    writer.add_scalar('reward_history', reward, self.training_state['episode'])
                    writer.add_scalar('Total steps', self.training_state['steps_done'], self.training_state['episode'])
                    writer.add_scalars('General Look', {'score_history': score,
                                                        'reward_history': reward}, self.training_state['episode'])

                    # Save checkpoint
                    if self.training_state['episode'] % self.intervals['save'] == 0:
                        checkpoint = {
                            'episode': self.training_state['episode'],
                            'steps_done': self.training_state['steps_done'],
                            'state_dict': self.policy.state_dict()
                        }
                        torch.save(checkpoint, self.save_dir + '//EPISODE{}.pt'.format(self.training_state['episode']))

                    # Convert images to video
                    frameSize = (640, 360)
                    video = cv2.VideoWriter("test_videos\\test_video_episode_{}_score_{}.avi".format(self.training_state['episode'], score), cv2.VideoWriter_fourcc(*'DIVX'), 7, frameSize)
                    
                    for img in image_array:
                        video.write(img)
                    
                    video.release()

                    if self.training_state['episode'] % self.intervals['network_update'] == 0:
                        self.updateNetworks()

                    self.training_state['episode'] += 1
                    end = time.time()
                    stopWatch = end - start
                    print("Episode is done, episode time: ", stopWatch)

                    if self.training_state['episode'] % self.intervals['test'] == 0:
                        self.test()

                    break
        writer.close()

    def test(self):
        """Test the DDQN agent."""
        self.test_network.load_state_dict(self.target.state_dict())

        start = time.time()
        steps = 0
        score = 0
        image_array = []
        state, next_state_image = self.env.reset()
        image_array.append(next_state_image)

        while True:
            state = self.transformToTensor(state)

            action = int(np.argmax(self.test_network(state).cpu().data.squeeze().numpy()))
            next_state, reward, done, next_state_image = self.env.step(action)
            image_array.append(next_state_image)

            if steps == self.training_config['max_steps']:
                done = 1

            state = next_state
            steps += 1
            score += reward

            if done:
                print("----------------------------------------------------------------------------------------")
                print("TEST, reward: {}, score: {}, total steps: {}".format(
                    reward, score, self.training_state['steps_done']))

                with open('tests.txt', 'a') as file:
                    file.write("TEST, reward: {}, score: {}, total steps: {}\n".format(
                        reward, score, self.training_state['steps_done']))

                writer.add_scalars('Test', {'score': score, 'reward': reward}, self.training_state['episode'])

                end = time.time()
                stopWatch = end - start
                print("Test is done, test time: ", stopWatch)

                # Convert images to video
                frameSize = (640, 360)
                video = cv2.VideoWriter("test_videos\\test_video_episode_{}_score_{}.avi".format(self.training_state['episode'], score), cv2.VideoWriter_fourcc(*'DIVX'), 7, frameSize)

                for img in image_array:
                    video.write(img)

                video.release()

                break


if __name__ == "__main__":
    ddqn_agent = DDQN_Agent()
    ddqn_agent.train()
