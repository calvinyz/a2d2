"""
Prioritized Experience Replay Memory
Implements prioritized experience replay for DDQN using a sum tree data structure.
Stores and samples experiences based on their TD error priority.
"""

import random
import numpy as np
from pathplanner.inspection.DDQN.sum_tree import SumTree

class Memory:
    """Prioritized Experience Replay Memory implementation."""
    
    # Hyperparameters for prioritized sampling
    PRIORITY_CONFIG = {
        'epsilon': 0.01,  # Small constant to ensure non-zero priority
        'alpha': 0.6,     # Priority exponent (how much prioritization to use)
        'beta_start': 0.4,  # Initial importance sampling weight
        'beta_increment': 0.001  # Increment for importance sampling weight
    }
    
    def __init__(self, capacity: int):
        """
        Initialize the memory buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.beta = self.PRIORITY_CONFIG['beta_start']

    def _get_priority(self, error: float) -> float:
        """
        Calculate priority value for an experience.
        
        Args:
            error: TD error of the experience
            
        Returns:
            float: Priority value
        """
        return (np.abs(error) + self.PRIORITY_CONFIG['epsilon']) ** self.PRIORITY_CONFIG['alpha']

    def add(self, error: float, state, action, reward, next_state):
        """
        Add an experience to memory.
        
        Args:
            error: TD error of the experience
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        priority = self._get_priority(error)
        self.tree.add(priority, state, action, reward, next_state)

    def sample(self, n: int) -> tuple:
        """
        Sample n experiences from memory using prioritized sampling.
        
        Args:
            n: Number of experiences to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, indices, importance_weights)
        """
        # Initialize storage for sampled experiences
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'indices': [],
            'priorities': []
        }

        # Calculate segment size for sampling
        segment_size = self.tree.total() / n
        
        # Update beta parameter for importance sampling
        self.beta = min(1.0, self.beta + self.PRIORITY_CONFIG['beta_increment'])

        # Sample experiences from each segment
        for i in range(n):
            segment_start = segment_size * i
            segment_end = segment_size * (i + 1)
            sample_value = random.uniform(segment_start, segment_end)
            
            # Get experience from tree
            idx, priority, state, action, reward, next_state = self.tree.get(sample_value)
            
            # Store sampled experience
            batch['states'].append(state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['next_states'].append(next_state)
            batch['indices'].append(idx)
            batch['priorities'].append(priority)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(batch['priorities']) / self.tree.total()
        importance_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        importance_weights /= importance_weights.max()

        return (
            batch['states'],
            batch['actions'],
            batch['rewards'],
            batch['next_states'],
            batch['indices'],
            importance_weights
        )

    def update(self, idx: int, error: float):
        """
        Update priority of experience in memory.
        
        Args:
            idx: Index of experience to update
            error: New TD error
        """
        priority = self._get_priority(error)
        self.tree.update(idx, priority)
