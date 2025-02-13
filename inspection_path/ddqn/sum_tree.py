"""
Sum Tree Data Structure
Implements a binary tree where each parent node is the sum of its children.
Used for efficient prioritized sampling in experience replay memory.
"""

import numpy as np
from typing import Tuple, Any

# SumTree: a binary tree data structure where the parent's value is the sum of its children
class SumTree:
    """
    Binary tree data structure for prioritized experience replay.
    Allows O(log n) updates and sampling based on priorities.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the SumTree.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        # Tree structure
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        
        # Experience storage
        self.states = np.zeros(capacity, dtype=object)
        self.actions = np.zeros(capacity, dtype=object)
        self.rewards = np.zeros(capacity, dtype=object)
        self.next_states = np.zeros(capacity, dtype=object)
        
        # Tracking variables
        self.write_index = 0
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx: int, change: float):
        """
        Propagate the priority change up through parent nodes.
        
        Args:
            idx: Index of the node to start propagation from
            change: Change in priority value
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx: int, value: float) -> int:
        """
        Retrieve the leaf node index for a given priority value.
        
        Args:
            idx: Current node index
            value: Priority value to search for
            
        Returns:
            int: Index of the leaf node
        """
        left = 2 * idx + 1
        right = left + 1

        # Return current node if it's a leaf
        if left >= len(self.tree):
            return idx

        # Recursively search left or right subtree
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        """
        Get total priority sum.
        
        Returns:
            float: Sum of all priorities
        """
        return self.tree[0]

    # store priority and sample
    def add(self, priority: float, state: Any, action: Any, 
            reward: Any, next_state: Any):
        """
        Add a new experience with its priority.
        
        Args:
            priority: Priority value of the experience
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Calculate tree index
        tree_idx = self.write_index + self.capacity - 1
        
        # Store experience
        self.states[self.write_index] = state
        self.actions[self.write_index] = action
        self.rewards[self.write_index] = reward
        self.next_states[self.write_index] = next_state
        
        # Update tree
        self.update(tree_idx, priority)
        
        # Update write index
        self.write_index = (self.write_index + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    # update priority
    def update(self, idx: int, priority: float):
        """
        Update priority at a specific index.
        
        Args:
            idx: Index to update
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    # get priority and sample
    def get(self, value: float) -> Tuple[int, float, Any, Any, Any, Any]:
        """
        Get an experience using a priority value.
        
        Args:
            value: Priority value to search for
            
        Returns:
            tuple: (tree_idx, priority, state, action, reward, next_state)
        """
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1

        return (idx, 
                self.tree[idx],
                self.states[data_idx],
                self.actions[data_idx],
                self.rewards[data_idx],
                self.next_states[data_idx])
