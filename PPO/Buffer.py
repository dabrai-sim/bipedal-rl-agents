# Buffer.py
import numpy as np
import torch as T

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent, with support
    for calculating advantages using Generalized Advantage Estimation (GAE).
    """
    def __init__(
        self,
        observationDim: int,
        actionDim: int,
        buffer_size: int,
        device: T.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        
        # Initialize buffers for each component
        self.states = T.zeros((buffer_size, observationDim), device=self.device)
        self.actions = T.zeros((buffer_size, actionDim), device=self.device)
        self.logProbs = T.zeros((buffer_size, 1), device=self.device)
        self.rewards = T.zeros((buffer_size, 1), device=self.device)
        self.doneFlags = T.zeros((buffer_size, 1), device=self.device)
        self.values = T.zeros((buffer_size, 1), device=self.device)
        
        # Pointer to keep track of where in the buffer we are
        self.pointer = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
    
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        logProb: np.ndarray,
        doneFlag: bool
    ):
        """Store a transition in the buffer"""
        idx = self.pointer % self.max_size
        
        # Convert numpy arrays to tensors and move them to device
        state_tensor = T.from_numpy(state).float().to(self.device)
        action_tensor = T.from_numpy(action).float().to(self.device)
        
        # Store the components
        self.states[idx] = state_tensor
        self.actions[idx] = action_tensor
        
        # Handle reward conversion
        if isinstance(reward, (np.ndarray, np.float32, np.float64)):
            self.rewards[idx] = T.tensor([reward], device=self.device)
        else:
            self.rewards[idx] = reward
        
        # Handle logProb conversion - ensure it's a tensor on the right device
        if isinstance(logProb, np.ndarray):
            logProb_tensor = T.from_numpy(logProb).float().to(self.device)
            self.logProbs[idx] = logProb_tensor
        else:
            # If it's already a tensor, ensure it's on the right device
            self.logProbs[idx] = logProb.to(self.device) if hasattr(logProb, 'to') else T.tensor([logProb], device=self.device)
        
        # Handle value conversion
        if isinstance(value, (np.ndarray, np.float32, np.float64)):
            value_tensor = T.tensor([float(value)], device=self.device)
            self.values[idx] = value_tensor
        elif hasattr(value, 'to') and hasattr(value, 'device'):  # If it's already a tensor
            self.values[idx] = value.to(self.device)
        else:
            self.values[idx] = T.tensor([float(value)], device=self.device)
        
        # Convert done flag to tensor
        self.doneFlags[idx] = float(doneFlag)
        
        # Update pointer
        self.pointer += 1
    
    def finish_path(self, last_value=0):
        """
        Compute advantages and returns for a completed trajectory.
        
        Args:
            last_value: The value estimate for the last state, used for bootstrapping.
        """
        path_slice = slice(self.path_start_idx, self.pointer)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.doneFlags[path_slice]
        
        # Convert last_value to tensor if needed and move to device
        if isinstance(last_value, (float, int, np.float32, np.float64)):
            last_value_tensor = T.tensor([[float(last_value)]], device=self.device)
        elif isinstance(last_value, np.ndarray):
            last_value_tensor = T.from_numpy(last_value).float().to(self.device).view(1, 1)
        elif hasattr(last_value, 'to') and hasattr(last_value, 'device'):
            last_value_tensor = last_value.to(self.device).view(1, 1)
        else:
            # Handle case where it might already be a tensor but needs reshaping
            last_value_tensor = last_value.view(1, 1) if hasattr(last_value, 'view') else T.tensor([[0.0]], device=self.device)
            
        # Append last value for incomplete trajectory
        values_plus = T.cat([values, last_value_tensor], dim=0)
        
        # Initialize advantages
        advantages = T.zeros_like(rewards)
        lastGAE = 0
        
        # Compute GAE advantages going backwards
        for t in reversed(range(len(rewards))):
            # If done, no bootstrapping
            next_nonterminal = 1.0 - dones[t]
            next_value = values_plus[t + 1]
            
            # Delta = r + gamma * V(s') * (1 - done) - V(s)
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            
            # GAE(s,a) = delta + gamma * lambda * (1 - done) * GAE(s',a')
            advantages[t] = lastGAE = delta + self.gamma * self.gae_lambda * next_nonterminal * lastGAE
        
        # Compute returns for simplicity (not actually needed with GAE)
        returns = advantages + values
        
        # Update path start index
        self.path_start_idx = self.pointer
        
        return advantages, returns
    
    def getPPOBatch(self):
        """Get all data currently in buffer for PPO update"""
        # Get valid slice from buffer
        valid_size = min(self.pointer, self.max_size)
        slice_indices = slice(0, valid_size)
        
        # Extract data
        states = self.states[slice_indices]
        actions = self.actions[slice_indices]
        old_logProbs = self.logProbs[slice_indices]
        
        # Compute returns and advantages
        advantages = T.zeros_like(self.rewards[slice_indices])
        returns = T.zeros_like(self.rewards[slice_indices])
        
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        
        # Compute GAE in reverse
        for i in reversed(range(valid_size)):
            next_non_terminal = 1.0 - self.doneFlags[i]
            next_values = prev_value if i == valid_size - 1 else self.values[i + 1]
            
            # Compute TD error (delta)
            delta = self.rewards[i] + self.gamma * next_values * next_non_terminal - self.values[i]
            
            # Update advantages
            advantages[i] = prev_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * prev_advantage
            
            # Compute returns
            returns[i] = prev_return = self.rewards[i] + self.gamma * prev_return * next_non_terminal
        
        # Reset buffer after using it
        self.pointer = 0
        self.path_start_idx = 0
        
        return states, actions, old_logProbs, returns, advantages