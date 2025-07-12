# agent.py
import numpy as np
import os
import pickle
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.core import Env
from Buffer import PPOBuffer
import torch.optim as optim 
from torch.distributions import Normal

class PPOPolicy(nn.Module):
    def __init__(
        self,
        shape: list,
        learningRate: float,
        device: T.device,
        outputActivation=nn.Tanh,  # PPO often uses tanh for bounded actions
    ):
        super().__init__()
        self.device = device
        
        # Define architecture
        inputDim = shape[0]
        hiddenDim1 = shape[1]
        hiddenDim2 = shape[2]
        outputDim = shape[3]
        
        # Define network layers for the policy
        self.linear1 = nn.Linear(inputDim, hiddenDim1)
        self.linear2 = nn.Linear(hiddenDim1, hiddenDim2)
        
        # Mean and log standard deviation layers
        self.mean = nn.Linear(hiddenDim2, outputDim)
        self.logStd = nn.Parameter(T.zeros(outputDim, device=device))
        
        # Output activation (typically tanh for bounded actions)
        self.outputActivation = outputActivation()
        
        # Move to device and create optimizer
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
    
    def forward(self, state: T.Tensor) -> tuple:
        # Ensure state is on correct device
        state = state.to(self.device)
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Pass through shared layers
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        # Get mean
        mean = self.mean(x)
        mean = self.outputActivation(mean)  # Apply action bounds
        
        # Use parameter for log_std (fixed std for all states)
        # This is a common simplification in PPO implementations
        logStd = self.logStd.expand_as(mean)
        std = T.exp(logStd)
        
        return mean, std
    
    def sample(self, state: T.Tensor) -> tuple:
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        mean, std = self.forward(state)
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample actions
        action = dist.sample()
        
        # Compute log probability
        logProb = dist.log_prob(action).sum(1, keepdim=True)
        
        return action, logProb, dist.entropy().sum(1, keepdim=True)
    
    def evaluate(self, state: T.Tensor, action: T.Tensor) -> tuple:
        """Evaluate log probability and entropy for given states and actions"""
        mean, std = self.forward(state)
        
        # Create distribution
        dist = Normal(mean, std)
        
        # Compute log probability
        logProb = dist.log_prob(action).sum(1, keepdim=True)
        
        # Compute entropy
        entropy = dist.entropy().sum(1, keepdim=True)
        
        return logProb, entropy
    
    def gradientDescentStep(self, loss: T.Tensor) -> None:
        loss = loss.to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping (common in PPO)
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        
        # Check if any parameters or gradients are on CPU
        for param in self.parameters():
            if param.grad is not None and param.grad.device != self.device:
                param.grad = param.grad.to(self.device)
        
        self.optimizer.step()
    
    def to(self, device):
        self.device = device
        return super().to(device)

class ValueNetwork(nn.Module):
    """Value function estimator for PPO"""
    def __init__(
        self,
        shape: list,
        learningRate: float,
        device: T.device,
    ):
        super().__init__()
        self.device = device
        
        # Define architecture
        inputDim = shape[0]
        hiddenDim1 = shape[1]
        hiddenDim2 = shape[2]
        
        # Define network layers for the value function
        self.linear1 = nn.Linear(inputDim, hiddenDim1)
        self.linear2 = nn.Linear(hiddenDim1, hiddenDim2)
        self.value = nn.Linear(hiddenDim2, 1)
        
        # Move to device and create optimizer
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
    
    def forward(self, state: T.Tensor) -> T.Tensor:
        # Ensure state is on correct device
        state = state.to(self.device)
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Pass through network
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        value = self.value(x)
        
        return value
    
    def gradientDescentStep(self, loss: T.Tensor) -> None:
        loss = loss.to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        
        # Check if any parameters or gradients are on CPU
        for param in self.parameters():
            if param.grad is not None and param.grad.device != self.device:
                param.grad = param.grad.to(self.device)
        
        self.optimizer.step()
    
    def to(self, device):
        self.device = device
        return super().to(device)

class PPOAgent:
    def __init__(
        self,
        env: Env,
        learningRate: float,
        gamma: float,
        gae_lambda: float = 0.95,  # GAE parameter
        clip_ratio: float = 0.2,   # PPO clip parameter
        shouldLoad: bool = True,
        saveFolder: str = "saved",
        ppo_epochs: int = 10,      # Number of PPO update epochs
        value_coef: float = 0.5,   # Value loss coefficient
        entropy_coef: float = 0.01 # Entropy coefficient
    ):
        self.observationDim = env.observation_space.shape[0]
        self.actionDim = env.action_space.shape[0]
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Check if the saveFolder path exists
        if not os.path.isdir(saveFolder):
            os.mkdir(saveFolder)
        self.envName = os.path.join(saveFolder, env.name + ".")
        name = self.envName
        
        # Better GPU detection with informative output
        if T.cuda.is_available():
            self.device = T.device("cuda")
            print(f"Using GPU: {T.cuda.get_device_name(0)}")
            print(f"GPU memory allocated: {T.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU memory reserved: {T.cuda.memory_reserved() / 1e9:.2f} GB")
        else:
            self.device = T.device("cpu")
            print("CUDA not available, using CPU instead")
        
        # Buffer initialization for PPO
        self.buffer = PPOBuffer(self.observationDim, self.actionDim, 2048, self.device, gamma, gae_lambda)
        
        # Create policy network (actor)
        if shouldLoad and os.path.exists(name + "PPOPolicy"):
            print("Loading saved PPO policy...")
            self.policy = pickle.load(open(name + "PPOPolicy", "rb"))
            self.policy.to(self.device)
            self.policy.device = self.device
            self.policy.optimizer = optim.Adam(self.policy.parameters(), lr=learningRate)
        else:
            print("Creating new PPO policy...")
            self.policy = PPOPolicy(
                [self.observationDim, 256, 256, self.actionDim],
                learningRate,
                self.device
            )
        
        # Create value network (critic)
        if shouldLoad and os.path.exists(name + "PPOValue"):
            print("Loading saved PPO value function...")
            self.value = pickle.load(open(name + "PPOValue", "rb"))
            self.value.to(self.device)
            self.value.device = self.device
            self.value.optimizer = optim.Adam(self.value.parameters(), lr=learningRate)
        else:
            print("Creating new PPO value function...")
            self.value = ValueNetwork(
                [self.observationDim, 256, 256],
                learningRate,
                self.device
            )

    def getAction(self, state: np.ndarray, evaluate: bool = False) -> tuple:
        """Get action from policy"""
        state_tensor = T.from_numpy(state).float().to(self.device)
        
        with T.no_grad():
            if evaluate:
                # During evaluation, use the mean action (no exploration)
                mean, _ = self.policy.forward(state_tensor)
                return mean.cpu().detach().numpy().flatten()
            else:
                # During training, sample from the policy
                action, logProb, _ = self.policy.sample(state_tensor)
                return action.cpu().detach().numpy().flatten(), logProb.cpu().detach().numpy().flatten()

    def value_estimate(self, state: np.ndarray) -> float:
        """Get value estimate for state"""
        state_tensor = T.from_numpy(state).float().to(self.device)
        
        with T.no_grad():
            value = self.value(state_tensor)
            # Keep as tensor for buffer storage
            return value

    def update(self):
        """Update policy and value function using PPO algorithm"""
        # Get data from buffer
        states, actions, old_logProbs, returns, advantages = self.buffer.getPPOBatch()
        
        # Normalize advantages (helps with training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch size for updates
        batch_size = min(64, states.size(0))
        
        # PPO update epochs
        for _ in range(self.ppo_epochs):
            # Generate random indices for mini-batches
            indices = T.randperm(states.size(0), device=self.device)
            
            # Process mini-batches
            for start_idx in range(0, states.size(0), batch_size):
                # Get mini-batch indices
                idx = indices[start_idx:start_idx + batch_size]
                
                # Extract mini-batch data
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_logProbs = old_logProbs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Compute current log probs and entropy
                logProbs, entropy = self.policy.evaluate(mb_states, mb_actions)
                
                # Compute value predictions
                values = self.value(mb_states)
                
                # Compute ratio for importance sampling
                ratio = T.exp(logProbs - mb_old_logProbs)
                
                # Compute surrogate objectives
                surr1 = ratio * mb_advantages
                surr2 = T.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                
                # PPO's clipped objective function
                policy_loss = -T.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy bonus for exploration
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update policy
                self.policy.gradientDescentStep(loss)
                
                # Update value function separately
                value_loss = F.mse_loss(self.value(mb_states), mb_returns)
                self.value.gradientDescentStep(value_loss)
            
            # Optional: Synchronize CUDA operations
            if self.device.type == 'cuda':
                T.cuda.synchronize()
                
    def save(self):
        """Save the policy and value networks"""
        name = self.envName
        pickle.dump(self.policy, open(name + "PPOPolicy", "wb"))
        pickle.dump(self.value, open(name + "PPOValue", "wb"))