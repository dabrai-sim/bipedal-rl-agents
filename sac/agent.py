import numpy as np
import os
import pickle
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from gymnasium.core import Env
from Buffer import Buffer
from Network import Network
import torch.optim as optim 
from torch.distributions import Normal

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        shape: list,
        learningRate: float,
        device: T.device,
        logStdMin: float = -20,
        logStdMax: float = 2,
    ):
        super().__init__()
        self.device = device
        self.logStdMin = logStdMin
        self.logStdMax = logStdMax
        
        # Define architecture
        inputDim = shape[0]
        hiddenDim1 = shape[1]
        hiddenDim2 = shape[2]
        outputDim = shape[3]
        
        # Define network layers
        self.linear1 = nn.Linear(inputDim, hiddenDim1)
        self.linear2 = nn.Linear(hiddenDim1, hiddenDim2)
        
        # Mean and log standard deviation layers
        self.mean = nn.Linear(hiddenDim2, outputDim)
        self.logStd = nn.Linear(hiddenDim2, outputDim)
        
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
        
        # Get mean and log_std
        mean = self.mean(x)
        logStd = self.logStd(x)
        logStd = T.clamp(logStd, self.logStdMin, self.logStdMax)
        
        return mean, logStd
    
    def sample(self, state: T.Tensor) -> tuple:
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        mean, logStd = self.forward(state)
        std = logStd.exp()
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        
        # Use tanh squashing to ensure actions are in [-1, 1]
        action = T.tanh(x_t)
        
        # Compute log probability, accounting for the tanh squashing
        logProb = normal.log_prob(x_t) - T.log(1 - action.pow(2) + 1e-6)
        logProb = logProb.sum(1, keepdim=True)
        
        return action, logProb, mean
    
    def gradientDescentStep(self, loss: T.Tensor, retainGraph: bool = False) -> None:
        loss = loss.to(self.device)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retainGraph)
        
        # Check if any parameters or gradients are on CPU
        for param in self.parameters():
            if param.grad is not None and param.grad.device != self.device:
                param.grad = param.grad.to(self.device)
        
        self.optimizer.step()
    
    def to(self, device):
        self.device = device
        return super().to(device)

class Agent:
    def __init__(
        self,
        env: Env,
        learningRate: float,
        gamma: float,
        tau: float,
        alpha: float = 0.2,  # Temperature parameter for entropy
        autoTuneAlpha: bool = True,  # Whether to automatically tune alpha
        targetEntropyRatio: float = 0.5,  # Target entropy = -dim(action_space) * ratio
        shouldLoad: bool = True,
        saveFolder: str = "saved",
    ):
        self.observationDim = env.observation_space.shape[0]
        self.actionDim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        
        # SAC-specific parameters
        self.alpha = alpha  # Temperature parameter for entropy
        self.autoTuneAlpha = autoTuneAlpha
        
        # Check if the saveFolder path exists
        if not os.path.isdir(saveFolder):
            os.mkdir(saveFolder)
        self.envName = os.path.join(saveFolder, env.name + ".")
        name = self.envName
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        
        # Buffer initialization
        if shouldLoad and os.path.exists(name + "Replay"):
            self.buffer = pickle.load(open(name + "Replay", "rb"))
            # Make sure buffer is on the correct device
            if hasattr(self.buffer, 'device') and self.buffer.device != self.device:
                self.buffer.device = self.device
                # Move buffer tensors to correct device if needed
                self.buffer.states = self.buffer.states.to(self.device)
                self.buffer.actions = self.buffer.actions.to(self.device)
                self.buffer.rewards = self.buffer.rewards.to(self.device)
                self.buffer.nextStates = self.buffer.nextStates.to(self.device)
                self.buffer.doneFlags = self.buffer.doneFlags.to(self.device)
        else:
            self.buffer = Buffer(self.observationDim, self.actionDim, self.device)
        
        # Helper function to load critic models correctly
        def load_critic_model(file_path, model_class, *args):
            if shouldLoad and os.path.exists(file_path):
                # Load the model
                model = pickle.load(open(file_path, "rb"))
                # Move model to correct device
                model.to(self.device)
                # Update device attribute
                model.device = self.device
                # Recreate optimizer
                model.optimizer = optim.Adam(model.parameters(), lr=learningRate)
                return model
            else:
                # Create new model
                return model_class(*args)
        
        # Create a new GaussianPolicy actor regardless of whether we're loading or not
        # We don't try to convert the TD3 actor to SAC since they have different architectures
        print("Creating new SAC Gaussian policy actor...")
        self.actor = GaussianPolicy(
            [self.observationDim, 256, 256, self.actionDim],
            learningRate,
            self.device
        )
        
        # Critics initialization - we can reuse TD3 critics
        self.critic1 = load_critic_model(
            name + "Critic1",
            Network,
            [self.observationDim + self.actionDim, 256, 256, 1],
            nn.Identity,
            learningRate,
            self.device
        )
        
        self.critic2 = load_critic_model(
            name + "Critic2",
            Network,
            [self.observationDim + self.actionDim, 256, 256, 1],
            nn.Identity,
            learningRate,
            self.device
        )
        
        # Target networks initialization
        if shouldLoad and os.path.exists(name + "TargetCritic1"):
            self.targetCritic1 = pickle.load(open(name + "TargetCritic1", "rb"))
            self.targetCritic1.to(self.device)
            self.targetCritic1.device = self.device
            self.targetCritic1.optimizer = optim.Adam(self.targetCritic1.parameters(), lr=learningRate)
        else:
            self.targetCritic1 = deepcopy(self.critic1)
            
        if shouldLoad and os.path.exists(name + "TargetCritic2"):
            self.targetCritic2 = pickle.load(open(name + "TargetCritic2", "rb"))
            self.targetCritic2.to(self.device)
            self.targetCritic2.device = self.device
            self.targetCritic2.optimizer = optim.Adam(self.targetCritic2.parameters(), lr=learningRate)
        else:
            self.targetCritic2 = deepcopy(self.critic2)
        
        # Setup automatic entropy tuning
        if self.autoTuneAlpha:
            self.targetEntropy = -self.actionDim * targetEntropyRatio
            
            # Initialize log_alpha
            if shouldLoad and os.path.exists(name + "LogAlpha"):
                self.logAlpha = pickle.load(open(name + "LogAlpha", "rb"))
                self.logAlpha = self.logAlpha.to(self.device)
            else:
                self.logAlpha = T.zeros(1, requires_grad=True, device=self.device)
                
            # Initialize alpha optimizer
            self.alphaOptimizer = optim.Adam([self.logAlpha], lr=learningRate)

    def getDeterministicAction(self, state: np.ndarray) -> np.ndarray:
        """Returns the mean of the policy for deterministic evaluation"""
        state_tensor = T.from_numpy(state).float().to(self.device)
        
        with T.no_grad():
            _, _, mean = self.actor.sample(state_tensor)
        
        return mean.cpu().detach().numpy().flatten()  # Flatten to handle batch dimension

    def getSACAction(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Gets action according to SAC policy (stochastic or deterministic)"""
        state_tensor = T.from_numpy(state).float().to(self.device)
        
        with T.no_grad():
            if evaluate:
                # During evaluation, use the mean action (no exploration)
                _, _, action = self.actor.sample(state_tensor)
            else:
                # During training, sample from the policy
                action, _, _ = self.actor.sample(state_tensor)
        
        return action.cpu().detach().numpy().flatten()  # Flatten to handle batch dimension

    def update(self, miniBatchSize: int):
        # Check if buffer has enough samples
        if self.buffer.currentSize < miniBatchSize:
            return
            
        # Randomly sample a mini-batch from the replay buffer
        miniBatch = self.buffer.getMiniBatch(miniBatchSize)
        
        # Create tensors and ensure they're on the correct device
        states = miniBatch["states"].to(self.device)
        actions = miniBatch["actions"].to(self.device)
        rewards = miniBatch["rewards"].to(self.device)
        nextStates = miniBatch["nextStates"].to(self.device)
        dones = miniBatch["doneFlags"].to(self.device)
        
        # 1. Update critics
        # Compute targets using target critics
        with T.no_grad():
            nextActions, nextLogProbs, _ = self.actor.sample(nextStates)
            
            # Target Q-values
            q1NextTarget = self.targetCritic1.forward(T.hstack([nextStates, nextActions]))
            q2NextTarget = self.targetCritic2.forward(T.hstack([nextStates, nextActions]))
            qNextTarget = T.min(q1NextTarget, q2NextTarget)
            
            # For entropy-regularized SAC, subtract the entropy term
            if self.autoTuneAlpha:
                alpha = self.logAlpha.exp()
            else:
                alpha = self.alpha
                
            # Compute entropy-regularized target
            targets = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * (qNextTarget - alpha * nextLogProbs)
        
        # Compute current Q estimates
        q1 = self.critic1.forward(T.hstack([states, actions]))
        q2 = self.critic2.forward(T.hstack([states, actions]))
        
        # Compute critic losses
        q1Loss = F.mse_loss(q1, targets)
        q2Loss = F.mse_loss(q2, targets)
        
        # Update critics
        self.critic1.gradientDescentStep(q1Loss, True)
        self.critic2.gradientDescentStep(q2Loss)
        
        # 2. Update actor
        newActions, logProbs, _ = self.actor.sample(states)
        
        # Compute actor loss
        # The actor aims to maximize Q-value and entropy
        q1New = self.critic1.forward(T.hstack([states, newActions]))
        q2New = self.critic2.forward(T.hstack([states, newActions]))
        qNew = T.min(q1New, q2New)
        
        if self.autoTuneAlpha:
            alpha = self.logAlpha.exp()
        else:
            alpha = self.alpha
            
        actorLoss = (alpha * logProbs - qNew).mean()
        
        # Update actor
        self.actor.gradientDescentStep(actorLoss)
        
        # 3. Update alpha (temperature parameter)
        if self.autoTuneAlpha:
            alphaLoss = -(self.logAlpha * (logProbs + self.targetEntropy).detach()).mean()
            
            self.alphaOptimizer.zero_grad()
            alphaLoss.backward()
            self.alphaOptimizer.step()
        
        # 4. Update target networks
        self.updateTargetNetwork(self.targetCritic1, self.critic1)
        self.updateTargetNetwork(self.targetCritic2, self.critic2)

    def updateTargetNetwork(self, targetNetwork: Network, network: Network):
        with T.no_grad():
            for targetParameter, parameter in zip(
                targetNetwork.parameters(), network.parameters()
            ):
                targetParameter.data.copy_(
                    (1 - self.tau) * targetParameter.data + self.tau * parameter.data
                )

    def save(self):
        name = self.envName
        pickle.dump(self.buffer, open(name + "Replay", "wb"))
        pickle.dump(self.actor, open(name + "Actor", "wb"))
        pickle.dump(self.critic1, open(name + "Critic1", "wb"))
        pickle.dump(self.critic2, open(name + "Critic2", "wb"))
        pickle.dump(self.targetCritic1, open(name + "TargetCritic1", "wb"))
        pickle.dump(self.targetCritic2, open(name + "TargetCritic2", "wb"))
        
        if self.autoTuneAlpha:
            pickle.dump(self.logAlpha, open(name + "LogAlpha", "wb"))