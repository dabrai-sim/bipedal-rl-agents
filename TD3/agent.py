import numpy as np
import os
import pickle
import torch as T
import torch.nn as nn
from copy import deepcopy
from gymnasium.core import Env
from Buffer import Buffer
from Network import Network
import torch.optim as optim 

class Agent:
    def __init__(
        self,
        env: Env,
        learningRate: float,
        gamma: float,
        tau: float,
        shouldLoad: bool = True,
        saveFolder: str = "saved",
    ):
        self.observationDim = env.observation_space.shape[0]
        self.actionDim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        # check if the saveFolder path exists
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
        
        # Helper function to load models correctly
        def load_model(file_path, model_class, *args):
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
        
        # Actor initialization
        self.actor = load_model(
            name + "Actor", 
            Network,
            [self.observationDim, 256, 256, self.actionDim],
            nn.Tanh,
            learningRate,
            self.device
        )
        
        # Critics initialization
        self.critic1 = load_model(
            name + "Critic1",
            Network,
            [self.observationDim + self.actionDim, 256, 256, 1],
            nn.Identity,
            learningRate,
            self.device
        )
        
        self.critic2 = load_model(
            name + "Critic2",
            Network,
            [self.observationDim + self.actionDim, 256, 256, 1],
            nn.Identity,
            learningRate,
            self.device
        )
        
        # Target networks initialization - these are deeper copies of the main networks
        if shouldLoad and os.path.exists(name + "TargetActor"):
            self.targetActor = pickle.load(open(name + "TargetActor", "rb"))
            self.targetActor.to(self.device)
            self.targetActor.device = self.device
            self.targetActor.optimizer = optim.Adam(self.targetActor.parameters(), lr=learningRate)
        else:
            self.targetActor = deepcopy(self.actor)
            
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

    def getNoisyAction(self, state: np.ndarray, sigma: float) -> np.ndarray:
        deterministicAction = self.getDeterministicAction(state)
        noise = np.random.normal(0, sigma, deterministicAction.shape)
        return np.clip(deterministicAction + noise, -1, +1)

    def getDeterministicAction(self, state: np.ndarray) -> np.ndarray:
        # Explicitly create tensor and move to device
        state_tensor = T.from_numpy(state).float().to(self.device)
        # Forward pass with tensor on correct device
        with T.no_grad():  # No need to track gradients for inference
            actions = self.actor.forward(state_tensor)
        # Move back to CPU for numpy conversion
        return actions.cpu().detach().numpy()

    def update(
        self,
        miniBatchSize: int,
        trainingSigma: float,
        trainingClip: float,
        updatePolicy: bool,
    ):
        # Check if buffer has enough samples
        if self.buffer.currentSize < miniBatchSize:
            return
            
        # randomly sample a mini-batch from the replay buffer
        miniBatch = self.buffer.getMiniBatch(miniBatchSize)
        
        # create tensors and ensure they're on the correct device
        states = miniBatch["states"].to(self.device)
        actions = miniBatch["actions"].to(self.device)
        rewards = miniBatch["rewards"].to(self.device)
        nextStates = miniBatch["nextStates"].to(self.device)
        dones = miniBatch["doneFlags"].to(self.device)
        
        # compute the targets
        with T.no_grad():  # Targets don't need gradients
            targets = self.computeTargets(
                rewards, nextStates, dones, trainingSigma, trainingClip
            )
        
        # do a single step on each critic network
        Q1Loss = self.computeQLoss(self.critic1, states, actions, targets)
        self.critic1.gradientDescentStep(Q1Loss, True)
        
        Q2Loss = self.computeQLoss(self.critic2, states, actions, targets)
        self.critic2.gradientDescentStep(Q2Loss)
        
        if updatePolicy:
            # do a single step on the actor network
            policyLoss = self.computePolicyLoss(states)
            self.actor.gradientDescentStep(policyLoss)
            
            # update target networks
            self.updateTargetNetwork(self.targetActor, self.actor)
            self.updateTargetNetwork(self.targetCritic1, self.critic1)
            self.updateTargetNetwork(self.targetCritic2, self.critic2)

    def computeTargets(
        self,
        rewards: T.Tensor,
        nextStates: T.Tensor,
        dones: T.Tensor,
        trainingSigma: float,
        trainingClip: float,
    ) -> T.Tensor:
        # Ensure inputs are on the correct device
        rewards = rewards.to(self.device)
        nextStates = nextStates.to(self.device)
        dones = dones.to(self.device)
        
        targetActions = self.targetActor.forward(nextStates.float())
        # create additive noise for target actions
        noise = T.normal(0, trainingSigma, targetActions.shape, device=self.device)
        clippedNoise = T.clip(noise, -trainingClip, +trainingClip)
        targetActions = T.clip(targetActions + clippedNoise, -1, +1)
        
        # compute targets
        input1 = T.hstack([nextStates, targetActions]).float()
        input2 = T.hstack([nextStates, targetActions]).float()
        
        targetQ1Values = T.squeeze(self.targetCritic1.forward(input1))
        targetQ2Values = T.squeeze(self.targetCritic2.forward(input2))
        
        targetQValues = T.minimum(targetQ1Values, targetQ2Values)
        return rewards + self.gamma * (1 - dones) * targetQValues

    def computeQLoss(
        self, network: Network, states: T.Tensor, actions: T.Tensor, targets: T.Tensor
    ) -> T.Tensor:
        # Ensure inputs are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        targets = targets.to(self.device)
        
        # compute the MSE of the Q function with respect to the targets
        combined_input = T.hstack([states, actions]).float()
        QValues = T.squeeze(network.forward(combined_input))
        return T.square(QValues - targets).mean()

    def computePolicyLoss(self, states: T.Tensor):
        # Ensure states are on the correct device
        states = states.to(self.device)
        
        actions = self.actor.forward(states.float())
        combined_input = T.hstack([states, actions]).float()
        QValues = T.squeeze(self.critic1.forward(combined_input))
        return -QValues.mean()

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
        pickle.dump(self.targetActor, open(name + "TargetActor", "wb"))
        pickle.dump(self.targetCritic1, open(name + "TargetCritic1", "wb"))
        pickle.dump(self.targetCritic2, open(name + "TargetCritic2", "wb"))