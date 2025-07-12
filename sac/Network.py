from typing import Callable
import torch as T
import torch.nn as nn
import torch.optim as optim


class Network(nn.Module):
    def __init__(
        self,
        shape: list,
        outputActivation: Callable,
        learningRate: float,
        device: T.device,
    ):
        super().__init__()
        self.device = device
        
        # initialize the network
        layers = []
        for i in range(1, len(shape)):
            dim1 = shape[i - 1]
            dim2 = shape[i]
            layers.append(nn.Linear(dim1, dim2))
            if i < len(shape) - 1:
                layers.append(nn.ReLU())
        layers.append(outputActivation())
        
        # Create the network and move it to the device
        self.network = nn.Sequential(*layers)
        self.to(self.device)
        
        # Create optimizer AFTER moving model to device
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)

    def forward(self, state: T.Tensor) -> T.Tensor:
        # Make sure state is on the correct device
        state = state.to(self.device)
        return self.network(state)

    def gradientDescentStep(self, loss: T.Tensor, retainGraph: bool = False) -> None:
        # Make sure loss is on the correct device
        loss = loss.to(self.device)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retainGraph)
        
        # Check if any parameters or gradients are on CPU
        for param in self.parameters():
            if param.grad is not None and param.grad.device != self.device:
                param.grad = param.grad.to(self.device)
        
        self.optimizer.step()
        
    def to(self, device):
        # Override the to() method to also update the device attribute
        self.device = device
        return super().to(device)