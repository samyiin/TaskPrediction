import numpy as np
import torch

# todo: this is temporary
STATE_SIZE = 8

class State:
    def __init__(self, initial_values, device='cpu'):
        """
        Initialize a state as a PyTorch tensor of fixed size.

        :param initial_values: List or array-like object with initial values.
        :param device: The device to store the tensor ('cpu' or 'cuda').
        """
        self.vector = torch.tensor(initial_values, dtype=torch.float32, device=device)

    def __repr__(self):
        return f"State(vector={self.vector})"

    def __getitem__(self, index):
        """Access an element of the state vector."""
        return self.vector[index]

    def __setitem__(self, index, value):
        """Set an element of the state vector."""
        self.vector[index] = value

    def __len__(self):
        """Return the size of the state vector."""
        return len(self.vector)

    def as_tensor(self):
        """Return the state as a NumPy array."""
        return self.vector



