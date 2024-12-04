"""
dependencies: This file depends on State, Action and Trajectory Object
"""

import torch
import torch.nn as nn
import torch.optim as optim
from Utils.State import STATE_SIZE


class CriticNetwork(nn.Module):
    """
    The critic network take a state as input and return V(s) as output
    The state we define so far is a vector of "state_dim" (one dimension)
    This can be changed later
    """

    def __init__(self, state_dim=STATE_SIZE, device='cpu'):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 1)  # Output layer

        self.activation = nn.ReLU()  # Non-linearity

        # more attributes for later use

        self.device = device
        self.value_estimates = []

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        value = self.fc3(x)  # Output a single scalar
        return value

    def _calculate_trajectory_advantages(self, trajectory, gamma):
        """
        Calculate the discounted accumulated advantages for a trajectory.
        The trajectory is only rolled out for T steps, so the advantage of last step is not necessary 0.
        For detail math look at PPO actor-critic notebook

        :param trajectory: A Trajectory object containing states, rewards, and other data.
        :param gamma: Discount factor for future rewards.
        :return: List of advantages for each step in the trajectory.
        """
        advantages = []
        g_t = 0  # Initialize cumulative reward

        # Get value estimates for all states in the trajectory
        self.value_estimates = [self.forward(state.as_tensor()) for state in trajectory.states]

        # Compute advantages backward
        for i in reversed(range(len(trajectory.rewards))):
            g_t = trajectory.rewards[i] + gamma * g_t
            advantage = g_t - self.value_estimates[i]
            advantages.insert(0, advantage)  # Prepend to maintain order

        # todo: Not so sure about this part: add the last discounted factor
        last_state_value = self.forward(trajectory.next_states[-1].as_tensor()) if len(
            trajectory.next_states) > 0 else 0.0
        discounted_last_state_value = last_state_value
        for i in reversed(range(len(advantages))):
            advantages[i] += discounted_last_state_value
            discounted_last_state_value = gamma * discounted_last_state_value

        return advantages

    def calculate_critic_network_training_data(self, trajectory, gamma, lambda_param):
        """
        Generate training data for the critic network based on the trajectory and calculated advantages.

        :param gamma:
        :param trajectory: A Trajectory object containing states, rewards, and other data.
        :param lambda_param: Scaling factor for the advantages (lambda in R_t = V(s_t) + lambda * A_t).
        :return: A tuple (states, targets) where:
                 - states: List of states from the trajectory.
                 - targets: Computed R_t values for each state.
        """
        advantages = self._calculate_trajectory_advantages(trajectory, gamma)
        state_tensors = [state.as_tensor() for state in trajectory.states]
        # self.value_estimates already calculated in self._calculate_trajectory_advantages
        targets = [v + lambda_param * a for v, a in zip(self.value_estimates, advantages)]
        # turn to tensor
        state_tensors = torch.stack(state_tensors),
        targets = torch.stack(targets),

        return state_tensors, targets

    def train_nn(self, inputs, targets, learning_rate=0.001, epochs=10):
        """
        Train the critic network using inputs and targets.

        :param inputs: Tensor of input states (batch_size, input_dim).
        :param targets: Tensor of target values (batch_size, 1).
        :param learning_rate: Learning rate for the optimizer.
        :param epochs: Number of epochs to train for.
        """
        assert inputs.size(0) == targets.size(0), "Inputs and targets must have the same batch size."

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.train()  # Set the network to training mode

            # Forward pass
            predictions = self.forward(inputs)
            loss = criterion(predictions, targets)

            # Backward pass
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Print loss for debugging (optional)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
