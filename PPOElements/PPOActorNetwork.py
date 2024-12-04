"""
dependencies: This file depends on Trajectory Object, and Environment object
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PPOElements.PPOEnvironment import Environment
from Utils.Action import Action, ACTION_SPACE_SIZE
from Utils.State import State, STATE_SIZE
from Utils.Trajectory import Trajectory
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, input_dim=STATE_SIZE, output_dim=ACTION_SPACE_SIZE, hidden_dim=128):
        """
        Initialize the Actor network.

        :param input_dim: Dimension of the input state vector.
        :param output_dim: Dimension of the output action space.
                           - For discrete actions, output_dim is the number of actions.
                           - For continuous actions, output_dim is the size of the action vector.
        :param hidden_dim: Number of units in the hidden layers (default: 128).
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, state, soft_max=True):
        """
        Forward pass through the network.

        :param soft_max:
        :param state: Input state vector (batch_size, input_dim).
        :return: Action probabilities for discrete action space
                 or action means for continuous action space.
        """
        x = F.relu(self.fc1(state))  # First hidden layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second hidden layer with ReLU activation
        if soft_max:
            action_output = F.softmax(self.fc3(x), dim=-1)  # Output layer with softmax for probabilities
            return action_output
        else:
            return self.fc3(x)

    def roll_out_trajectory(self, initial_state: State, environment: Environment, T: int) -> Trajectory:
        """
        Roll out a trajectory of length T For now assume T < 5

        :param initial_state:
        :param environment:
        :return: Trajectory object
        """
        trajectory = Trajectory()
        state = initial_state


        for t in range(T):
            # Convert state to tensor and get action probabilities
            state_tensor = state.as_tensor()
            action_probs = self.forward(state_tensor, soft_max=True).squeeze(0).detach().numpy()

            # Sample an action from the probabilities
            action = np.random.choice(len(action_probs), p=action_probs)
            action_obj = Action(action)

            # Interact with the environment
            reward, next_state = environment.step(state, action_obj)

            # Add to the trajectory
            trajectory.add_step(state, action_obj, reward, next_state, done=False)

            # Update state
            state = next_state

        return trajectory
