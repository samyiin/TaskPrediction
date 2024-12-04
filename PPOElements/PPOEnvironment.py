"""
dependency: depend on state and action object.
"""

import numpy as np
import torch
from Utils.State import State
from Utils.Action import Action

class Environment:
    def __init__(self):
        """
        Initialize the PPO environment.
        to make it more generic, we could have just pass state_transition_function and reward_function as input
        But here we will just define everything in this file, we are not writing a training framework
        """
        pass

    def state_transition_function(self, state: State, action: Action):
        """
        For now we just randomly select a 0 in the state and make it 1
        :param state:
        :param action:
        :return:
        """
        # todo: Here will not work if we are on GPU, need to change
        state_tensor = state.as_tensor()  # Use the tensor representation
        zero_indices = torch.where(state_tensor == 0)[0]  # Find indices of zeros

        if zero_indices.numel() == 0:
            # No zeros to change, return the state as-is
            return state_tensor

        # Randomly pick one of the zero indices
        selected_index = zero_indices[torch.randint(len(zero_indices), (1,))].item()

        # Turn the selected zero into one
        state_tensor[selected_index] = 1
        return State(state_tensor)

    def reward_function(self, state, action):
        """
        For now we will just reward -1 until the last state
        :param state:
        :param action:
        :return:
        """
        return -1

    def step(self, state, action):
        """
        Given the current state and action, return the reward and next state.

        :param state: The current state.
        :param action: The action taken in the current state.
        :return: A tuple (reward, next_state).
        """
        reward = self.reward_function(state, action)
        next_state = self.state_transition_function(state, action)
        return reward, next_state
