"""
Somehow we didn't intend to, but trajectory doesn't depend on the "state" and "action", basically state and action can
be any kind and trajectory would still work fine.
But, we will just keep in mind that trajectory actually depend on them.
"""



class Trajectory:
    """
    A trajectory is a list of tuples (s1, a1, r1, s2)
    """
    def __init__(self):
        self.states = []       # List to store states (s1, s2, ...)
        self.actions = []      # List to store actions (a1, a2, ...)
        self.rewards = []      # List to store rewards (r1, r2, ...)
        self.next_states = []  # List to store next states (s2, s3, ...)
        self.dones = []        # List to store done flags (True if terminal state)

    def add_step(self, state, action, reward, next_state, done):
        """
        Adds a single step to the trajectory.
        :param state: Current state (s)
        :param action: Action taken (a)
        :param reward: Reward received (r)
        :param next_state: Next state after action (s')
        :param done: Boolean flag indicating if the episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear(self):
        """Clears the trajectory data."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def __len__(self):
        """Returns the number of steps in the trajectory."""
        return len(self.states)

    def get_data(self):
        """
        Returns the trajectory data as a dictionary.
        Useful for converting to other formats or processing.
        :return: Dictionary with keys 'states', 'actions', 'rewards', 'next_states', 'dones'.
        """
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
        }