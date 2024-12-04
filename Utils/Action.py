# for now assume there are 50 actions(words) to chose from
ACTION_SPACE_SIZE = 50


class Action:
    def __init__(self, value):
        """
        Initialize the Action with an integer value.

        :param value: The integer representing the action.
        """
        if not isinstance(value, int):
            raise ValueError("Action must be an integer.")
        self.value = value

    def __repr__(self):
        return f"Action(value={self.value})"

    def get_action(self):
        return self.value