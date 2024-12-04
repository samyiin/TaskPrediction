from PPOElements.PPOEnvironment import Environment
from PPOElements.PPOCriticNetwork import CriticNetwork
from PPOElements.PPOActorNetwork import ActorNetwork
from Utils.State import State, STATE_SIZE
import torch


# create the initial state
vector = torch.zeros(STATE_SIZE)
initial_state = State(vector)

# create the environment
env = Environment()

# create the Actor network
actor_network = ActorNetwork()

# create the Critic Network
critic_network = CriticNetwork()

# use actor network to rollout for 5 steps
trajectory = actor_network.roll_out_trajectory(initial_state=initial_state, environment=env, T=5)

# use this trajectory to generate training data for critic network
state_tensors, targets = critic_network.calculate_critic_network_training_data(trajectory=trajectory, gamma=0.99,
                                                                               lambda_param=0.99)
# need to do some unsqueeze to get the right target, but wait for Nitay now
print(1)


