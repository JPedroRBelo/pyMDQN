# -*- coding: utf-8 -*-
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda")


Transition = namedtuple('Transition',
                        ('sgray','sdepth' 'action', 'next_sgray','next_sdepth', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# Now, let's define our model. But first, let quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# :math:`R_{t_0}` is also known as the *return*. The discount,
# :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. It makes rewards from the uncertain far
# future less important for our agent than the ones in the near future
# that it can be fairly confident about.
#
# The main idea behind Q-learning is that if we had a function
# :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# However, we don't know everything about the world, so we don't have
# access to :math:`Q^*`. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# :math:`Q^*`.
#
# For our training update rule, we'll use a fact that every :math:`Q`
# function for some policy obeys the Bellman equation:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# The difference between the two sides of the equality is known as the
# temporal difference error, :math:`\delta`:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))
#
# To minimise this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of :math:`Q` are very noisy. We calculate
# this over a batch of transitions, :math:`B`, sampled from the replay
# memory:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *expected return* of
# taking each action given the current input.
#



class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16, kernel_size=4,stride=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16,out_channels=32, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            )
        self.classifier = nn.Sequential(
            nn.Linear(64*2*2,256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(34*2*2)
        x = self.classifier(x)
        return x


######################################################################
# Input extraction
# ^^^^^^^^^^^^^^^^
#
# The code below are utilities for extracting and processing rendered
# images from the environment. It uses the ``torchvision`` package, which
# makes it easy to compose image transforms. Once you run the cell it will
# display an example patch that it extracted.
#

resize = T.Compose([T.ToPILImage(),
                    T.Resize((self.proc_frame_size,self.proc_frame_size), interpolation=Image.BILINEAR),
                    T.ToTensor()])



def get_screen():
   
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

BATCH_SIZE = 25
GAMMA = 0.999
EPS_START = 0.99
EPS_END = 0.1
EPS_DECAY = 1000
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = 4

gray_policy_net = DQN().to(device)
gray_target_net = DQN().to(device)
gray_target_net.load_state_dict(gray_target_net.state_dict())
gray_target_net.eval()

depth_policy_net = DQN().to(device)
depth_target_net = DQN().to(device)
depth_target_net.load_state_dict(depth_target_net.state_dict())
depth_target_net.eval()


gray_optimizer = optim.RMSprop(gray_policy_net.parameters())
depth_optimizer = optim.RMSprop(depth_policy_net.parameters())
memory = ReplayMemory(2000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            #colocar depth 
            action = gray_policy_net(state).max(1)[1].view(1, 1)
            return action
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []



######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    gray_non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_sgray)), device=device, dtype=torch.bool)
    gray_non_final_next_states = torch.cat([s for s in batch.next_sgray
                                                if s is not None])

    depth_non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_sdepth)), device=device, dtype=torch.bool)
    depth_non_final_next_states = torch.cat([s for s in batch.next_sdepth
                                                if s is not None])
    sgray_batch = torch.cat(batch.sgray)
    sdepth_batch = torch.cat(batch.sdepth)

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    sgray_action_values = gray_policy_net(sgray_batch).gather(1, action_batch)
    sdepth_action_values = depth_policy_net(sdepth_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_sgray_values = torch.zeros(BATCH_SIZE, device=device)
    next_sgray_values[gray_non_final_mask] = gray_target_net(gray_non_final_next_states).max(1)[0].detach()


    next_sdepth_values = torch.zeros(BATCH_SIZE, device=device)
    next_sdepth_values[depth_non_final_mask] = depth_target_net(depth_non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_sgray_action_values = (next_sgray_values * GAMMA) + reward_batch
    expected_sdepth_action_values = (next_sdepth_values * GAMMA) + reward_batch

    # Compute Huber loss
    gray_loss = F.smooth_l1_loss(sgray_action_values, expected_sgray_action_values.unsqueeze(1))
    depth_loss = F.smooth_l1_loss(sdepth_action_values, expected_sdepth_action_values.unsqueeze(1))

    # Optimize the model
    gray_optimizer.zero_grad()
    gray_loss.backward()
    for param in gray_policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    gray_optimizer.step()

    # Optimize the model
    depth_optimizer.zero_grad()
    depth_loss.backward()
    for param in depth_policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    depth_optimizer.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
#

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

######################################################################
# Here is the diagram that illustrates the overall resulting data flow.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# Actions are chosen either randomly or based on a policy, getting the next
# step sample from the gym environment. We record the results in the
# replay memory and also run optimization step on every iteration.
# Optimization picks a random batch from the replay memory to do training of the
# new policy. "Older" target_net is also used in optimization to compute the
# expected Q values; it is updated occasionally to keep it current.
#