# Implementation of Deep Q-learning using OpenAI gym, see https://www.youtube.com/watch?v=NP8pXZdU-5U

import numpy as np
import random
import torch
from torch import nn
import itertools
import gym
from collections import deque

# CONSTANTS
GAMMA=0.99              # discount rate for computing the TD target
BATCH_SIZE=32           # how many transitions we will sample
BUFFER_SIZE=50000       # max number of transitions to store before overwriting
MIN_REPLAY_SIZE=1000.   # how many transitions in the replay buffer before doing gradients
EPSILON_START=1.0       # starting value of epsilon
EPSILON_END=0.82        # ending value of epsilon
EPSILON_DECAY=10000     # number of steps to decrease epsilon to final value
TARGET_UPDATE_FREQ=1000 # freqeuncy of setting the target parameters to the online parameters
LEARNING_RATE=5e-4      # learning rate
REWARD_THRESHOLD=20     # after mean reward reaches this threshold, we stop training and let the computer play

# NN MODEL
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape)) # number of neurons in the input layer
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )
    
    def forward(self, x):
        return self.net(x)
        
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0)) # reason for the unsqueeze function is that we need to created a fake batch dimension since the tensor's 0th dimension is expected to be the batch size
        
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item() # choosing the action with the highest Q value
        
        return action

env = gym.make('CartPole-v0', render_mode='rgb_array')
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.], maxlen=100)

episode_reward = 0.

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

# INITIALIZE REPLAY BUFFER
obs = env.reset()
obs = np.array(obs if isinstance(obs, (list, np.ndarray)) else obs[0])

for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, done, _, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    
    if done:
        obs = env.reset()
        obs = np.array(obs if isinstance(obs, (list, np.ndarray)) else obs[0])


# MAIN TRAINING LOOP
obs = env.reset()
obs = np.array(obs if isinstance(obs, (list, np.ndarray)) else obs[0])


for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()
    
    if rnd_sample <=  epsilon:
        # Exploration
        action = env.action_space.sample()
    else:
        # Exploitation
        action = online_net.act(obs)
    
    new_obs, rew, done, _, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    episode_reward += rew
    
    if done:
        obs = env.reset()
        obs = np.array(obs if isinstance(obs, (list, np.ndarray)) else obs[0])
        rew_buffer.append(episode_reward)
        episode_reward = 0.
    
    # PLAY THE GAME AFTER SOLVED
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= REWARD_THRESHOLD:
            while True:
                action = online_net.act(obs)
                print(action)
                obs, _, done, _, _ = env.step(action)
                obs = np.array(obs if isinstance(obs, (list, np.ndarray)) else obs[0])
                env.render()
                if done:
                    env.reset()
                    
    
    # GRADIENT STEP
    transitions = random.sample(replay_buffer, BATCH_SIZE) # sample random transitions
    
    obses = np.array([t[0] for t in transitions])
    obses = np.array([np.array(o) if isinstance(o, (list, np.ndarray)) else np.array(o[0]) for o in obses])
    actions = np.array([t[1] for t in transitions])
    rews = np.array([t[2] for t in transitions])
    dones = np.array([t[3] for t in transitions])
    new_obses = np.array([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)


    # CALCULATE TARGETS FOR THE LOSS FUNCTION
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values


    # COMPUTE LOSS
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t) # Q-values for the action that we took at the time of the transition
                                                                           # The gather function applies the index=actions_t in dimension=1 to the tensor q_values
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)          # Huber loss (as mentioned in the discussions about the paper on GitHub


    # GRADIENT DESCENT STEP
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # UPDATE TARGET NETWORK
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
        
    # LOGGING
    if step % 1000 == 0:
        print()
        print("Step: ", step)
        print("Avg reward: ", np.mean(rew_buffer))

