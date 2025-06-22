import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

# DQN Architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Loading Environment and Model
env = gym.make("LunarLander-v3", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = DQN(state_dim, action_dim)
q_net.load_state_dict(torch.load("lunarlander_dqn.pth"))
q_net.eval()

# Evalution loop
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        action = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state

print(f"Evaluation finished. Total reward: {total_reward:.2f}")
env.close()
