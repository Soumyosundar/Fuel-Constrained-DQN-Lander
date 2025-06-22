import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

env = gym.make("LunarLander-v3", render_mode="human")

mars_gravity = -3.71
env.unwrapped.world.gravity = (0, mars_gravity)

# DQN Model
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

# Landing Quality Check
def is_safe_landing(state):
    x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = state
    legs_down = left_leg > 0.5 and right_leg > 0.5
    small_angle = abs(angle) < 0.1
    soft_velocity = abs(x_vel) < 0.5 and abs(y_vel) < 0.5
    return legs_down and small_angle and soft_velocity

# Loading Environment and Model
env = gym.make("LunarLander-v3", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = DQN(state_dim, action_dim)

# Load from saved checkpoint
checkpoint = torch.load("checkpoint.pth")
q_net.load_state_dict(checkpoint["model_state_dict"])
q_net.eval()

# Visualization Configuration
EPISODES_TO_VISUALIZE = 5
total_rewards = []

for ep in range(EPISODES_TO_VISUALIZE):
    state, _ = env.reset(seed=None)
    done = False
    total_reward = 0

    print(f"\n--- Episode {ep + 1} ---")

    while not done:
        with torch.no_grad():
            action = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

    if not is_safe_landing(state):
        print("Rough or unsafe landing detected. Penalizing reward.")
        total_reward -= 100

    total_rewards.append(total_reward)
    print(f"Episode {ep + 1} finished. Adjusted total reward: {total_reward:.2f}")
    time.sleep(1.0)

# Final Summarization
average_reward = np.mean(total_rewards)
print(f"\n Average adjusted reward over {EPISODES_TO_VISUALIZE} episodes: {average_reward:.2f}")

env.close()
