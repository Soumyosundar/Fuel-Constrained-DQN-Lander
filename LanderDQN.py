import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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

# Memory Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Training Function
def train(q_net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = q_net(states).gather(1, actions)

    next_actions = q_net(next_states).argmax(1, keepdim=True)
    max_next_q_values = target_net(next_states).gather(1, next_actions).detach()

    expected_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Training Configuration
EPISODES = 1
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE = 10
MEMORY_SIZE = 50000
MAX_FUEL = 100
SUCCESS_THRESHOLD = 200

env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

# Tracking Metrics
rewards = []
losses = []
accuracies = []
successes = 0

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.001

# Training Loop
for episode in range(EPISODES):
    state, _ = env.reset(seed=episode)
    total_reward = 0
    done = False
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    fuel_left = MAX_FUEL
    total_steps = 0
    greedy_actions = 0
    episode_losses = []

    while not done:
        total_steps += 1
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = q_values.argmax().item()
                greedy_actions += 1

        if fuel_left <= 0 and action == 2:
            action = 0
        if action == 2:
            fuel_left -= 1

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            x_pos = env.unwrapped.lander.position.x
            pad_left = env.unwrapped.helipad_x1
            pad_right = env.unwrapped.helipad_x2
            on_ground = next_state[6] == 1.0
            if on_ground and not (pad_left <= x_pos <= pad_right):
                reward -= 100
                total_reward += -100

        if abs(state[0]) < 0.1:
            reward += 0.1
        if abs(state[2]) < 0.1 and abs(state[3]) < 0.1:
            reward += 0.1

        memory.push((state, action, reward, next_state, done))
        loss = train(q_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA)
        if loss is not None:
            episode_losses.append(loss)

        state = next_state
        total_reward += reward

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())

    accuracy = (greedy_actions / total_steps) * 100 if total_steps > 0 else 0
    avg_loss = np.mean(episode_losses) if episode_losses else 0

    # Success Conditions
    if done:
        left_leg, right_leg = state[6], state[7]
        landed_softly = abs(state[2]) < 0.5 and abs(state[3]) < 0.5
        landed_upright = abs(state[4]) < 0.1
        if left_leg > 0.5 and right_leg > 0.5 and landed_softly and landed_upright:
            successes += 1

    success_rate = (successes / (episode + 1)) * 100

    rewards.append(total_reward)
    losses.append(avg_loss)
    accuracies.append(accuracy)

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Fuel Left: {fuel_left}, "
          f"Greedy Action Rate: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}, Success Rate: {success_rate:.2f}%")

    torch.save({
        'episode': episode,
        'model_state_dict': q_net.state_dict(),
        'target_model_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "checkpoint.pth")

# Saving the model
torch.save(q_net.state_dict(), "lunarlander_dqn.pth")
torch.save(target_net.state_dict(), "target_net.pth")
print("Model saved.")

env.close()

# Results displayed on graph
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(rewards)
plt.title("Total Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1, 3, 2)
plt.plot(losses)
plt.title("Average Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")

plt.subplot(1, 3, 3)
plt.plot(accuracies)
plt.title("Greedy Action Accuracy")
plt.xlabel("Episode")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()
