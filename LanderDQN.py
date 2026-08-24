"""
Deep Q-Network (DQN) Implementation for Lunar Lander
Fuel-Constrained LunarLander-v2 Environment
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
from LanderENV import LanderConfig, create_environment, reset_environment, step_environment

# ============================================================================
# DQN Neural Network Model
# ============================================================================

class DQN(nn.Module):
    """Deep Q-Network with 2 hidden layers of 128 neurons each"""

    def __init__(self, state_dim=LanderConfig.STATE_DIM, action_dim=LanderConfig.ACTION_DIM):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, LanderConfig.HIDDEN_LAYER_1),
            nn.ReLU(),
            nn.Linear(LanderConfig.HIDDEN_LAYER_1, LanderConfig.HIDDEN_LAYER_2),
            nn.ReLU(),
            nn.Linear(LanderConfig.HIDDEN_LAYER_2, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# Experience Replay Buffer
# ============================================================================

class ReplayMemory:
    """Experience replay buffer for storing and sampling transitions"""

    def __init__(self, capacity=LanderConfig.REPLAY_BUFFER_SIZE):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        """Store a transition (state, action, reward, next_state, done)"""
        self.memory.append(transition)

    def sample(self, batch_size):
        """Sample random batch from memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ============================================================================
# Training Function
# ============================================================================

def train_step(q_net, target_net, memory, optimizer, batch_size, gamma):
    """
    Perform a single training step (gradient descent on batch)

    Args:
        q_net: Policy network
        target_net: Target network
        memory: Replay buffer
        optimizer: Adam optimizer
        batch_size: Batch size for gradient descent
        gamma: Discount factor

    Returns:
        Loss value (or None if batch not ready)
    """
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    # Convert to tensors
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Compute Q-values for current states
    q_values = q_net(states).gather(1, actions)

    # Compute target Q-values using target network
    next_actions = q_net(next_states).argmax(1, keepdim=True)
    max_next_q_values = target_net(next_states).gather(1, next_actions).detach()
    expected_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute MSE loss
    loss = nn.MSELoss()(q_values, expected_q_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# ============================================================================
# Main Training Loop
# ============================================================================

def train_dqn(num_episodes, trial_name, device="cpu"):
    """
    Train DQN agent on Lunar Lander environment

    Args:
        num_episodes: Number of training episodes
        trial_name: Name of this trial (for checkpoint naming)
        device: 'cpu' or 'cuda'

    Returns:
        Trained model, target network, rewards history, losses history, checkpoint path
    """

    # Load configuration
    config = LanderConfig()

    # Create environment
    env = create_environment(render=False)

    # Initialize networks
    q_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # Initialize optimizer and memory
    optimizer = optim.Adam(q_net.parameters(), lr=config.LEARNING_RATE)
    memory = ReplayMemory(capacity=config.REPLAY_BUFFER_SIZE)

    # Tracking metrics
    rewards_history = []
    losses_history = []

    print(f"\n{'='*70}")
    print(f"TRAINING: {trial_name}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*70}\n")

    # Epsilon-greedy parameters
    epsilon = config.EPSILON_INITIAL

    # Training loop
    for episode in range(num_episodes):
        state = reset_environment(env, seed=episode)
        total_reward = 0
        done = False
        fuel_left = config.MAX_FUEL
        episode_losses = []

        # Epsilon decay (exponential)
        epsilon = config.EPSILON_MIN + (config.EPSILON_INITIAL - config.EPSILON_MIN) * \
                  np.exp(-config.EPSILON_DECAY * episode)

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                    action = q_values.argmax().item()

            # Step environment with fuel constraint
            next_state, reward, done, fuel_left = step_environment(env, action, fuel_left)

            # Store experience in replay buffer
            memory.push((state, action, reward, next_state, done))

            # Train on batch
            loss = train_step(q_net, target_net, memory, optimizer,
                            config.BATCH_SIZE, config.DISCOUNT_FACTOR)
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # Update target network periodically
        if episode % config.TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Store metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        rewards_history.append(total_reward)
        losses_history.append(avg_loss)

        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_history[-1000:])
            avg_loss_1k = np.mean(losses_history[-1000:])
            print(f"Episode {episode + 1:6d}/{num_episodes} | "
                  f"Avg Reward (last 1k): {avg_reward:8.2f} | "
                  f"Avg Loss (last 1k): {avg_loss_1k:8.4f} | "
                  f"Epsilon: {epsilon:.4f}")

    # Save checkpoint
    checkpoint_path = f"checkpoint_{trial_name}.pth"
    torch.save({
        'model_state_dict': q_net.state_dict(),
        'target_model_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon
    }, checkpoint_path)

    print(f"\n✓ Training complete. Model saved to {checkpoint_path}\n")

    env.close()

    return q_net, target_net, rewards_history, losses_history, checkpoint_path

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_dqn(checkpoint_path, num_eval_episodes, trial_name, device="cpu"):
    """
    Evaluate a trained DQN model (no learning, epsilon locked at 0.05)

    Args:
        checkpoint_path: Path to trained model checkpoint
        num_eval_episodes: Number of evaluation episodes
        trial_name: Name of this trial
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary with evaluation metrics
    """

    config = LanderConfig()

    # Load environment
    env = create_environment(render=False)

    # Load model
    q_net = DQN().to(device)
    target_net = DQN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    q_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['target_model_state_dict'])
    q_net.eval()
    target_net.eval()

    # Tracking metrics
    rewards = []
    losses = []
    greedy_accuracies = []

    print(f"\n{'='*70}")
    print(f"EVALUATION: {trial_name}")
    print(f"Episodes: {num_eval_episodes}")
    print(f"Epsilon (locked): {config.EPSILON_EVAL}")
    print(f"Learning: DISABLED")
    print(f"{'='*70}\n")

    for episode in range(num_eval_episodes):
        state = reset_environment(env, seed=1000 + episode)
        total_reward = 0
        done = False
        fuel_left = config.MAX_FUEL
        total_steps = 0
        greedy_actions = 0
        episode_losses = []

        while not done:
            total_steps += 1

            # Epsilon-greedy with FIXED epsilon (0.05)
            if np.random.random() < config.EPSILON_EVAL:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                    action = q_values.argmax().item()
                greedy_actions += 1

            # Step environment with fuel constraint
            next_state, reward, done, fuel_left = step_environment(env, action, fuel_left)

            # Compute prediction loss (without updating weights)
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action_tensor = torch.tensor([action], dtype=torch.long).unsqueeze(0).to(device)
                reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                done_tensor = torch.tensor([done], dtype=torch.float32).unsqueeze(0).to(device)

                q_pred = q_net(state_tensor).gather(1, action_tensor)
                next_actions = q_net(next_state_tensor).argmax(1, keepdim=True)
                q_target = target_net(next_state_tensor).gather(1, next_actions).detach()
                q_target = reward_tensor + config.DISCOUNT_FACTOR * q_target * (1 - done_tensor)

                loss = nn.MSELoss()(q_pred, q_target).item()
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # Calculate episode metrics
        greedy_accuracy = (greedy_actions / total_steps) * 100 if total_steps > 0 else 0
        avg_loss = np.mean(episode_losses) if episode_losses else 0

        rewards.append(total_reward)
        losses.append(avg_loss)
        greedy_accuracies.append(greedy_accuracy)

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_accuracy = np.mean(greedy_accuracies[-100:])
            print(f"Episode {episode + 1:4d}/{num_eval_episodes} | "
                  f"Avg Reward (last 100): {avg_reward:8.2f} | "
                  f"Avg Accuracy (last 100): {avg_accuracy:6.2f}%")

    # Compute final statistics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    avg_accuracy = np.mean(greedy_accuracies)
    std_accuracy = np.std(greedy_accuracies)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS: {trial_name}")
    print(f"{'='*70}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Reward Range: [{min_reward:.2f}, {max_reward:.2f}]")
    print(f"Average Loss (MSE): {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"Average Greedy Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"{'='*70}\n")

    env.close()

    return {
        'rewards': rewards,
        'losses': losses,
        'greedy_accuracies': greedy_accuracies,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'avg_loss': avg_loss,
        'std_loss': std_loss,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy
    }

# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python LanderDQN.py <num_episodes> <trial_name> [eval_checkpoint] [eval_episodes]")
        print("\nExamples:")
        print("  Train only:  python LanderDQN.py 10000 Trial_1")
        print("  Eval only:   python LanderDQN.py 0 Trial_1 checkpoint_Trial_1.pth 1000")
        sys.exit(1)

    num_episodes = int(sys.argv[1])
    trial_name = sys.argv[2]
    eval_checkpoint = sys.argv[3] if len(sys.argv) > 3 else None
    eval_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

    if num_episodes > 0:
        train_dqn(num_episodes, trial_name)

    if eval_checkpoint:
        evaluate_dqn(eval_checkpoint, eval_episodes, trial_name)
