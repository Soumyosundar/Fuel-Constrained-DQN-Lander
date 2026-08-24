"""
Lunar Lander Environment Configuration
Fuel-Constrained LunarLander-v2 Environment
"""

import gymnasium as gym
import numpy as np

# Environment Configuration (from paper Table 1)
class LanderConfig:
    # Environment Settings
    STATE_DIM = 8
    ACTION_DIM = 4

    # Fuel Constraints
    MAX_FUEL = 100
    FUEL_PER_STEP = 1  # Main engine uses 1 fuel per step

    # Reward Shaping
    CRASH_PENALTY = -100
    LANDING_BONUS = 100

    # DQN Hyperparameters (Table 1)
    LEARNING_RATE = 1e-3
    DISCOUNT_FACTOR = 0.99
    REPLAY_BUFFER_SIZE = 10000
    BATCH_SIZE = 128
    TARGET_UPDATE_INTERVAL = 10

    # Exploration Strategy
    EPSILON_INITIAL = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    EPSILON_EVAL = 0.05  # Locked during evaluation

    # Neural Network Architecture
    HIDDEN_LAYER_1 = 128
    HIDDEN_LAYER_2 = 128

    @classmethod
    def get_summary(cls):
        return {
            'state_dim': cls.STATE_DIM,
            'action_dim': cls.ACTION_DIM,
            'max_fuel': cls.MAX_FUEL,
            'learning_rate': cls.LEARNING_RATE,
            'discount_factor': cls.DISCOUNT_FACTOR,
            'replay_buffer': cls.REPLAY_BUFFER_SIZE,
            'batch_size': cls.BATCH_SIZE,
            'epsilon_decay': cls.EPSILON_DECAY,
            'network': f"{cls.STATE_DIM}-{cls.HIDDEN_LAYER_1}-{cls.HIDDEN_LAYER_2}-{cls.ACTION_DIM}"
        }

def create_environment(render=False):
    """Create LunarLander-v2 environment with fuel constraints"""
    if render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2")

    return env

def reset_environment(env, seed=None):
    """Reset environment and return initial state"""
    state, info = env.reset(seed=seed)
    return state

def step_environment(env, action, fuel_left):
    """
    Step environment with fuel constraint

    Args:
        env: Gymnasium environment
        action: Action to take (0-3)
        fuel_left: Current fuel remaining

    Returns:
        next_state, reward, done, fuel_left, info
    """
    # Check fuel constraint
    if fuel_left <= 0 and action == 2:  # Can't use main engine if out of fuel
        action = 0

    # Track fuel usage
    if action == 2:  # Main engine
        fuel_left -= LanderConfig.FUEL_PER_STEP

    # Step environment
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    return next_state, reward, done, fuel_left

def get_action_meaning(action):
    """Get human-readable action description"""
    meanings = {
        0: "Do Nothing",
        1: "Fire Left Engine",
        2: "Fire Main Engine",
        3: "Fire Right Engine"
    }
    return meanings.get(action, "Unknown")

def print_config():
    """Print environment configuration"""
    print("\n" + "="*70)
    print("LUNAR LANDER ENVIRONMENT CONFIGURATION")
    print("="*70)
    config = LanderConfig.get_summary()
    for key, value in config.items():
        print(f"  {key:<25}: {value}")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_config()

    # Test environment creation
    env = create_environment(render=False)
    state = reset_environment(env, seed=42)

    print(f"✓ Environment created successfully")
    print(f"✓ State shape: {state.shape}")
    print(f"✓ State dimension matches config: {state.shape[0] == LanderConfig.STATE_DIM}")
    print(f"✓ Action space: {LanderConfig.ACTION_DIM} discrete actions")

    env.close()
