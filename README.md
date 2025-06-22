# Fuel Constrained DQN Agent on LunarLander-v2

This repository contains a Deep Q-Network (DQN) agent trained on the LunarLander-v2 environment from [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/), with a custom reward shaping mechanism that penalizes excessive fuel usage. The goal is to promote efficient landings while maintaining high success rates.

## Overview

Traditional reinforcement learning agents trained on LunarLander often prioritize landing success at the cost of realism, expending large amounts of fuel. This project modifies the reward function to:

- Penalize excess main engine and side engine usage.
- Encourage fuel efficient control strategies.
- Maintain a balance between landing success and fuel conservation.

## Key Components (LanderDQN.py)

- **Custom DQN Agent**: Fully implemented from scratch using PyTorch. (L
- **Modified Reward Function**: Adds negative rewards based on engine thrust.
- **Training & Evaluation Scripts**: Easily train and test performance across episodes.
- **Performance Visualization (Lander.py)**: Includes plots comparing baseline and fuel-constrained agents.
 
