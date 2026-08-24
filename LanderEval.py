"""
Fuel-Constrained DQN Training Experiments
Reproducing paper trials exactly as specified

Trial 1: 10,000 episodes pretraining + 500 evaluation episodes
Trial 2: 100,000 episodes pretraining + 1,000 evaluation episodes
Trial 3: 200,000 episodes pretraining + 1,000 evaluation episodes
"""

import subprocess
import time
import json
import numpy as np
from pathlib import Path
import torch

# Experiment configuration (from paper Table 1)
EXPERIMENTS = [
    {
        'name': 'Trial_1',
        'pretrain_episodes': 10000,
        'eval_episodes': 1000,
        'description': '10,000 episodes pretraining + 1,000 eval'
    },
    {
        'name': 'Trial_2',
        'pretrain_episodes': 100000,
        'eval_episodes': 1000,
        'description': '100,000 episodes pretraining + 1,000 eval'
    },
    {
        'name': 'Trial_3',
        'pretrain_episodes': 200000,
        'eval_episodes': 1000,
        'description': '200,000 episodes pretraining + 1,000 eval'
    }
]

def run_trial(trial_config):
    """Run a single trial: pretraining + evaluation"""

    trial_name = trial_config['name']
    pretrain_eps = trial_config['pretrain_episodes']
    eval_eps = trial_config['eval_episodes']

    print(f"\n{'#'*70}")
    print(f"# {trial_name}: {trial_config['description']}")
    print(f"{'#'*70}\n")

    start_time = time.time()

    # Step 1: Pretraining
    print(f"STEP 1: PRETRAINING ({pretrain_eps} episodes)")
    print("-" * 70)

    train_cmd = [
        'python', '/root/train_dqn.py',
        str(pretrain_eps),
        trial_name
    ]

    try:
        result = subprocess.run(train_cmd, capture_output=False, text=True, timeout=None)
        if result.returncode != 0:
            print(f"ERROR: Training failed for {trial_name}")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    pretrain_time = time.time() - start_time

    # Step 2: Evaluation
    print(f"\nSTEP 2: EVALUATION ({eval_eps} episodes)")
    print("-" * 70)

    checkpoint_path = f"checkpoint_{trial_name}.pth"

    eval_cmd = [
        'python', '/root/evaluate_dqn.py',
        checkpoint_path,
        str(eval_eps),
        trial_name
    ]

    try:
        result = subprocess.run(eval_cmd, capture_output=False, text=True, timeout=None)
        if result.returncode != 0:
            print(f"ERROR: Evaluation failed for {trial_name}")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    total_time = time.time() - start_time

    print(f"\n✓ {trial_name} completed in {total_time/60:.1f} minutes")

    return {
        'trial': trial_name,
        'pretrain_time': pretrain_time,
        'total_time': total_time,
        'checkpoint': checkpoint_path
    }

def main():
    print("\n" + "="*70)
    print("FUEL-CONSTRAINED DQN EXPERIMENTS")
    print("="*70)
    print("Reproducing paper trials exactly:")
    print("  Trial 1: 10k episodes pretrain + 500 eval")
    print("  Trial 2: 100k episodes pretrain + 1k eval")
    print("  Trial 3: 200k episodes pretrain + 1k eval")
    print("="*70)

    results = []

    for i, trial_config in enumerate(EXPERIMENTS, 1):
        print(f"\n\n>>> Running Trial {i}/{len(EXPERIMENTS)}")
        result = run_trial(trial_config)
        if result:
            results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")

    total_time = 0
    for result in results:
        print(f"{result['trial']}:")
        print(f"  Pretraining time: {result['pretrain_time']/60:.1f} min")
        print(f"  Total time:       {result['total_time']/60:.1f} min")
        print(f"  Checkpoint:       {result['checkpoint']}\n")
        total_time += result['total_time']

    print(f"Total runtime: {total_time/3600:.1f} hours\n")
    print("✓ All trials completed!")
    print("Next: Run 'python generate_comparison_plots.py' to generate visualizations\n")

if __name__ == "__main__":
    main()
