"""
Generate comparison plots for all three trials
Recreates the paper figures: Total Reward, Average Loss, and Greedy Action Accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from evaluate_dqn import DQN, evaluate_dqn
import torch
import gymnasium as gym

# Trial configurations
TRIALS = [
    {
        'name': 'Trial_1',
        'checkpoint': 'checkpoint_Trial_1.pth',
        'eval_episodes': 500,
        'label': 'Trial 1: 10k episodes'
    },
    {
        'name': 'Trial_2',
        'checkpoint': 'checkpoint_Trial_2.pth',
        'eval_episodes': 1000,
        'label': 'Trial 2: 100k episodes'
    },
    {
        'name': 'Trial_3',
        'checkpoint': 'checkpoint_Trial_3.pth',
        'eval_episodes': 1000,
        'label': 'Trial 3: 200k episodes'
    }
]

def load_or_evaluate_trial(trial_config):
    """Load evaluation results or run evaluation if not cached"""

    cache_file = f"eval_results_{trial_config['name']}.json"

    # Try to load from cache
    if Path(cache_file).exists():
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return {
                'rewards': np.array(data['rewards']),
                'losses': np.array(data['losses']),
                'greedy_accuracies': np.array(data['greedy_accuracies']),
                'avg_reward': data['avg_reward'],
                'avg_loss': data['avg_loss'],
                'avg_accuracy': data['avg_accuracy']
            }

    # Otherwise, run evaluation
    print(f"Running evaluation for {trial_config['name']}...")
    results = evaluate_dqn(
        trial_config['checkpoint'],
        trial_config['eval_episodes'],
        trial_config['name']
    )

    # Cache results
    cache_data = {
        'rewards': results['rewards'],
        'losses': results['losses'],
        'greedy_accuracies': results['greedy_accuracies'],
        'avg_reward': float(results['avg_reward']),
        'avg_loss': float(results['avg_loss']),
        'avg_accuracy': float(results['avg_accuracy'])
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

    return results

def generate_comparison_plots():
    """Generate side-by-side comparison plots for all three trials"""

    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70 + "\n")

    # Load results for all trials
    all_results = {}
    for trial in TRIALS:
        print(f"\n--- Processing {trial['name']} ---")
        all_results[trial['name']] = load_or_evaluate_trial(trial)

    # Create figure with 3x3 subplots (one row per trial, 3 metrics per trial)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('DQN Lunar Lander Performance Comparison: 3 Training Durations',
                 fontsize=16, fontweight='bold', y=0.995)

    # Define colors for each trial
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue

    # Generate plots for each trial
    for idx, trial in enumerate(TRIALS):
        trial_name = trial['name']
        results = all_results[trial_name]
        color = colors[idx]

        # Get evaluation episodes
        eval_eps = trial['eval_episodes']
        x_axis = np.arange(1, eval_eps + 1)

        print(f"\nGenerating plots for {trial_name}...")
        print(f"  Reward range: [{results['rewards'].min():.2f}, {results['rewards'].max():.2f}]")
        print(f"  Loss range: [{results['losses'].min():.4f}, {results['losses'].max():.4f}]")
        print(f"  Accuracy range: [{results['greedy_accuracies'].min():.2f}%, {results['greedy_accuracies'].max():.2f}%]")

        # Row: Total Reward
        ax = axes[idx, 0]
        ax.plot(x_axis, results['rewards'], color=color, linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=results['avg_reward'], color=color, linestyle='--',
                   linewidth=2, label=f"Mean: {results['avg_reward']:.2f}")
        ax.fill_between(x_axis, results['rewards'], alpha=0.2, color=color)
        ax.set_title(f'{trial["label"]}\nTotal Reward', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2)

        # Row: Average Loss (MSE)
        ax = axes[idx, 1]
        ax.plot(x_axis, results['losses'], color=color, linewidth=1.5, alpha=0.8)
        ax.axhline(y=results['avg_loss'], color=color, linestyle='--',
                   linewidth=2, label=f"Mean: {results['avg_loss']:.4f}")
        ax.fill_between(x_axis, results['losses'], alpha=0.2, color=color)
        ax.set_title(f'{trial["label"]}\nAverage Loss (MSE)', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2)

        # Row: Greedy Action Accuracy
        ax = axes[idx, 2]
        ax.plot(x_axis, results['greedy_accuracies'], color=color, linewidth=1.5, alpha=0.8)
        ax.axhline(y=results['avg_accuracy'], color=color, linestyle='--',
                   linewidth=2, label=f"Mean: {results['avg_accuracy']:.2f}%")
        ax.fill_between(x_axis, results['greedy_accuracies'], alpha=0.2, color=color)
        ax.set_title(f'{trial["label"]}\nGreedy Action Accuracy', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save figure
    output_path = 'comparison_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plots saved to {output_path}")

    # Create summary statistics figure
    create_summary_table(all_results, TRIALS)

    return fig, all_results

def create_summary_table(all_results, trials):
    """Create a summary table of key metrics"""

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    summary_data = []
    for trial in trials:
        trial_name = trial['name']
        results = all_results[trial_name]
        summary_data.append([
            trial['label'],
            f"{results['avg_reward']:.2f}",
            f"{results['rewards'].min():.2f}",
            f"{results['rewards'].max():.2f}",
            f"{results['avg_loss']:.4f}",
            f"{results['avg_accuracy']:.2f}%"
        ])

    # Print table
    print(f"\n{'Trial':<25} {'Avg Reward':<15} {'Min Reward':<15} {'Max Reward':<15} {'Avg Loss':<15} {'Avg Accuracy':<15}")
    print("-" * 100)
    for row in summary_data:
        print(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<15} {row[5]:<15}")

    # Create a separate summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Summary Comparison: Key Metrics Across All Trials',
                 fontsize=14, fontweight='bold')

    labels = [trial['label'].replace('Trial', 'T') for trial in trials]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    # Average Rewards
    avg_rewards = [all_results[trial['name']]['avg_reward'] for trial in trials]
    axes[0].bar(labels, avg_rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_title('Average Reward', fontweight='bold')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(avg_rewards):
        axes[0].text(i, v + 10, f'{v:.1f}', ha='center', fontweight='bold')

    # Average Loss
    avg_losses = [all_results[trial['name']]['avg_loss'] for trial in trials]
    axes[1].bar(labels, avg_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_title('Average Loss (MSE)', fontweight='bold')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(avg_losses):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

    # Average Accuracy
    avg_accuracies = [all_results[trial['name']]['avg_accuracy'] for trial in trials]
    axes[2].bar(labels, avg_accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].set_title('Average Greedy Accuracy', fontweight='bold')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_ylim([0, 100])
    axes[2].grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(avg_accuracies):
        axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    summary_path = 'summary_comparison.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Summary comparison saved to {summary_path}")

    # Save summary as JSON
    summary_json = {
        'trials': [
            {
                'name': trial['name'],
                'label': trial['label'],
                'avg_reward': float(all_results[trial['name']]['avg_reward']),
                'min_reward': float(all_results[trial['name']]['rewards'].min()),
                'max_reward': float(all_results[trial['name']]['rewards'].max()),
                'avg_loss': float(all_results[trial['name']]['avg_loss']),
                'avg_accuracy': float(all_results[trial['name']]['avg_accuracy'])
            }
            for trial in trials
        ]
    }

    with open('summary_stats.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"✓ Summary statistics saved to summary_stats.json")

    print("\n" + "="*70)

if __name__ == "__main__":
    fig, results = generate_comparison_plots()
    plt.show()
