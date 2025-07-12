# eval.py
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import torch as T

# Print CUDA availability info
print(f"PyTorch version: {T.__version__}")
print(f"CUDA available: {T.cuda.is_available()}")
if T.cuda.is_available():
    print(f"CUDA device: {T.cuda.get_device_name(0)}")

# Find all CSV files with training data
csv_files = glob.glob("BipedalWalker-v3_*ppo-data.csv")

plt.figure(figsize=(12, 6))

for csv_file in csv_files:
    # Load data
    df = pd.read_csv(csv_file, header=None, names=['Episode', 'Reward', 'Running_Avg'])
    
    # Apply smoothing to the running average for better visualization
    window_size = 10
    df['Smoothed_Avg'] = df['Running_Avg'].rolling(window=window_size, min_periods=1).mean()
    
    # Plot running average
    plt.plot(df['Episode'], df['Smoothed_Avg'], label=csv_file.split('-data')[0])

    # Print statistics
    print(f"Model: {csv_file.split('-data')[0]}")
    print(f"Max reward: {df['Reward'].max():.2f}")
    print(f"Average of last 100 episodes: {df['Reward'].tail(100).mean():.2f}")
    print(f"Standard deviation of last 100 episodes: {df['Reward'].tail(100).std():.2f}")
    print("-----------------------------------")

plt.title('PPO Training Progress on BipedalWalker-v3')
plt.xlabel('Episodes')
plt.ylabel('Running Average Reward')
plt.axhline(y=300, color='r', linestyle='--', label='Solving threshold')
plt.legend()
plt.grid(True)
plt.savefig('ppo_training_progress.png', dpi=300)
plt.show()