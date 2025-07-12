import pandas as pd
import matplotlib.pyplot as plt
import glob

# Find all CSV files with training data
csv_files = glob.glob("BipedalWalker-v3_*sac-data.csv")

plt.figure(figsize=(12, 6))

for csv_file in csv_files:
    # Load data
    df = pd.read_csv(csv_file, header=None, names=['Episode', 'Reward', 'Running_Avg'])
    
    # Plot running average
    plt.plot(df['Episode'], df['Running_Avg'], label=csv_file.split('-data')[0])

plt.title('SAC Training Progress on BipedalWalker-v3')
plt.xlabel('Episodes')
plt.ylabel('Running Average Reward')
plt.legend()
plt.grid(True)
plt.savefig('training_progress.png')
plt.show()