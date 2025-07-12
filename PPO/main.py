# main.py
import csv
import gym
import torch as T
from agent import PPOAgent
from os import path
from time import time
import numpy as np
# Fix deprecated aliases that some Gym versions still reference
np.bool8 = np.bool_
np.bool_ = np.bool_
np.bool = bool

# Print PyTorch and CUDA info
print(f"PyTorch version: {T.__version__}")
print(f"CUDA available: {T.cuda.is_available()}")
if T.cuda.is_available():
    print(f"CUDA version: {T.version.cuda}")
    print(f"GPU device count: {T.cuda.device_count()}")
    print(f"Current GPU: {T.cuda.current_device()}")
    print(f"GPU name: {T.cuda.get_device_name()}")

# OPTIMIZED HYPERPARAMETERS
gamma = 0.99  # Keep discount factor - this is standard
gae_lambda = 0.95  # Keep GAE lambda - this is standard
learningRate = 3e-4  # Keep learning rate - this is generally optimal for PPO
clip_ratio = 0.2  # Keep clip ratio - this is standard for PPO
value_coef = 0.5  # Keep value loss coefficient
entropy_coef = 0.01  # Keep entropy coefficient (could increase to 0.02 for more exploration)
resume = True  # Keep resuming from checkpoint
render = True  # CHANGE: Turn off rendering during training to increase speed
ppo_epochs = 10  # CHANGE: Reduce from 100 to 10 (still effective but much faster)
steps_per_epoch = 4096  # CHANGE: Increase to collect more data per update
miniBatchSize = 512  # CHANGE: Increase for better GPU utilization
max_episodes = 5500  # Keep maximum episodes
envName = "BipedalWalker-v3"

for trial in range(1):  # Reduced to 1 trial for testing
    renderMode = "human" if render else None
    env = gym.make(envName, render_mode=renderMode)
    env.name = envName + "_" + str(trial)
    csvName = env.name + "ppo-data.csv"
    
    agent = PPOAgent(
        env, 
        learningRate, 
        gamma, 
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        shouldLoad=resume,
        ppo_epochs=ppo_epochs,
        value_coef=value_coef,
        entropy_coef=entropy_coef
    )
    
    print(f"Using device: {agent.device}")
    
    state, info = env.reset()
    episode = 0
    steps_since_update = 0
    runningReward = None

    # determine the last episode if we have saved training in progress
    if resume and path.exists(csvName):
        fileData = list(csv.reader(open(csvName)))
        lastLine = fileData[-1]
        episode = int(lastLine[0])

    start_time = time()
    total_steps = 0
    
    while episode <= max_episodes:
        try:
            # Get value prediction for current state
            value = agent.value_estimate(state)
            
            # Choose action from policy (stochastic during training)
            action, logProb = agent.getAction(state)
            
            # Take step in environment
            nextState, reward, terminated, truncated, info = env.step(action)
            
            # Store transition in PPO buffer
            agent.buffer.store(state, action, reward, value, logProb, terminated)
            
            # Update state
            state = nextState
            steps_since_update += 1
            total_steps += 1
            
            # If trajectory ended
            if terminated or truncated:
                # Get value of final state (0 if terminated)
                last_value = 0
                if not terminated:
                    # Bootstrap value for truncated episodes
                    last_value = agent.value_estimate(state)
                
                # Compute advantages for the finished trajectory
                agent.buffer.finish_path(last_value)
                
                # Reset environment
                state, info = env.reset()
                episode += 1
                
                # Only evaluate and log periodically
                if episode % 10 == 0:
                    # Evaluate the deterministic agent
                    sumRewards = 0.0
                    eval_state, info = env.reset()
                    eval_terminated = eval_truncated = False
                    
                    while not eval_terminated and not eval_truncated:
                        # Use deterministic actions for evaluation
                        eval_action = agent.getAction(eval_state, evaluate=True)
                        eval_nextState, eval_reward, eval_terminated, eval_truncated, info = env.step(eval_action)
                        
                        if render:
                            env.render()
                            
                        eval_state = eval_nextState
                        sumRewards += eval_reward
                    
                    # Keep running average
                    runningReward = (
                        sumRewards
                        if runningReward is None
                        else runningReward * 0.99 + sumRewards * 0.01
                    )
                    
                    # Log progress
                    fields = [episode, sumRewards, runningReward]
                    with open(env.name + "ppo-data.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(fields)
                    
                    # Print episode info with GPU memory info if using GPU
                    elapsed_time = time() - start_time
                    gpu_info = ""
                    if agent.device.type == 'cuda':
                        gpu_info = f" --- GPU memory: {T.cuda.memory_allocated() / 1e9:.2f}GB"
                    
                    print(
                        f"Episode {episode:6d} (Steps: {total_steps}) --- "
                        f"Time: {elapsed_time:.2f}s --- "
                        f"Reward: {sumRewards:.2f} --- "
                        f"Running Avg: {runningReward:.2f}{gpu_info}"
                    )
                    start_time = time()
                    
                    # Save model
                    agent.save()
            
            # Perform PPO update when enough steps are collected
            if steps_since_update >= steps_per_epoch:
                print(f"Updating policy after {steps_since_update} steps...")
                agent.update()
                steps_since_update = 0
                
        except Exception as e:
            print(f"Error during training: {e}")
            # Try to continue from the next episode
            state, info = env.reset()
            continue

# At the end of training, if using GPU, print final memory usage
if agent.device.type == 'cuda':
    print(f"Final GPU memory allocated: {T.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Final GPU memory reserved: {T.cuda.memory_reserved() / 1e9:.2f} GB")
    # Optionally free memory
    T.cuda.empty_cache()