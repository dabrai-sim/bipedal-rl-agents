# Human-Inspired Walking with Deep Reinforcement Learning 🦿🧠

A reinforcement learning research project that teaches an agent to walk like a human in the `BipedalWalker-v3` environment using advanced techniques like Behavior Cloning, Reward Shaping, and Deep RL algorithms (PPO, TD3, SAC).

## 📌 Problem Statement

Design and train a reinforcement learning agent that can achieve **stable, energy-efficient, and human-like walking patterns**. The core challenge involves teaching the agent to balance and move realistically across uneven terrain using expert data, reward engineering, and policy optimization.

---

## 🌍 Environment

- **Name**: `BipedalWalker-v3` (OpenAI Gym)
- **State Space**: 24-dimensional continuous vector
- **Action Space**: 4-dimensional continuous torque actions (hip and knee joints)
- **Objective**: Walk forward while balancing energy usage and mimicking expert motion

---

## 🧠 Methodology

### 🔄 MDP Formulation
- **States (S)**: 24D continuous vector
- **Actions (A)**: 4D continuous torque (range: [-1, 1])
- **Reward (R)**:  R = R(env) + 0.5 \* cosine\_similarity(s, s\_ref) - 0.01 \* Σ(a²)

where:
- `R(env)`: base reward from BipedalWalker
- `cosine_similarity`: match with expert motion
- `Σ(a²)`: energy penalty

### 🔧 Training Pipeline
- **Synthetic Expert Data Generation**: Generated trajectories saved as `.npy` files
- **Behavior Cloning (BC)**: Supervised pretraining from expert trajectories
- **Reward Shaping**: Includes motion similarity and energy constraints
- **Reinforcement Learning Models**:
- **PPO**: 5250 episodes
- **TD3**: 2008 episodes
- **SAC**: 1500 episodes

---

## 📊 Results

All models learned to walk with different advantages:

| Model | Strengths |
|-------|-----------|
| **PPO** | Simplicity & Stability |
| **TD3** | Precision & Control |
| **SAC** | Exploration & Energy Efficiency |

The **SAC agent** demonstrated the most stable and human-like walking behavior, with smoother gait and lower energy consumption.

---

## 📽️ Visualizations

Training curves and walking behaviors were recorded to visualize gait, stability, and learning progress across episodes.

---

## ✅ Conclusion

By combining imitation learning (Behavior Cloning) and deep RL techniques, we successfully trained agents that walk efficiently and naturally. This project demonstrates the power of hybrid learning methods in solving complex locomotion problems.

---

## 👥 Authors

- Simrann Dabrai – 22070126111  
- Tejas Thange – 22070126121  
- V Kavyasri – 22070126129

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- OpenAI Gym (with Box2D)
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
````

---

## 📌 Future Work

* Add Curriculum Learning for smoother skill acquisition
* Fine-tune reward weights using Bayesian optimization
* Deploy the agent into a 3D simulator like MuJoCo or IsaacGym

---

⭐ If you found this project interesting, give it a star!

