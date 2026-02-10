from stable_baselines3 import PPO
import numpy as np
# This line tells Python: "Go to supply_chain_env.py and grab the class"
from supply_chain_env import SupplyChainEnv 

# 1. Load the Environment and the Trained Model
env = SupplyChainEnv()
model = PPO.load("supply_chain_ppo_model")

# 1. Load the Environment and the Trained Model
env = SupplyChainEnv()
model = PPO.load("supply_chain_ppo_model")

# 2. Run a test for 30 "days"
obs, info = env.reset()
total_reward = 0

print("\n--- 🏁 Starting Evaluation (30 Days) ---")

for day in range(30):
    # The AI looks at the state and predicts the best action
    action, _states = model.predict(obs, deterministic=True)
    
    # Apply the action to the world
    obs, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    print(f"Day {day+1}: Reward = {reward:.2f}")

print("---------------------------------------")
print(f"📊 Total Reward over 30 days: {total_reward:.2f}")
print(f"📈 Average Daily Reward: {total_reward/30:.2f}")