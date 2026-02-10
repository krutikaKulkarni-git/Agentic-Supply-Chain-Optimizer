import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from stable_baselines3 import PPO



class SupplyChainEnv(gym.Env):
    def __init__(self):
        super(SupplyChainEnv, self).__init__()
        
        # 1. SETUP DATA PATH
        file_path = Path(__file__).parent / "Supply chain logistics problem.xlsx"
        
        # 2. LOAD KAGGLE DATA
        # Note: 'openpyxl' must be installed: pip install openpyxl
        self.wh_capacities = pd.read_excel(file_path, sheet_name="WhCapacities")
        self.freight_rates = pd.read_excel(file_path, sheet_name="FreightRates")
        self.order_list = pd.read_excel(file_path, sheet_name="OrderList")
        
        print("✅ Kaggle data loaded successfully!")
        sys.stdout.flush()

        # 3. DEFINE SPACES
        # Observation: 50 nodes, each with [Inventory, LeadTime, Demand]
        # We use the actual max capacity from your Excel file
        max_cap = self.wh_capacities['Daily Capacity '].max() 
        self.observation_space = spaces.Box(low=0, high=max_cap, shape=(50, 3), dtype=np.float32)
        
        # Action: Move amount (-50 to 50) for each node
        self.action_space = spaces.Box(low=-50, high=50, shape=(50,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0  # Initialize step counter
        
        inventory = np.random.uniform(50, 200, size=(50, 1))
        lead_time = np.random.uniform(1, 5, size=(50, 1))
        demand = np.random.uniform(5, 20, size=(50, 1))
        
        self.state = np.hstack((inventory, lead_time, demand)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        
        current_inventory = self.state[:, 0]
        current_demand = self.state[:, 2]
        
        # Physics: Inventory Update
        new_inventory = np.clip(current_inventory + action - current_demand, 0, 1000)
        stockouts = np.maximum(current_demand - (current_inventory + action), 0)
        
        # Calculate reward
        reward = self._calculate_reward(action, stockouts, new_inventory)
        
        # Update internal state
        self.state[:, 0] = new_inventory
        
        # Truncation logic
        truncated = self.current_step >= 30 
        terminated = False
        
        # Return Normalized State for the AI (Crucial for PPO stability)
        norm_state = self.state.copy()
        norm_state[:, 0] /= 1000  # Scale inventory by capacity
        norm_state[:, 2] /= 50    # Scale demand by max expected
        
        return norm_state, float(reward), terminated, truncated, {}

    def _calculate_reward(self, action, stockouts, inventory):
        # Balanced Reward Shaping
        ship_cost = np.sum(np.abs(action)) * 1.0
        stockout_penalty = np.sum(stockouts) * 15.0
        hold_cost = np.sum(inventory) * 2.0
    
        # Added "Sweet Spot" Bonus: +2.0 for maintaining safety stock (between 20-100)
        safety_bonus = np.sum((inventory >= 20) & (inventory <= 100)) * 2.0
        
        return (safety_bonus - (ship_cost + stockout_penalty + hold_cost))



#TESTING

env = SupplyChainEnv()
obs, info = env.reset()
print(f"Initial State Shape: {obs.shape}") # Should be (50, 3)
sys.stdout.flush()

# Take one random action
random_action = env.action_space.sample()
obs, reward, term, trunc, info = env.step(random_action)
print(f"Reward after 1 step: {reward}")
sys.stdout.flush()



#PPO

# 1. Create the Environment
env = SupplyChainEnv()

# 2. Create the AI "Brain" (PPO)
# 'MlpPolicy' means it's a standard neural network
model = PPO("MlpPolicy", env, verbose=1)

# 3. Start Training (The AI plays the game for 10,000 steps)
print("🚀 Training the AI... this might take a minute.")
model.learn(total_timesteps=100000)

# 4. Save the Brain
model.save("supply_chain_ppo_model")
print("✅ Training Complete! Model saved as 'supply_chain_ppo_model'")


