import os
import numpy as np
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from supply_chain_env import SupplyChainEnv
from stable_baselines3 import PPO

# 1. Setup Vertex AI 
# It will use your authenticated 'gcloud' credentials automatically
llm = ChatVertexAI(
    model_name="gemini-2.0-flash",
    project="mlops-krutika-feb2024",
    location="us-central1",
    temperature=0
)

# 2. Define the Functions
def get_inventory_status():
    """Checks current safety stock levels across the network."""
    env = SupplyChainEnv()
    state, _ = env.reset()
    low_stock_count = np.sum(state[:, 0] < 50)
    total_inv = np.sum(state[:, 0])
    return f"Inventory Report: {low_stock_count} nodes are below safety stock. Total units: {total_inv:.0f}."

def simulate_stress_test(scenario: str):
    """Simulates impact of external disruptions."""
    if "strike" in scenario.lower():
        return "CRITICAL: Port strike will increase lead times by 5 days and spike shipping costs."
    return "Scenario analyzed: Minor impact expected."

def apply_ppo_optimization(scenario_name: str):
    """Loads the trained PPO model to rebalance inventory for a specific scenario."""
    # 1. Load the Environment and the Model
    env = SupplyChainEnv()
    model = PPO.load("supply_chain_ppo_model")
    
    # 2. Simulate the 'Port Strike' effect (e.g., reduce incoming stock)
    obs, _ = env.reset()
    total_optimized_reward = 0
    
    # Run a mini-simulation for 7 days to see how the AI handles the strike
    for _ in range(7):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_optimized_reward += reward
        
    return f"Optimization Complete for {scenario_name}. The PPO model rebalanced the 50 nodes. Expected 7-day reward: {total_optimized_reward:.2f}"

# 3. Bind the Tools
# This is the modern replacement for the old 'initialize_agent'
tools = [get_inventory_status, simulate_stress_test, apply_ppo_optimization]
llm_with_tools = llm.bind_tools(tools)

# 4. Run the Agentic Query
print("--- 🤖 Agentic Optimizer (Vertex AI) Online ---")

query = "Check the inventory health. If there is a risk of a port strike, run the PPO optimization to find the best rebalancing strategy and tell me the predicted reward."
messages = [HumanMessage(content=query)]

# Gemini identifies which tools to use based on your query
ai_msg = llm_with_tools.invoke(messages)

# Execute the identified tools
for tool_call in ai_msg.tool_calls:
    if tool_call["name"] == "get_inventory_status":
        print(f"🛠️ Tool Result: {get_inventory_status()}")
    if tool_call["name"] == "simulate_stress_test":
        print(f"🛠️ Tool Result: {simulate_stress_test('port strike')}")

print(f"\n🧠 Gemini's Final Reasoning: {ai_msg.content}")