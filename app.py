import streamlit as st
import pandas as pd
import numpy as np
from supply_chain_env import SupplyChainEnv # Added missing import
from agent_logic import get_inventory_status, simulate_stress_test

# 1. SETUP (Must be first!)
st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")
st.title("📦 Agentic Supply Chain Optimizer")

# 2. KPI METRICS (Top Row)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Network Inventory", value="6,022 Units")
with col2:
    st.metric(label="System Health", value="Optimal", delta="Healthy")
with col3:
    st.metric(label="Model Status", value="PPO Ready", delta="Trained")

st.divider()

# 3. MAIN LAYOUT
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📈 Full Network Inventory (50 Nodes)")
    
    # Initialize env and get 50-node state
    env = SupplyChainEnv()
    state, _ = env.reset()
    inventory_data = state[:, 0]
    
    node_names = [f"Node {i+1}" for i in range(50)]
    
    # Let user select specific nodes to keep the chart clean
    selected_nodes = st.multiselect(
        "Select nodes to visualize:", 
        options=node_names, 
        default=["Node 1", "Node 5", "Node 10", "Node 25", "Node 50"]
    )
    
    # Create DataFrame and plot
    full_data = pd.DataFrame([inventory_data], columns=node_names)
    st.line_chart(full_data[selected_nodes])

    st.subheader("📋 Warehouse Capacity Details")
    df = pd.read_excel("Supply chain logistics problem.xlsx", sheet_name="WhCapacities")
    st.dataframe(df, use_container_width=True)

with right_col:
    st.subheader("🤖 Agent Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the supply chain..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            if "status" in prompt.lower():
                response = get_inventory_status()
            else:
                response = "I am monitoring the 50-node network. Ask me for an 'inventory status'!"
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})