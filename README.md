# 📦 Agentic Supply Chain Optimizer
**A Digital Twin & Autonomous Reasoning System for Global Logistics**

This project demonstrates a next-generation supply chain management tool. It integrates **Deep Reinforcement Learning (DRL)** to solve complex inventory rebalancing problems and **Generative AI (Gemini 2.0)** to provide executive-level reasoning during supply chain disruptions.

---

## 🎯 Executive Summary
Traditional supply chain tools report "what happened." This system tells you **"what will happen"** and **"how to fix it"**. By simulating a 50-node warehouse network using Kaggle logistics data, the system trains an AI "Brain" (PPO Model) to minimize costs while an LLM "Manager" (Gemini) handles strategic decision-making through natural language.



---

## 🛠️ Technical Architecture
The system is built on a three-tier architecture:

1. **Simulation Environment (The Body):** A custom Python environment built with `Gymnasium` that models inventory physics, lead times, and shipping costs based on real-world logistics datasets.
2. **Optimization Engine (The Brain):** A Proximal Policy Optimization (PPO) model that learns to rebalance stock across 50 nodes to maximize profit and minimize stockouts.
3. **Agentic Interface (The Voice):** Powered by **Google Vertex AI (Gemini 2.0 Flash)**, allowing managers to ask questions like *"How will a port strike affect our network?"* and receive both a risk analysis and a proposed AI solution.

---

## 📊 Key Performance Indicators (KPIs)
The dashboard tracks real-time metrics essential for supply chain visibility:
- **Total Network Inventory:** Global stock levels across all 50 warehouses.
- **Low-Stock Alerts:** Automated identification of nodes below safety stock levels.
- **Stress-Test Simulation:** Real-time "What-If" analysis for demand spikes or shipping delays.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.12+
- Google Cloud Project with **Vertex AI API** enabled.
- A linked Google Cloud Billing account (for Gemini API access).

### 2. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/Agentic-Supply-Chain-Optimizer.git](https://github.com/YOUR_USERNAME/Agentic-Supply-Chain-Optimizer.git)
cd Agentic-Supply-Chain-Optimizer
pip install -r requirements.txt


### 3. Authentication
Authenticate your local machine with Google Cloud to allow the agent to use Gemini:

Bash
gcloud auth application-default login

4. Running the Dashboard
Launch the interactive Streamlit interface:

Bash
streamlit run app.py

##📂 File Structure
app.py: Streamlit dashboard and UI layout.

agent_logic.py: Gemini tool-calling and LLM reasoning logic.

supply_chain_env.py: Custom RL environment modeling the 50-node network.

supply_chain_ppo_model.zip: The pre-trained Reinforcement Learning model.

Supply chain logistics problem.xlsx: Raw logistics data from Kaggle.


