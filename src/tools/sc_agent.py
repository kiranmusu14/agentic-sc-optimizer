import os
import torch
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from src.env.network_env import SelfHealingSupplyChain
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

# Load the API key from the .env file securely
import os
from dotenv import load_dotenv

# Load the key from your .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 1. Load your trained "Brain" and Environment
model = PPO.load("./data/models/sc_optimizer_final")
env = SelfHealingSupplyChain(num_nodes=50)

# 2. Define the Tools for the Agent
def check_network_risk(query):
    """Calculates real-time risk across the 50-node network."""
    obs, _ = env.reset()
    inventory = obs[:50]
    low_stock_nodes = np.where(inventory < 100)[0]
    high_stock_nodes = np.where(inventory > 500)[0]
    return f"Risk Analysis: Found {len(low_stock_nodes)} nodes with critical stockout risk. " \
           f"Identified {len(high_stock_nodes)} nodes with surplus 'trapped inventory' for rebalancing."

def run_redistribution_logic(query):
    """Predicts the optimal movement paths using the DRL model."""
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    total_moved = np.sum(np.abs(action))
    return f"The Self-Healing Optimizer recommends redistributing {total_moved:.2f} units " \
           f"across the network to maintain the 96% SLA."

# 3. Initialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [
    Tool(
        name="NetworkRiskChecker",
        func=check_network_risk,
        description="Analyzes the 50-node network for stockout risks and trapped inventory."
    ),
    Tool(
        name="OptimizationPredictor",
        func=run_redistribution_logic,
        description="Predicts the best redistribution moves to minimize fulfillment costs."
    )
]

# 4. Create the Reasoning Agent
agent = initialize_agent(
    tools, 
    llm, 
    agent="zero-shot-react-description", 
    verbose=True
)

if __name__ == "__main__":
    print("🤖 Agentic Supply Chain Optimizer is Online.")
    agent.run("Provide a risk summary of the network and suggest a redistribution strategy.")