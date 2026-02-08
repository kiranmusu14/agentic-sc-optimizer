import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from stable_baselines3 import PPO
from src.env.network_env import SelfHealingSupplyChain

def run_visualization():
    # 1. Load Environment and Trained Model
    env = SelfHealingSupplyChain(num_nodes=50)
    model = PPO.load("./data/models/sc_optimizer_final")
    
    # 2. Get a sample state
    obs, _ = env.reset()
    inventory = obs[:50]
    
    # 3. Predict the 'Self-Healing' redistribution action
    action, _ = model.predict(obs)
    
    # 4. Setup the Graph Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(env.graph, seed=42) # Consistent layout
    
    # 5. Color nodes based on inventory level (Health Map)
    # Lower inventory = Redder, Higher = Greener
    node_colors = []
    for inv in inventory:
        if inv < 100:
            node_colors.append('red')    # Critical Risk
        elif inv < 300:
            node_colors.append('orange') # Warning
        else:
            node_colors.append('green')  # Healthy
            
    # 6. Draw the Network
    nx.draw(env.graph, pos, node_color=node_colors, with_labels=True, 
            node_size=500, font_size=8, edge_color='gray', alpha=0.7)
    
    plt.title("Supply Chain Risk Heatmap: 50-Node Retail Network")
    plt.annotate("Green: Healthy | Orange: Warning | Red: Stockout Risk", 
                 xy=(0.05, 0.05), xycoords='axes fraction')
    
    # Save the visualization
    plt.savefig("data/network_risk_map.png")
    print("✅ Visualization saved to data/network_risk_map.png")
    plt.show()

if __name__ == "__main__":
    run_visualization()