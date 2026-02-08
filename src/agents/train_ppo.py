import os
from stable_baselines3 import PPO
from src.env.network_env import SelfHealingSupplyChain

def train():
    env = SelfHealingSupplyChain(num_nodes=50)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./data/logs/")
    print("🚀 Starting DRL Training...")
    model.learn(total_timesteps=50000)
    os.makedirs("./data/models/", exist_ok=True)
    model.save("./data/models/sc_optimizer_final")
    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    train()
