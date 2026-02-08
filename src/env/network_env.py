import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

class SelfHealingSupplyChain(gym.Env):
    def __init__(self, num_nodes=50):
        super().__init__()
        self.num_nodes = num_nodes
        self.graph = nx.scale_free_graph(num_nodes).to_directed()
        self.observation_space = spaces.Box(low=0, high=1000, shape=(num_nodes * 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-100, high=100, shape=(num_nodes,), dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([self.inventory, self.current_demand]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = self.np_random.uniform(200, 600, size=(self.num_nodes,))
        self.current_demand = self.np_random.poisson(lam=30, size=(self.num_nodes,))
        return self._get_obs(), {}

    def step(self, action):
        self.inventory = np.clip(self.inventory + action, 0, 1000)
        sales = np.minimum(self.inventory, self.current_demand)
        stockouts = np.maximum(0, self.current_demand - self.inventory)
        self.inventory -= sales
        reward = (np.sum(sales)*15) - (np.sum(self.inventory)*0.2 + np.sum(np.abs(action))*1.5 + np.sum(stockouts)*25)
        self.current_demand = self.np_random.poisson(lam=30, size=(self.num_nodes,))
        return self._get_obs(), reward, False, False, {"stockouts": np.sum(stockouts)}
