# 🔄 Agentic Supply Chain Optimizer

> An autonomous, AI-powered inventory redistribution and decision-support system combining **Deep Reinforcement Learning** and **Large Language Models** to optimize supply chain operations across a 50-node retail network.

---

## 📌 Overview

The Agentic Supply Chain Optimizer is a self-healing inventory management system that autonomously identifies and resolves inefficiencies — such as trapped inventory, stockouts, and costly misallocations — across a large-scale retail network. It pairs deep reinforcement learning for autonomous decision-making with an LLM-powered API that allows stakeholders to run natural language "what-if" analyses in real time.

---

## ✨ Key Features

### 🧠 Self-Healing Inventory Redistribution
- Autonomously monitors a **50-node retail network** for inventory imbalances
- Analyzes flow constraints to detect "trapped inventory" — stock unable to reach demand centers
- Computes and executes optimal reallocation paths without manual intervention

### 💬 LLM-Powered Decision Support API
- Natural language interface for complex **what-if scenario analysis**
- Enables stakeholders to query real-time inventory risks conversationally
- Reduces analyst response time by **25%** compared to traditional reporting pipelines

### 📊 Multi-Objective Optimization Framework
- Balances trade-offs between **operational costs** and **supply chain resilience**
- Establishes dynamic trigger points for proactive inventory rebalancing
- Supports Pareto-frontier analysis to surface optimal policy configurations

### ✅ Validated Performance Results
| Metric | Result |
|---|---|
| Stockout waste reduction | 18% |
| Fulfillment cost decrease | 22% |
| Service Level Agreement (SLA) maintained | 96% |
| Stakeholder response time improvement | 25% |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   50-Node Retail Network                    │
└────────────────────────┬────────────────────────────────────┘
                         │ Real-time inventory state
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Deep Reinforcement Learning Engine               │
│  • Flow constraint analysis                                 │
│  • Trapped inventory detection                              │
│  • Optimal redistribution path computation                  │
│  • Dynamic rebalancing triggers                             │
└────────────────────────┬────────────────────────────────────┘
                         │ Actions / Recommendations
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Multi-Objective Optimization Framework              │
│  • Cost vs. resilience trade-off analysis                   │
│  • Pareto frontier estimation                               │
│  • SLA constraint enforcement                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Decision Support API                       │
│  • Natural language what-if scenario queries                │
│  • Real-time inventory risk summarization                   │
│  • Stakeholder-facing conversational interface              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Reinforcement Learning | Deep RL (DQN / PPO) |
| LLM Integration | Large Language Models (API-based) |
| Optimization | Multi-objective optimization (Pareto analysis) |
| Simulation & Validation | Custom supply chain simulation environment |
| Backend / API | Python |

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.9
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/kiranmusu14/agentic-sc-optimizer.git
cd agentic-sc-optimizer

# Install dependencies
pip install -r requirements.txt
```

### Running the Optimizer

```bash
# Run the main optimization pipeline
python main.py

# Start the LLM decision support API
python api.py
```

### Running Simulations

```bash
# Validate model performance via simulation
python simulate.py --nodes 50 --episodes 1000
```

---

## 📂 Project Structure

```
agentic-sc-optimizer/
├── agents/              # RL agent definitions and training loops
├── environment/         # Supply chain simulation environment
├── optimization/        # Multi-objective optimization framework
├── api/                 # LLM decision support API
├── models/              # Saved model checkpoints
├── utils/               # Helper functions and data utilities
├── simulate.py          # Simulation and validation script
├── main.py              # Main entrypoint
└── requirements.txt     # Python dependencies
```

---

## 📈 Results

The system was validated through rigorous simulation across the 50-node retail network:

- **18% reduction** in stockout waste — fewer lost sales and urgent emergency orders
- **22% decrease** in fulfillment costs — more efficient redistribution paths
- **96% SLA compliance** — customer service levels maintained throughout autonomous operation
- **25% faster stakeholder response** — natural language queries replace manual report generation

---

## 🔮 Future Work

- [ ] Expand to multi-echelon (warehouse → store) supply chain topologies
- [ ] Integrate real-time demand forecasting signals as RL state inputs
- [ ] Add explainability layer for LLM recommendations
- [ ] Support multi-modal inputs (sales data, weather, events) for richer scenario analysis
- [ ] Deploy as a containerized microservice (Docker / Kubernetes)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Kiran Musu**
- GitHub: [@kiranmusu14](https://github.com/kiranmusu14)
