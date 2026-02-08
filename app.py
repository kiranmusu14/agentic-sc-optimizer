import streamlit as st
import numpy as np
from src.tools.sc_agent import agent
from src.utils.visualize import run_visualization

st.set_page_config(page_title="Agentic Supply Chain Optimizer", layout="wide")

st.title("📦 Agentic Supply Chain Optimizer")
st.markdown("### Deep Reinforcement Learning & LLM Decision Support")

query = st.text_input("Ask the Optimizer a 'What-If' question:", 
                     "Give me a risk summary of the 50-node network.")

if st.button("Run Analysis"):
    with st.spinner("Agent is reasoning..."):
        response = agent.run(query)
        st.write("### 🤖 Agent Response:")
        st.info(response)

if st.sidebar.button("Generate Network Heatmap"):
    st.sidebar.write("Generating current network state...")
    run_visualization()
    st.sidebar.image("data/network_risk_map.png")
