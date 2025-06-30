# Minimal RL + LLM Demo

This project demonstrates the core concepts of Reinforcement Learning (RL) and Large Language Models (LLMs) through a simple, hands-on example.

## Project Overview
- **Environment:** A basic text-based game where the agent must find the treasure by choosing the correct action.
- **Agents:**
  - **LLM-guided Agent:** Uses a function to suggest the best action based on the current state, simulating how an LLM might assist an RL agent.
  - **Q-learning Agent:** Learns from experience using the Q-learning algorithm to improve its actions over time.
- **Interactive App:** Includes a Streamlit web app for stepping through episodes and visualizing the agent's decisions.


## How to Run

### Command-Line Version
1. Make sure you have Python 3 installed.
2. Run the script:
   ```bash
   python3 rl_llm_demo.py
   ```
3. When prompted, choose the agent type ('llm' or 'q') and observe the agent's behavior in the terminal.

### Streamlit Interactive App
1. Install Streamlit if you haven't already:
   ```bash
   pip install streamlit
   ```
2. Start the app:
   ```bash
   streamlit run app.py
   ```
3. Use the web interface to select the agent type, step through actions, and reset episodes as needed.

## Files
- `rl_llm_demo.py`: Main script for running the RL + LLM demo in the terminal.
- `app.py`: Streamlit app for interactive exploration.

---
This project is intended for educational purposes and as a starting point for further exploration of RL and LLM integration. 
