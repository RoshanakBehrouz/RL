# Minimal RL + LLM Demo

This project demonstrates the basic concepts of Reinforcement Learning (RL) and Large Language Models (LLMs) in a simple, easy-to-understand way.

## What does this project do?
- **Environment:** Simulates a tiny text-based game where the agent must find the treasure by choosing the correct action.
- **LLM (Simulated):** Uses a function to suggest the best action based on the current state description, mimicking how an LLM might help an RL agent.
- **RL Agent:** Chooses actions using a mix of exploration (random choice) and exploitation (following the LLM's suggestion).
- **Learning:** For simplicity, the agent does not update its policy, but the structure shows how RL and LLMs can interact.

## Why is this useful?
- Shows how RL agents can use LLMs to interpret text environments or get action suggestions.
- Provides a minimal, hands-on example suitable for learning or demonstration purposes.

## How to run
1. Make sure you have Python 3 installed.
2. Run the demo script:
   ```bash
   python rl_llm_demo.py
   ```
3. Watch the output in your terminal. The agent will play a few episodes of the game, using the simulated LLM to help choose actions.

## File Overview
- `rl_llm_demo.py`: The main script containing the environment, agent, LLM helper, and training loop.

## Customization
- You can replace the `llm_suggest_action` function with a real LLM call (e.g., OpenAI API) for more advanced experiments.

---
This project is designed to be as simple as possible for educational purposes. Perfect for showing your understanding of RL and LLM concepts! 