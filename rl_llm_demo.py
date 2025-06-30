# Minimal RL + LLM Demo
# This script demonstrates the concept of using RL with an LLM in a simple text-based environment.
# The 'LLM' here is a function that suggests the best action based on the state description.

import random

# --- Simple Text-Based Environment ---
class SimpleTextGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "You are in a room. Exits: north, east."
        self.done = False
        return self.state

    def step(self, action):
        if action.lower() == "go north":
            self.state = "You found the treasure! Game over."
            self.done = True
            reward = 1
        else:
            self.state = "Nothing happens. Try again."
            reward = 0
        return self.state, reward, self.done

# --- 'LLM' Helper Function ---
def llm_suggest_action(state_text):
    """
    Simulates an LLM by suggesting the best action for the given state.
    In a real project, this could call OpenAI or HuggingFace API.
    """
    if "Exits: north" in state_text:
        return "go north"
    return "look around"

# --- RL Agent ---
class RLAgent:
    def __init__(self):
        self.epsilon = 0.2  # Exploration rate

    def select_action(self, state_text):
        # With probability epsilon, pick a random action (exploration)
        if random.random() < self.epsilon:
            return random.choice(["go north", "go east", "look around"])
        # Otherwise, use the LLM suggestion (exploitation)
        return llm_suggest_action(state_text)

    def update(self, state, action, reward, next_state):
        # For this simple demo, we don't implement learning
        pass

# --- Main Training Loop ---
def main():
    env = SimpleTextGame()
    agent = RLAgent()

    for episode in range(3):
        print(f"\nEpisode {episode+1}")
        state = env.reset()
        done = False
        step_count = 0
        while not done and step_count < 5:
            print(f"State: {state}")
            action = agent.select_action(state)
            print(f"Agent action: {action}")
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            step_count += 1
            if reward > 0:
                print("Reward received! 🎉")
        print(f"Final state: {state}")

if __name__ == "__main__":
    main() 