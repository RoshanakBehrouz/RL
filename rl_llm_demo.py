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

# --- RL Agent (LLM-guided) ---
class RLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.actions = ["go north", "go east", "look around"]
        self.q_table = {}       # (state, action) -> value

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state_text):
        # Epsilon-greedy: explore or exploit
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Exploit: use LLM suggestion, but if multiple actions have same Q, use LLM to break ties
        q_values = [self.get_q(state_text, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        if len(best_actions) == 1:
            return best_actions[0]
        # If tie, use LLM suggestion among best actions
        llm_action = llm_suggest_action(state_text)
        if llm_action in best_actions:
            return llm_action
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        old_q = self.get_q(state, action)
        next_qs = [self.get_q(next_state, a) for a in self.actions]
        max_next_q = max(next_qs) if next_qs else 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}  # (state, action) -> value
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = actions

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Otherwise, pick the action with the highest Q-value
        q_values = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_values)
        # In case of ties, randomly choose among the best
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        old_q = self.get_q(state, action)
        next_qs = [self.get_q(next_state, a) for a in self.actions]
        max_next_q = max(next_qs) if next_qs else 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

# --- Main Training Loop ---
def main():
    env = SimpleTextGame()
    actions = ["go north", "go east", "look around"]

    # Choose agent type: 'llm' for LLM-guided, 'q' for Q-learning
    agent_type = input("Choose agent type ('llm' or 'q'): ").strip().lower()
    if agent_type == 'q':
        agent = QLearningAgent(actions)
        print("Using Q-learning agent.")
    else:
        agent = RLAgent()
        print("Using LLM-guided agent.")

    for episode in range(5):
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