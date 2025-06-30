import streamlit as st
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
    if "Exits: north" in state_text:
        return "go north"
    return "look around"

# --- RL Agent (LLM-guided) ---
class RLAgent:
    def __init__(self):
        self.epsilon = 0.2
        self.actions = ["go north", "go east", "look around"]

    def select_action(self, state_text):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return llm_suggest_action(state_text)

    def update(self, state, action, reward, next_state):
        pass

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        next_qs = [self.get_q(next_state, a) for a in self.actions]
        max_next_q = max(next_qs) if next_qs else 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

# --- Streamlit App ---
st.set_page_config(page_title="RL + LLM Interactive Demo", layout="centered")
st.title("Minimal RL + LLM Interactive Demo")
st.write("Step through a simple text-based game using either an LLM-guided agent or a Q-learning agent.")

# --- Session State Initialization ---
if 'env' not in st.session_state:
    st.session_state.env = SimpleTextGame()
if 'agent_type' not in st.session_state:
    st.session_state.agent_type = 'llm'
if 'agent' not in st.session_state:
    st.session_state.agent = RLAgent()
if 'state' not in st.session_state:
    st.session_state.state = st.session_state.env.reset()
if 'done' not in st.session_state:
    st.session_state.done = False
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0
if 'last_action' not in st.session_state:
    st.session_state.last_action = None
if 'last_reward' not in st.session_state:
    st.session_state.last_reward = None

# --- Agent Selection ---
ag_type = st.radio("Choose agent type:", ["LLM-guided", "Q-learning"],
                  index=0 if st.session_state.agent_type == 'llm' else 1)

if ag_type == "LLM-guided":
    if st.session_state.agent_type != 'llm':
        st.session_state.agent = RLAgent()
        st.session_state.agent_type = 'llm'
        st.session_state.state = st.session_state.env.reset()
        st.session_state.done = False
        st.session_state.step_count = 0
        st.session_state.last_action = None
        st.session_state.last_reward = None
else:
    if st.session_state.agent_type != 'q':
        st.session_state.agent = QLearningAgent(["go north", "go east", "look around"])
        st.session_state.agent_type = 'q'
        st.session_state.state = st.session_state.env.reset()
        st.session_state.done = False
        st.session_state.step_count = 0
        st.session_state.last_action = None
        st.session_state.last_reward = None

# --- Display Current State ---
st.subheader("Environment State:")
st.info(st.session_state.state)

# --- Step Button ---
col1, col2 = st.columns(2)
with col1:
    step = st.button("Next Step", disabled=st.session_state.done)
with col2:
    reset = st.button("Reset Episode")

if step and not st.session_state.done:
    action = st.session_state.agent.select_action(st.session_state.state)
    next_state, reward, done = st.session_state.env.step(action)
    st.session_state.agent.update(st.session_state.state, action, reward, next_state)
    st.session_state.last_action = action
    st.session_state.last_reward = reward
    st.session_state.state = next_state
    st.session_state.done = done
    st.session_state.step_count += 1

if reset:
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.step_count = 0
    st.session_state.last_action = None
    st.session_state.last_reward = None
    if st.session_state.agent_type == 'q':
        st.session_state.agent = QLearningAgent(["go north", "go east", "look around"])
    else:
        st.session_state.agent = RLAgent()

# --- Show Last Action and Reward ---
if st.session_state.last_action is not None:
    st.write(f"**Agent action:** {st.session_state.last_action}")
    st.write(f"**Reward:** {st.session_state.last_reward}")
    st.write(f"**Done:** {st.session_state.done}")
    st.write(f"**Step:** {st.session_state.step_count}")

if st.session_state.done:
    st.success("Episode finished! Click 'Reset Episode' to play again.") 