import streamlit as st
import random

# --- HuggingFace Transformers for LLM ---
try:
    from transformers import pipeline
except ImportError:
    st.warning("Please install the 'transformers' library: pip install transformers")
    pipeline = None

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

# --- LLM Helper using HuggingFace and Few-shot Prompting ---
@st.cache_resource(show_spinner=False)
def get_llm_pipeline():
    if pipeline is None:
        return None
    return pipeline("text-generation", model="distilgpt2")


def llm_suggest_action_with_reasoning(state_text):
    """
    Uses a HuggingFace LLM to suggest an action using few-shot prompting.
    The prompt gives examples of state-action pairs to guide the model.
    """
    llm = get_llm_pipeline()
    if llm is None:
        return ("[LLM not available]", "look around")
    # Few-shot prompt: show examples, then ask for the next action
    prompt = (
        "State: You are in a room. Exits: north, east.\n"
        "Action: go north\n"
        "State: Nothing happens. Try again.\n"
        "Action: look around\n"
        f"State: {state_text}\n"
        "Action:"
    )
    try:
        output = llm(prompt, max_new_tokens=5, num_return_sequences=1)[0]['generated_text']
    except Exception as e:
        return (f"[LLM error: {e}]", "look around")
    # Extract the action after the last 'Action:'
    action = "look around"  # default
    if 'Action:' in output:
        action_line = output.split('Action:')[-1].strip().split('\n')[0]
        # Only allow valid actions
        for valid_action in ["go north", "go east", "look around"]:
            if valid_action in action_line.lower():
                action = valid_action
                break
        else:
            # If LLM output is not a valid action, return a warning
            return (f"[LLM output not recognized: {action_line}]", "look around")
    else:
        return (f"[LLM output missing 'Action:']", "look around")
    return ("", action)  # No reasoning, just action

# --- RL Agent (LLM-guided, now with real LLM) ---
class RLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = ["go north", "go east", "look around"]
        self.q_table = {}  # (state, action) -> value
        self.last_reasoning = ""

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state_text):
        # Epsilon-greedy: explore or exploit
        if random.random() < self.epsilon:
            self.last_reasoning = "Exploring: chose a random action."
            return random.choice(self.actions)
        # Exploit: use LLM suggestion, but if multiple actions have same Q, use LLM to break ties
        # Find best Q-value(s)
        q_values = [self.get_q(state_text, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        if len(best_actions) == 1:
            return best_actions[0]
        # If tie, use LLM suggestion among best actions
        reasoning, llm_action = llm_suggest_action_with_reasoning(state_text)
        self.last_reasoning = reasoning
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

# --- Q-Learning Agent (unchanged) ---
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
st.write("Step through a simple text-based game using either an LLM-guided agent (with real LLM action suggestion) or a Q-learning agent.")

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
if 'last_reasoning' not in st.session_state:
    st.session_state.last_reasoning = ""

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
        st.session_state.last_reasoning = ""
else:
    if st.session_state.agent_type != 'q':
        st.session_state.agent = QLearningAgent(["go north", "go east", "look around"])
        st.session_state.agent_type = 'q'
        st.session_state.state = st.session_state.env.reset()
        st.session_state.done = False
        st.session_state.step_count = 0
        st.session_state.last_action = None
        st.session_state.last_reward = None
        st.session_state.last_reasoning = ""

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
    # For LLM agent, show reasoning (now empty)
    if hasattr(st.session_state.agent, 'last_reasoning'):
        st.session_state.last_reasoning = getattr(st.session_state.agent, 'last_reasoning', "")
    else:
        st.session_state.last_reasoning = ""
    st.session_state.state = next_state
    st.session_state.done = done
    st.session_state.step_count += 1

if reset:
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.step_count = 0
    st.session_state.last_action = None
    st.session_state.last_reward = None
    st.session_state.last_reasoning = ""
    if st.session_state.agent_type == 'q':
        st.session_state.agent = QLearningAgent(["go north", "go east", "look around"])
    else:
        st.session_state.agent = RLAgent()

# --- Show Last Action, Reward, and Reasoning ---
if st.session_state.last_action is not None:
    st.write(f"**Agent action:** {st.session_state.last_action}")
    st.write(f"**Reward:** {st.session_state.last_reward}")
    st.write(f"**Done:** {st.session_state.done}")
    st.write(f"**Step:** {st.session_state.step_count}")
    # Reasoning is now omitted for distilgpt2

if st.session_state.done:
    st.success("Episode finished! Click 'Reset Episode' to play again.") 