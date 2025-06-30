import openai
import numpy as np

# Set your OpenAI API key
openai.api_key = "to be written here the API key"

# Define possible prompts (actions)
prompts = [
    "What is the capital of France?",
    "Name the capital city of France.",
    "France's capital is?",
    "Which city is the capital of France?",
    "Tell me the capital of France."
]

# Q-table: one value per prompt
Q = np.zeros(len(prompts))

# RL parameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2  # exploration rate
episodes = 30

# Reward function: +1 if "Paris" in answer, else 0
def get_reward(response):
    return 1 if "paris" in response.lower() else 0

# Function to query the LLM
def query_llm(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or another engine
        prompt=prompt,
        max_tokens=10
    )
    return response.choices[0].text.strip()

# RL loop
for ep in range(episodes):
    # Epsilon-greedy action selection
    if np.random.rand() < epsilon:
        action = np.random.randint(len(prompts))
    else:
        action = np.argmax(Q)
    prompt = prompts[action]
    response = query_llm(prompt)
    reward = get_reward(response)
    # Q-learning update
    Q[action] = Q[action] + alpha * (reward + gamma * np.max(Q) - Q[action])
    print(f"Episode {ep+1}: Prompt: '{prompt}' | Response: '{response}' | Reward: {reward}")

print("\nBest prompt found:", prompts[np.argmax(Q)])
