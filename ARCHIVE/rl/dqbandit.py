import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the DQN model
class DQNBandit(nn.Module):
    def __init__(self, num_bandits):
        super(DQNBandit, self).__init__()
        self.num_bandits = num_bandits
        self.fc = nn.Linear(num_bandits, 32)  # Fully connected layer
        self.output_layer = nn.Linear(32, 1)  # Output layer

    def forward(self, state):
        x = torch.relu(self.fc(state))
        return self.output_layer(x)

# Define the DQN agent
class DQNAgent:
    def __init__(self, num_bandits, learning_rate=0.001, gamma=0.99):
        self.num_bandits = num_bandits
        self.model = DQNBandit(num_bandits)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma

    def choose_bandit(self, state):
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def update_model(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        current_q = self.model(state)[action]
        next_q = torch.max(self.model(next_state))

        target_q = reward + self.gamma * next_q
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Function to run the bandit experiment
def run_bandit_experiment(num_bandits, num_steps, true_means):
    agent = DQNAgent(num_bandits)
    rewards = []

    for step in range(num_steps):
        state = np.ones(num_bandits)  # Simplified state representation
        chosen_bandit = agent.choose_bandit(state)
        reward = np.random.normal(true_means[chosen_bandit], 1.0)
        next_state = np.ones(num_bandits)  # Simplified next state representation

        # Update the DQN model based on the observed reward
        agent.update_model(state, chosen_bandit, reward, next_state)
        rewards.append(reward)

    return rewards

# Number of bandits
num_bandits = 5

# True means of the bandits (unknown to the agent)
true_means = np.random.normal(0, 1, num_bandits)

# Number of time steps
num_steps = 1000

# Run the bandit experiment with DQN
rewards = run_bandit_experiment(num_bandits, num_steps, true_means)

# Plot the cumulative rewards over time
cumulative_rewards = np.cumsum(rewards) / np.arange(1, num_steps + 1)
plt.plot(cumulative_rewards)
plt.title('Deep Q-Learning Bandit Algorithm (PyTorch)')
plt.xlabel('Time Step')
plt.ylabel('Average Cumulative Reward')
plt.show()
