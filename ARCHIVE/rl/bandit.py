import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, num_bandits, epsilon):
        self.num_bandits = num_bandits
        self.epsilon = epsilon
        self.q_values = np.zeros(num_bandits)  # Estimated values of each bandit
        self.action_counts = np.zeros(num_bandits)  # Number of times each bandit has been chosen

    def choose_bandit(self):
        if np.random.rand() < self.epsilon:
            # Explore: Choose a random bandit
            return np.random.randint(self.num_bandits)
        else:
            # Exploit: Choose the bandit with the highest estimated value
            return np.argmax(self.q_values)

    def update_bandit(self, bandit, reward):
        # Update the estimated value of the chosen bandit
        self.action_counts[bandit] += 1
        self.q_values[bandit] += (reward - self.q_values[bandit]) / self.action_counts[bandit]

def run_bandit_experiment(num_bandits, num_steps, true_means, epsilon):
    bandit = EpsilonGreedyBandit(num_bandits, epsilon)
    rewards = []

    for step in range(num_steps):
        chosen_bandit = bandit.choose_bandit()
        # Simulate the reward from the chosen bandit (assuming a normal distribution with true mean)
        reward = np.random.normal(true_means[chosen_bandit], 1.0)
        bandit.update_bandit(chosen_bandit, reward)
        rewards.append(reward)

    return rewards

# Number of bandits
num_bandits = 5

# True means of the bandits (unknown to the agent)
true_means = np.random.normal(0, 1, num_bandits)

# Number of time steps
num_steps = 1000

# Exploration-exploitation trade-off parameter (epsilon)
epsilon = 0.1

# Run the bandit experiment
rewards = run_bandit_experiment(num_bandits, num_steps, true_means, epsilon)

# Plot the cumulative rewards over time
cumulative_rewards = np.cumsum(rewards) / np.arange(1, num_steps + 1)
plt.plot(cumulative_rewards)
plt.title('Epsilon-Greedy Bandit Algorithm')
plt.xlabel('Time Step')
plt.ylabel('Average Cumulative Reward')
plt.show()
