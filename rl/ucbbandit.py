import numpy as np
import matplotlib.pyplot as plt

class MultiArmBandit:
    # TODO: epsilon?
    def __init__(self, num_arms, true_means, alpha):
        self.num_arms = num_arms
        self.true_means = true_means
        self.alpha = alpha

        # Initialize variables
        self.num_pulls = np.zeros(num_arms)
        self.cumulative_rewards = np.zeros(num_arms)
        self.timestep = 0
        self.avg_cumulative_rewards = []

    def pull_arm(self, arm):
        reward = np.random.normal(loc=self.true_means[arm], scale=1.0)
        self.num_pulls[arm] += 1
        self.cumulative_rewards[arm] += reward
        self.timestep += 1
        self.avg_cumulative_rewards.append(np.sum(self.cumulative_rewards) / self.timestep)

    def choose_arm_ucb(self):
        ucb_values = np.zeros(self.num_arms)
        
        for arm in range(self.num_arms):
            if self.num_pulls[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                mean_reward = self.cumulative_rewards[arm] / self.num_pulls[arm]
                uncertainty = np.sqrt(self.alpha * np.log(self.timestep) / self.num_pulls[arm])
                ucb_values[arm] = mean_reward + uncertainty
        
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm

# Example usage with plotting average cumulative reward over time
num_arms = 5
true_means = [1.0, 0.8, 0.5, 0.3, 0.1]
alpha = 2.0

bandit = MultiArmBandit(num_arms, true_means, alpha)

# Play for a fixed number of steps
num_steps = 1000
for _ in range(num_steps):
    chosen_arm = bandit.choose_arm_ucb()
    bandit.pull_arm(chosen_arm)

# Plot the average cumulative reward over time
plt.plot(bandit.avg_cumulative_rewards)
plt.xlabel('Timestep')
plt.ylabel('Average Cumulative Reward')
plt.title('Average Cumulative Reward over Timesteps')
plt.show()

# Print the estimated means for each arm
for arm in range(num_arms):
    estimated_mean = bandit.cumulative_rewards[arm] / bandit.num_pulls[arm] if bandit.num_pulls[arm] > 0 else 0.0
    print(f"Arm {arm + 1}: True Mean = {true_means[arm]}, Estimated Mean = {estimated_mean}, Pulls = {bandit.num_pulls[arm]}")
