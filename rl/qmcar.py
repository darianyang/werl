import numpy as np
import gymnasium as gym

# Q-learning agent for MountainCar environment
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_prob=0.2):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Determine the state space size based on the observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_space_size = tuple(((self.env.observation_space.high - self.env.observation_space.low) * np.array([10, 100])).astype(int) + 1)
        elif isinstance(env.observation_space, gym.spaces.Dict):
            # Assuming the relevant values are stored under keys 'position' and 'velocity'
            self.state_space_size = tuple(((self.env.observation_space.spaces['position'].high - self.env.observation_space.spaces['position'].low) * np.array([10, 100])).astype(int) + 1)
        else:
            raise ValueError("Unsupported observation space type")

        self.action_space_size = env.action_space.n
        self.q_table = np.zeros(self.state_space_size + (self.action_space_size,))

    def discretize_state(self, state):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            # state_discrete = (state - self.env.observation_space.low) * np.array([10, 100])
            # return tuple(state_discrete.astype(int))
            state_discrete = tuple(((self.env.observation_space.low) * np.array([10, 100])).astype(int))
            return state_discrete
            # state_discrete = (state - self.env.observation_space.low) * np.array([10, 100])
            # return tuple(state_discrete.astype(int))

        elif isinstance(self.env.observation_space, gym.spaces.Dict):
            # Assuming the relevant values are stored under keys 'position' and 'velocity'
            state_discrete = (state['position'] - self.env.observation_space.spaces['position'].low) * np.array([10, 100])
            return tuple(state_discrete.astype(int))
        else:
            raise ValueError("Unsupported observation space type")

    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return self.env.action_space.sample()  # Explore
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])

    def update_q_table(self, state, action, reward, next_state):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        current_q = self.q_table[discrete_state][action]
        max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[discrete_state][action] = new_q


# Training function for Q-learning agent in MountainCar environment
def train_q_learning_agent(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            step_result = env.step(action)

            # Ensure that step_result has at least 4 values
            if len(step_result) < 4:
                raise ValueError("Invalid step result from environment")

            next_state, reward, done, _ = step_result[:4]
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


# Main function
if __name__ == "__main__":
    # Create MountainCar environment
    env = gym.make('MountainCar-v0', render_mode="human")

    # Initialize Q-learning agent
    q_learning_agent = QLearningAgent(env)

    # Train the Q-learning agent
    train_q_learning_agent(env, q_learning_agent)

    # Test the trained agent
    state = env.reset()
    total_reward = 0

    while True:
        env.render()
        action = q_learning_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            break

    print(f"Test Total Reward: {total_reward}")

    env.close()
