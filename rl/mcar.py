import gym
import matplotlib.pyplot as plt

def move_car(env, num_steps=200):
    positions = []
    velocities = []

    for _ in range(num_steps):
        action = env.action_space.sample()  # Sample a random action
        observation, _, _, _ = env.step(action)[:4]
        
        # Extract position and velocity from the observation
        position, velocity = observation

        positions.append(position)
        velocities.append(velocity)

        # Render the environment
        env.render()

    return positions, velocities

def plot_movements(positions, velocities):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(positions)
    plt.title('Car Position Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Position')

    plt.subplot(2, 1, 2)
    plt.plot(velocities)
    plt.title('Car Velocity Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Velocity')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create the MountainCar environment
    env = gym.make('MountainCar-v0', render_mode="human")
    env.reset()

    # Move the car and track movements
    positions, velocities = move_car(env)

    # Plot the movements
    #plot_movements(positions, velocities)

    # Close the environment
    env.close()
