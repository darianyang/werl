import matplotlib.pyplot as plt
import numpy as np

class Particle:
    def __init__(self, mass, friction, temperature, initial_position):
        self.mass = mass
        self.friction = friction
        self.temperature = temperature
        self.position = initial_position

    def update(self, dt):
        # Simulating overdamped motion in 1D with a double-well potential
        gradient_U = 4 * self.position**3 - 2 * self.position

        # Calculate the random displacement (noise) with correct scaling
        # sqrt of the expected variance = stdev for scaling
        stdev_x_rand = np.sqrt(2 * self.temperature * dt / (self.mass * self.friction))
        x_rand = np.random.normal(0, stdev_x_rand)

        # Update position
        self.position -= (dt / self.friction) * gradient_U + x_rand

# Function to calculate the double-well potential
def double_well_potential(x):
    return x**4 - x**2

# Simulation parameters
mass = 1
friction = 10
temperature = 300.0  # Kelvin
initial_position = 0.0
total_time = 5.0  # total simulation time in seconds
dt = 0.01  # time step in seconds

# Particle initialization
particle = Particle(mass, friction, temperature, initial_position)

# Lists to store data for plotting
time_points = np.arange(0, total_time, dt)
positions = []

# Main simulation loop
for time in time_points:
    particle.update(dt)
    positions.append(particle.position)

# Calculate potential values for plotting
potential_values = double_well_potential(np.linspace(-2, 2, 100))

# Plotting the particle trajectory and potential
plt.plot(time_points, positions, label='Particle Trajectory')
#plt.plot(np.linspace(-2, 2, 100), potential_values, label='Double-Well Potential', linestyle='--', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()
plt.title('Single Particle in 1D Double-Well Potential (ODLD)')
plt.show()
