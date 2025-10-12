"""
What this script shows:
The arm swings back and forth under gravity (non-linear).
Blue arrow: tangential velocity vector.
Green arrow: total acceleration vector (centripetal + tangential).
You can see how angular acceleration is not constant, especially 
near the extremes of motion.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------
# Parameters
# ------------------------
L = 1.0            # length of limb in meters
g = 9.81           # gravity
theta0 = np.pi/4   # initial angle (45 degrees)
omega0 = 0.0       # initial angular velocity
t_max = 5          # total time in seconds
dt = 0.01          # time step

# ------------------------
# Time array
# ------------------------
t = np.arange(0, t_max, dt)
theta = np.zeros_like(t)
omega = np.zeros_like(t)
alpha = np.zeros_like(t)

# Initial conditions
theta[0] = theta0
omega[0] = omega0

# ------------------------
# Numerical integration (Euler method)
# ------------------------
for i in range(1, len(t)):

    # nonlinear angular acceleration
    alpha[i-1] = - (g / L) * np.sin(theta[i-1])

    omega[i] = omega[i-1] + alpha[i-1] * dt
    theta[i] = theta[i-1] + omega[i] * dt

# Compute last alpha
alpha[-1] = - (g / L) * np.sin(theta[-1])

# ------------------------
# Animation setup
# ------------------------
fig, ax = plt.subplots()
ax.set_xlim(-1.2*L, 1.2*L)
ax.set_ylim(-1.2*L, 1.2*L)
ax.set_aspect('equal')

# Limb line
line, = ax.plot([], [], 'o-', lw=4, color='red')
# Velocity and acceleration vectors
vel_vec = ax.arrow(0,0,0,0, head_width=0.05, head_length=0.1, 
                   fc='blue', ec='blue')
acc_vec = ax.arrow(0,0,0,0, head_width=0.05, head_length=0.1, 
                   fc='green', ec='green')

def update(frame):
    x = L * np.sin(theta[frame])
    y = -L * np.cos(theta[frame])
    
    # Update limb line
    line.set_data([0, x], [0, y])
    
    # Tangential velocity vector
    vx = omega[frame] * L * np.cos(theta[frame])
    vy = omega[frame] * L * np.sin(theta[frame])
    
    # Tangential + centripetal acceleration vector
    at = alpha[frame] * L
    ac = (omega[frame]**2) * L
    ax_vec = (-ac * np.sin(theta[frame]) + at * np.cos(theta[frame]))
    ay_vec = (-ac * np.cos(theta[frame]) - at * np.sin(theta[frame]))
    
    # Remove previous arrows
    ax.patches = []
    
    # Draw velocity (blue)
    ax.arrow(x, y, vx*0.1, vy*0.1, head_width=0.05, head_length=0.1, 
             fc='blue', ec='blue')
    # Draw acceleration (green)
    ax.arrow(x, y, ax_vec*0.1, ay_vec*0.1, head_width=0.05, 
             head_length=0.1, fc='green', ec='green')
    
    return line,

ani = FuncAnimation(fig, update, frames=len(t), interval=dt*1000)
plt.show()
