import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------
# Parameters
# ------------------------
L = 1.0          # length of pendulum
g = 9.81         # gravity
theta0 = 0.5     # initial angle (rad)
t_max = 10
dt = 0.02
t = np.arange(0, t_max, dt)

# ------------------------
# Kinematics-only pendulum
# θ(t) defined directly
# ------------------------

# some arbitrary motion
theta_kin = theta0 * np.cos(2 * t)

omega_kin = np.gradient(theta_kin, dt)
alpha_kin = np.gradient(omega_kin, dt)

# ------------------------
# Dynamics-driven pendulum (small angle)
# α = -g/L * θ
# ------------------------
theta_dyn = np.zeros_like(t)
omega_dyn = np.zeros_like(t)
alpha_dyn = np.zeros_like(t)
theta_dyn[0] = theta0
omega_dyn[0] = 0.0

for i in range(1, len(t)):
    # small-angle approximation
    alpha_dyn[i-1] = - (g / L) * theta_dyn[i-1]

    omega_dyn[i] = omega_dyn[i-1] + alpha_dyn[i-1] * dt
    theta_dyn[i] = theta_dyn[i-1] + omega_dyn[i] * dt

alpha_dyn[-1] = - (g / L) * theta_dyn[-1]

# ------------------------
# Animation setup
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax in axes:
    ax.set_xlim(-L-0.2, L+0.2)
    ax.set_ylim(-L-0.2, 0.2)
    ax.set_aspect('equal')

lines = [axes[0].plot([], [], 'o-', lw=4, color='red')[0],
         axes[1].plot([], [], 'o-', lw=4, color='blue')[0]]

axes[0].set_title("Kinematics-only")
axes[1].set_title("Dynamics-driven")

def update(frame):
    # Kinematics pendulum
    x_kin = L * np.sin(theta_kin[frame])
    y_kin = -L * np.cos(theta_kin[frame])
    lines[0].set_data([0, x_kin], [0, y_kin])
    
    # Dynamics pendulum
    x_dyn = L * np.sin(theta_dyn[frame])
    y_dyn = -L * np.cos(theta_dyn[frame])
    lines[1].set_data([0, x_dyn], [0, y_dyn])
    
    return lines

ani = FuncAnimation(fig, update, frames=len(t), interval=dt*1000)
plt.show()
