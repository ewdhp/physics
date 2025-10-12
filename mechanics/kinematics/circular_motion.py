import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 5
t_max = 10
dt = 0.01
t = np.arange(0, t_max, dt)

# Non-constant angular acceleration example: alpha(t) = 0.5 * sin(t)
alpha = 0.5 * np.sin(t)
omega0 = 2

# Numerical integration for angular velocity
omega = omega0 + np.cumsum(alpha) * dt

# Numerical integration for angular displacement
theta = np.cumsum(omega) * dt

# Linear velocities
v_t = r * omega
a_c = v_t**2 / r
a_t = r * alpha

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, alpha, label="Angular Acceleration α(t)")
plt.plot(t, omega, label="Angular Velocity ω(t)")
plt.plot(t, theta, label="Angular Displacement θ(t)")
plt.xlabel("Time (s)")
plt.legend()
plt.title("Non-Uniform Circular Motion")
plt.show()
