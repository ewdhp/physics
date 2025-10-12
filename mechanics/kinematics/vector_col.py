import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Define vectors
# ----------------------------
# Initial vector (before collision)
v_i = np.array([2, 3])   # change as needed
# Final vector (after collision)
v_f = np.array([4, 0])   # change as needed

# ----------------------------
# Compute angle between vectors
# ----------------------------
dot = np.dot(v_i, v_f)
mag_i = np.linalg.norm(v_i)
mag_f = np.linalg.norm(v_f)

theta = np.arccos(dot / (mag_i * mag_f))  # in radians
theta_deg = np.degrees(theta)

print(f"Angle between vectors = {theta_deg:.2f}°")

# ----------------------------
# Plot setup
# ----------------------------
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)

# Draw initial and final vectors
ax.quiver(0, 0, v_i[0], v_i[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Initial v_i')
ax.quiver(0, 0, v_f[0], v_f[1], angles='xy', scale_units='xy', scale=1, color='red', label='Final v_f')

# Draw angle arc
angle_points = np.linspace(0, theta, 50)
arc = np.array([np.cos(angle_points), np.sin(angle_points)])
arc = arc * min(mag_i, mag_f) * 0.3   # scale for visibility
ax.plot(arc[0], arc[1], color='green', linestyle='--')
ax.text(arc[0, 25], arc[1, 25], f"θ = {theta_deg:.1f}°", fontsize=10, color='green')

# Limits
all_x = [v_i[0], v_f[0]]
all_y = [v_i[1], v_f[1]]
ax.set_xlim(-1, max(all_x) + 2)
ax.set_ylim(-1, max(all_y) + 2)

ax.legend()
plt.title("Vector Direction Change After Collision")
plt.show()
