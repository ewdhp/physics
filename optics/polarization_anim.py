"""
How it works

The blue vector = the rotating electric field of incoming light.

The red vector = the transmitted component through the polarizer.

The dashed black line = the polarizer’s axis.

The intensity text shows how much light passes (I = I₀cos²θ).

When θ → 90°, the red vector disappears — total blocking.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
I0 = 1.0
theta_deg = np.linspace(0, 180, 180)
theta_rad = np.deg2rad(theta_deg)

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title("Polarizer Blocking Light at Different Angles")
ax.set_xlabel("x-axis (polarizer axis)")
ax.set_ylabel("y-axis (electric field)")

# --- Plot elements ---
light_vector, = ax.plot([], [], 'b-', lw=2, label="Incident E-field")
projection_vector, = ax.plot([], [], 'r-', lw=2, label="Transmitted E-field")
polarizer_axis, = ax.plot([0, 1], [0, 0], 'k--', lw=2, label="Polarizer axis")

# --- Text annotation ---
intensity_text = ax.text(-1.1, 1.05, '', fontsize=12)
ax.legend()

# --- Initialization function ---
def init():
    light_vector.set_data([], [])
    projection_vector.set_data([], [])
    intensity_text.set_text('')
    return light_vector, projection_vector, intensity_text

# --- Update for each frame ---
def update(i):
    angle = theta_rad[i]
    E0 = np.array([np.sin(angle), np.cos(angle)])  # rotating E-field
    proj = np.array([E0[0], 0])                   # projection on polarizer axis (x-axis)
    intensity = np.dot(proj, proj)

    # Update vectors
    light_vector.set_data([0, E0[0]], [0, E0[1]])
    projection_vector.set_data([0, proj[0]], [0, proj[1]])
    intensity_text.set_text(f"Angle: {theta_deg[i]:.0f}° | I = {I0*np.cos(angle)**2:.2f}")

    return light_vector, projection_vector, intensity_text

# --- Create the animation ---
ani = animation.FuncAnimation(fig, update, frames=len(theta_deg),
                              init_func=init, interval=50, blit=True)

plt.show()
