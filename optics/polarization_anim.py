"""
COMPLETE POLARIZATION EXPLANATION
==================================

## What is Polarization?

Light is an electromagnetic wave with an oscillating electric field (E-field). 
Polarization describes the direction and pattern of this oscillation.

### Types of Polarization:
1. Unpolarized Light: E-field oscillates randomly in all directions perpendicular to travel
2. Linear Polarization: E-field oscillates in one fixed plane
3. Circular/Elliptical Polarization: E-field traces circular or elliptical patterns

## Malus's Law (The Key Physics)

When polarized light passes through a polarizer:
    I = I₀ cos²(θ)

Where:
  - I₀ = incident light intensity
  - θ = angle between incident E-field and polarizer axis
  - I = transmitted intensity

### What This Means:
  - θ = 0°: Light aligned with polarizer → full transmission (cos²(0°) = 1)
  - θ = 45°: Light at 45° angle → 50% transmission (cos²(45°) ≈ 0.5)
  - θ = 90°: Light perpendicular to polarizer → complete blocking (cos²(90°) = 0)

## Physical Intuition

Think of it like:
  - Polarizer = a narrow gate only allowing vertical bars through
  - Light wave = bars at various angles
  - When bars align with gate → passes through
  - When bars perpendicular to gate → blocked

The red vector's length represents how much light actually gets through at each angle!

## HOW THIS SCRIPT WORKS

Frame-by-frame animation showing:

1. BLUE VECTOR (Incident E-field):
   - Rotates from 0° to 180°
   - Represents unpolarized light arriving at the polarizer
   - Direction: sin(θ) in x, cos(θ) in y

2. DASHED BLACK LINE (Polarizer Axis):
   - Fixed along x-axis (horizontal)
   - Only allows light polarized along this direction through

3. RED VECTOR (Transmitted E-field):
   - The component of blue vector along the polarizer axis
   - Calculated by projecting E-field onto x-axis
   - Disappears as θ → 90°

4. INTENSITY TEXT:
   - Shows I = I₀cos²(θ)
   - Decreases as angle increases
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
