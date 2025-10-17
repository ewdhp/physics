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

## HOW TO FIND EACH VARIABLE

### I₀ (Incident Light Intensity)
Definition: The intensity of light BEFORE it hits the polarizer.

How to find it:
  - Experimentally: Measure light intensity with no polarizer using a light meter/photodiode
  - In this script: Set as I0 = 1.0 (normalized to 1 for simplicity)
  - Formula: I₀ = P / A (Power / cross-sectional area)

### θ (Angle Between E-field and Polarizer Axis)
Definition: The angle between the oscillation direction of the incident E-field 
            and the polarizer's transmission axis.

How to find it:
  - In this script: Rotates from 0° to 180° automatically
  - Experimentally: Rotate the polarizer and measure the angle on its scale
  - Physically: Use vector dot product if you know both field directions

### I (Transmitted Light Intensity)
Definition: The intensity of light AFTER it passes through the polarizer.

How to find it:
  - Experimentally: Measure with a light meter placed after the polarizer
  - From Malus's Law: I = I₀ cos²(θ)
  - In this script: Calculated and displayed in real-time
  - Visually: The RED VECTOR length represents this transmitted intensity!
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
