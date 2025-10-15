import numpy as np
import matplotlib   
try:
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')
    print("⚠️  Using Agg backend - plots will be saved but not displayed")
import matplotlib.pyplot as plt

# --- Material refractive indices ---
n1 = 1.0     # medium 1: air
n2 = 1.5     # medium 2: glass

# --- Angles of incidence ---
theta_i = np.linspace(0, 90, 1000) * np.pi / 180  # radians

# --- Fresnel reflection coefficients ---
# Using Snell's law
theta_t = np.arcsin(n1 / n2 * np.sin(theta_i))

# Fresnel equations for reflection
rs = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
rp = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))

# Reflectance (intensity ratio)
Rs = np.abs(rs)**2
Rp = np.abs(rp)**2

# Brewster angle (where Rp = 0)
theta_B = np.arctan(n2 / n1) * 180 / np.pi

# --- Plot ---
plt.figure(figsize=(8,6))
plt.plot(theta_i * 180 / np.pi, Rs, label='Rₛ (perpendicular polarization)')
plt.plot(theta_i * 180 / np.pi, Rp, label='Rₚ (parallel polarization)')
plt.axvline(theta_B, color='r', linestyle='--', label=f'Brewster angle ≈ {theta_B:.1f}°')

plt.title("Reflection Coefficients vs. Angle of Incidence")
plt.xlabel("Angle of Incidence (degrees)")
plt.ylabel("Reflectance (fraction of light reflected)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
