import numpy as np
import matplotlib
# Set backend for GUI display
try:
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')
    print("⚠️  Using Agg backend - plots will be saved but not displayed")
import matplotlib.pyplot as plt
# Incident light intensity
I0 = 1.0

# Angle between light polarization and polarizer axis (degrees)
theta_deg = np.linspace(0, 180, 500)
theta_rad = np.deg2rad(theta_deg)

# Malus' Law
I_transmitted = I0 * np.cos(theta_rad)**2

# Plotting
plt.figure(figsize=(8,5))
plt.plot(theta_deg, I_transmitted, color='blue', lw=2)
plt.title("Light Transmission Through a Polarizer (Malus' Law)")
plt.xlabel("Angle between light and polarizer axis (degrees)")
plt.ylabel("Transmitted Intensity")
plt.grid(True)
plt.axvline(90, color='red', linestyle='--', label='Complete blocking at 90°')
plt.legend()
plt.show()
