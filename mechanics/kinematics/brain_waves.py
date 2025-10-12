import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------
# Parameters
# ------------------------
t_max = 20          # total time in seconds
dt = 0.01           # time step
t = np.arange(0, t_max, dt)

# Base EEG properties
# Slow-wave sleep (delta) -> REM (beta-like)
freq_start = 2      # Hz (delta)
freq_end = 20       # Hz (REM-like)
amp_start = 100     # microvolts
amp_end = 20        # microvolts

# Linear interpolation for smooth transition
freq = freq_start + (freq_end - freq_start) * t / t_max
amp = amp_start + (amp_end - amp_start) * t / t_max

# Compute instantaneous phase for varying frequency
phase = 2 * np.pi * np.cumsum(freq) * dt
signal = amp * np.sin(phase)

# ------------------------
# Animation setup
# ------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 5)  # display 5 seconds at a time
ax.set_ylim(-120, 120)
ax.set_xlabel("Time (s)")
ax.set_ylabel("EEG amplitude (µV)")
ax.set_title("Simulated EEG transition from Slow-wave to REM")

line, = ax.plot([], [], lw=2, color='purple')

# For scrolling window effect
window = int(5/dt)

def update(frame):
    if frame < window:
        xdata = t[:frame]
        ydata = signal[:frame]
    else:
        xdata = t[frame-window:frame]
        ydata = signal[frame-window:frame]
    line.set_data(xdata, ydata)
    return line,

ani = FuncAnimation(fig, update, frames=len(t), interval=dt*1000)
plt.show()
