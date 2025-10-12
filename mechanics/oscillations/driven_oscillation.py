import numpy as np
import matplotlib.pyplot as plt

def driven_shm(A, omega0, gamma, F0, omega_drive, t):
    # Particular solution for driven, damped oscillator (steady-state)
    denom = np.sqrt((omega0**2 - omega_drive**2)**2 + (2*gamma*omega_drive)**2)
    x = (F0 * np.cos(omega_drive * t)) / denom
    return x

def main():
    F0 = float(input("Enter driving force amplitude (N): "))
    f0 = float(input("Enter natural frequency (Hz): "))
    gamma = float(input("Enter damping coefficient (1/s): "))
    f_drive = float(input("Enter driving frequency (Hz): "))
    t = np.linspace(0, 20, 2000)
    omega0 = 2 * np.pi * f0
    omega_drive = 2 * np.pi * f_drive
    x = driven_shm(1, omega0, gamma, F0, omega_drive, t)
    plt.plot(t, x)
    plt.title("Driven Oscillation (Steady-State)")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
