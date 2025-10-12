import numpy as np
import matplotlib.pyplot as plt

def damped_shm(A, gamma, omega0, phi, t):
    return A * np.exp(-gamma * t) * np.cos(omega0 * t + phi)

def main():
    A = float(input("Enter amplitude (m): "))
    f0 = float(input("Enter natural frequency (Hz): "))
    gamma = float(input("Enter damping coefficient (1/s): "))
    phi = float(input("Enter phase (rad): "))
    t = np.linspace(0, 10, 1000)
    omega0 = 2 * np.pi * f0
    x = damped_shm(A, gamma, omega0, phi, t)
    plt.plot(t, x)
    plt.title("Damped Oscillation")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
