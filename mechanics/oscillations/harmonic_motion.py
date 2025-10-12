import numpy as np
import matplotlib.pyplot as plt

def shm(A, omega, phi, t):
    return A * np.cos(omega * t + phi)

def main():
    A = float(input("Enter amplitude (m): "))
    f = float(input("Enter frequency (Hz): "))
    phi = float(input("Enter phase (rad): "))
    t = np.linspace(0, 5, 500)
    omega = 2 * np.pi * f
    x = shm(A, omega, phi, t)
    plt.plot(t, x)
    plt.title("Simple Harmonic Motion")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
