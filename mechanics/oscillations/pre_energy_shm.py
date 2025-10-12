import numpy as np
import matplotlib.pyplot as plt

def energy_shm(A, k, m, t):
    omega = np.sqrt(k/m)
    x = A * np.cos(omega * t)
    v = -A * omega * np.sin(omega * t)
    KE = 0.5 * m * v**2
    PE = 0.5 * k * x**2
    return KE, PE

def main():
    A = float(input("Enter amplitude (m): "))
    k = float(input("Enter spring constant (N/m): "))
    m = float(input("Enter mass (kg): "))
    t = np.linspace(0, 10, 500)
    KE, PE = energy_shm(A, k, m, t)
    plt.plot(t, KE, label='Kinetic Energy')
    plt.plot(t, PE, label='Potential Energy')
    plt.title('Energy in Simple Harmonic Motion')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
