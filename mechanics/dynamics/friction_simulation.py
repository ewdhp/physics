import numpy as np
import matplotlib.pyplot as plt

def friction_simulation(mass, force, mu_k, t_max):
    g = 9.81
    friction = mu_k * mass * g
    net_force = force - friction
    a = net_force / mass if net_force > 0 else 0
    t = np.linspace(0, t_max, 100)
    v = a * t
    x = 0.5 * a * t**2
    return t, v, x, friction

def plot_friction(t, v, x, friction, mu_k):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, v)
    plt.title("Velocity vs Time (with Friction)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(t, x)
    plt.title("Position vs Time (with Friction)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.suptitle(f"Kinetic Friction: μ={mu_k}, F_friction={friction:.2f} N", y=1.05)
    plt.show()

def main():
    mass = float(input("Enter mass (kg): "))
    force = float(input("Enter applied force (N): "))
    mu_k = float(input("Enter coefficient of kinetic friction: "))
    t_max = float(input("Enter total time (s): "))
    t, v, x, friction = friction_simulation(mass, force, mu_k, t_max)
    plot_friction(t, v, x, friction, mu_k)

if __name__ == "__main__":
    main()
