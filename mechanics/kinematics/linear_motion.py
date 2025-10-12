import numpy as np
import matplotlib.pyplot as plt

def linear_motion(v0, a, t_max):
    t = np.linspace(0, t_max, 100)
    x = v0 * t + 0.5 * a * t**2
    v = v0 + a * t
    return t, x, v

def plot_linear_motion(t, x, v, v0, a):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, x)
    plt.title("Position vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(t, v)
    plt.title("Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.suptitle(f"Linear Motion: v0={v0} m/s, a={a} m/s²", y=1.05)
    plt.show()

def main():
    v0 = float(input("Enter initial velocity (m/s): "))
    a = float(input("Enter acceleration (m/s²): "))
    t_max = float(input("Enter total time (s): "))
    t, x, v = linear_motion(v0, a, t_max)
    plot_linear_motion(t, x, v, v0, a)

if __name__ == "__main__":
    main()
