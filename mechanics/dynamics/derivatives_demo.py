import numpy as np
import matplotlib.pyplot as plt

# Demonstrate position, velocity, and acceleration using derivatives

def position_function(t):
    # Example: x(t) = 3t^2 + 2t + 1
    return 3 * t**2 + 2 * t + 1

def velocity_function(t):
    # v(t) = dx/dt = 6t + 2
    return 6 * t + 2

def acceleration_function(t):
    # a(t) = dv/dt = 6
    return np.full_like(t, 6)

def main():
    t = np.linspace(0, 5, 100)
    x = position_function(t)
    v = velocity_function(t)
    a = acceleration_function(t)

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.title('Position vs Time')
    plt.ylabel('Position (m)')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, v)
    plt.title('Velocity vs Time (dx/dt)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, a)
    plt.title('Acceleration vs Time (dv/dt)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    print("This demo shows how velocity is the derivative of position, and acceleration is the derivative of velocity.")
    print("Example: If x(t) = 3t² + 2t + 1, then v(t) = dx/dt = 6t + 2, and a(t) = dv/dt = 6.")

if __name__ == "__main__":
    main()
