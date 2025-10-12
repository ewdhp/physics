import numpy as np
import matplotlib.pyplot as plt

def projectile_motion(v0, angle_deg, g=9.81):
    """
    Simulate projectile motion for a given initial speed and launch angle.
    Args:
        v0 (float): Initial speed (m/s)
        angle_deg (float): Launch angle (degrees)
        g (float): Acceleration due to gravity (m/s^2)
    Returns:
        t (ndarray): Time points
        x (ndarray): X positions
        y (ndarray): Y positions
    """
    angle_rad = np.radians(angle_deg)
    t_flight = 2 * v0 * np.sin(angle_rad) / g
    t = np.linspace(0, t_flight, num=100)
    x = v0 * np.cos(angle_rad) * t
    y = v0 * np.sin(angle_rad) * t - 0.5 * g * t**2
    return t, x, y

def plot_trajectory(x, y, v0, angle_deg):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.title(f"Projectile Motion: v0={v0} m/s, angle={angle_deg}°")
    plt.xlabel("Horizontal Distance (m)")
    plt.ylabel("Vertical Distance (m)")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()

def main():
    v0 = float(input("Enter initial speed (m/s): "))
    angle = float(input("Enter launch angle (degrees): "))
    t, x, y = projectile_motion(v0, angle)
    plot_trajectory(x, y, v0, angle)

if __name__ == "__main__":
    main()
