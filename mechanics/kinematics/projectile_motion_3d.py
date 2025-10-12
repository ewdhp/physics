import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def projectile_motion_3d(v0, theta_deg, phi_deg, g=9.81):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    t_flight = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_flight, 100)
    x = v0 * np.cos(theta) * np.cos(phi) * t
    y = v0 * np.cos(theta) * np.sin(phi) * t
    z = v0 * np.sin(theta) * t - 0.5 * g * t**2
    return x, y, z

def plot_trajectory_3d(x, y, z, v0, theta_deg, phi_deg):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"3D Projectile: v0={v0} m/s, θ={theta_deg}°, φ={phi_deg}°")
    plt.show()

def main():
    v0 = float(input("Enter initial speed (m/s): "))
    theta = float(input("Enter elevation angle θ (degrees): "))
    phi = float(input("Enter azimuth angle φ (degrees): "))
    x, y, z = projectile_motion_3d(v0, theta, phi)
    plot_trajectory_3d(x, y, z, v0, theta, phi)

if __name__ == "__main__":
    main()
