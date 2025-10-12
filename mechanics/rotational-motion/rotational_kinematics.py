def rotational_kinematics(theta0, omega0, alpha, t):
    # theta = theta0 + omega0*t + 0.5*alpha*t^2
    theta = theta0 + omega0 * t + 0.5 * alpha * t**2
    # omega = omega0 + alpha*t
    omega = omega0 + alpha * t
    return theta, omega

def main():
    theta0 = float(input("Enter initial angle (rad): "))
    omega0 = float(input("Enter initial angular velocity (rad/s): "))
    alpha = float(input("Enter angular acceleration (rad/s^2): "))
    t = float(input("Enter time (s): "))
    theta, omega = rotational_kinematics(theta0, omega0, alpha, t)
    print(f"Final angle: {theta:.2f} rad")
    print(f"Final angular velocity: {omega:.2f} rad/s")

if __name__ == "__main__":
    main()
