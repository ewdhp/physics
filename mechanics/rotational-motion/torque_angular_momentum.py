def torque(r, F, phi_deg):
    import math
    phi = math.radians(phi_deg)
    return r * F * math.sin(phi)

def angular_momentum(I, omega):
    return I * omega

def main():
    r = float(input("Enter lever arm (m): "))
    F = float(input("Enter force (N): "))
    phi = float(input("Enter angle between r and F (degrees): "))
    tau = torque(r, F, phi)
    print(f"Torque: {tau:.2f} N·m")

    I = float(input("Enter moment of inertia (kg·m^2): "))
    omega = float(input("Enter angular velocity (rad/s): "))
    L = angular_momentum(I, omega)
    print(f"Angular momentum: {L:.2f} kg·m^2/s")

if __name__ == "__main__":
    main()
