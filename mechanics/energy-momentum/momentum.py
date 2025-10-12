def momentum(mass, velocity):
    return mass * velocity

def impulse(force, time):
    return force * time

def main():
    mass = float(input("Enter mass (kg): "))
    velocity = float(input("Enter velocity (m/s): "))
    p = momentum(mass, velocity)
    print(f"Momentum: {p:.2f} kg·m/s")

    force = float(input("Enter force (N): "))
    time = float(input("Enter time (s): "))
    J = impulse(force, time)
    print(f"Impulse: {J:.2f} N·s")

if __name__ == "__main__":
    main()
