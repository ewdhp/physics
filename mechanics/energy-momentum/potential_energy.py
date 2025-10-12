def gravitational_potential_energy(mass, height, g=9.81):
    return mass * g * height

def elastic_potential_energy(k, x):
    return 0.5 * k * x**2

def main():
    print("1. Gravitational Potential Energy")
    mass = float(input("Enter mass (kg): "))
    height = float(input("Enter height (m): "))
    gpe = gravitational_potential_energy(mass, height)
    print(f"Gravitational PE: {gpe:.2f} J")

    print("\n2. Elastic Potential Energy")
    k = float(input("Enter spring constant (N/m): "))
    x = float(input("Enter displacement (m): "))
    epe = elastic_potential_energy(k, x)
    print(f"Elastic PE: {epe:.2f} J")

if __name__ == "__main__":
    main()
