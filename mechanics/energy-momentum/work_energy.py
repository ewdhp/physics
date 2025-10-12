def work_energy(mass, force, distance):
    # Work = Force * distance
    work = force * distance
    # Kinetic energy = 1/2 m v^2
    # Assume starts from rest: work = delta KE
    v = (2 * work / mass) ** 0.5
    return work, v

def main():
    mass = float(input("Enter mass (kg): "))
    force = float(input("Enter constant force (N): "))
    distance = float(input("Enter distance (m): "))
    work, v = work_energy(mass, force, distance)
    print(f"Work done: {work:.2f} J")
    print(f"Final velocity (from work-energy): {v:.2f} m/s")

if __name__ == "__main__":
    main()
