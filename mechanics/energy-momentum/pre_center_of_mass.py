def center_of_mass(masses, positions):
    # 1D center of mass
    return sum(m * x for m, x in zip(masses, positions)) / sum(masses)

def main():
    n = int(input("Enter number of masses: "))
    masses = []
    positions = []
    for i in range(n):
        m = float(input(f"Mass {i+1} (kg): "))
        x = float(input(f"Position {i+1} (m): "))
        masses.append(m)
        positions.append(x)
    com = center_of_mass(masses, positions)
    print(f"Center of mass: {com:.2f} m")

if __name__ == "__main__":
    main()
