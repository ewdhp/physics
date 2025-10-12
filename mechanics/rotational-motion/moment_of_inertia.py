def moment_of_inertia(masses, radii):
    # I = sum(m_i * r_i^2)
    return sum(m * r**2 for m, r in zip(masses, radii))

def main():
    n = int(input("Enter number of point masses: "))
    masses = []
    radii = []
    for i in range(n):
        m = float(input(f"Mass {i+1} (kg): "))
        r = float(input(f"Distance {i+1} from axis (m): "))
        masses.append(m)
        radii.append(r)
    I = moment_of_inertia(masses, radii)
    print(f"Moment of inertia: {I:.2f} kg·m^2")

if __name__ == "__main__":
    main()
