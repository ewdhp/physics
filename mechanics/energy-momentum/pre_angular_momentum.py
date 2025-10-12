def angular_momentum(m, v, r):
    # L = r x p = m * v * r (for perpendicular motion)
    return m * v * r

def main():
    m = float(input("Enter mass (kg): "))
    v = float(input("Enter velocity (m/s): "))
    r = float(input("Enter radius (m): "))
    L = angular_momentum(m, v, r)
    print(f"Angular momentum: {L:.2f} kg·m^2/s")

if __name__ == "__main__":
    main()
