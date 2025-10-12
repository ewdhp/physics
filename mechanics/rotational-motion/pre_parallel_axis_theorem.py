def parallel_axis_theorem(I_cm, m, d):
    # I = I_cm + m*d^2
    return I_cm + m * d**2

def main():
    I_cm = float(input("Enter moment of inertia about center of mass (kg·m^2): "))
    m = float(input("Enter mass (kg): "))
    d = float(input("Enter distance from new axis (m): "))
    I = parallel_axis_theorem(I_cm, m, d)
    print(f"Moment of inertia about new axis: {I:.2f} kg·m^2")

if __name__ == "__main__":
    main()
