def atwood_machine(m1, m2, g=9.81):
    """
    Returns acceleration and tension in an ideal Atwood machine.
    """
    a = g * (m1 - m2) / (m1 + m2)
    T = 2 * m1 * m2 * g / (m1 + m2)
    return a, T

def main():
    m1 = float(input("Enter mass 1 (kg): "))
    m2 = float(input("Enter mass 2 (kg): "))
    a, T = atwood_machine(m1, m2)
    print(f"Acceleration: {a:.2f} m/s²")
    print(f"Tension: {T:.2f} N")

if __name__ == "__main__":
    main()
