def elastic_collision(m1, v1, m2, v2):
    # 1D elastic collision
    v1f = (v1 * (m1 - m2) + 2 * m2 * v2) / (m1 + m2)
    v2f = (v2 * (m2 - m1) + 2 * m1 * v1) / (m1 + m2)
    return v1f, v2f

def inelastic_collision(m1, v1, m2, v2):
    # 1D perfectly inelastic collision (stick together)
    vf = (m1 * v1 + m2 * v2) / (m1 + m2)
    return vf

def main():
    print("1. Elastic Collision")
    m1 = float(input("Enter mass 1 (kg): "))
    v1 = float(input("Enter velocity 1 (m/s): "))
    m2 = float(input("Enter mass 2 (kg): "))
    v2 = float(input("Enter velocity 2 (m/s): "))
    v1f, v2f = elastic_collision(m1, v1, m2, v2)
    print(f"Final velocities: v1' = {v1f:.2f} m/s, v2' = {v2f:.2f} m/s")

    print("\n2. Inelastic Collision")
    vf = inelastic_collision(m1, v1, m2, v2)
    print(f"Final velocity (stuck together): {vf:.2f} m/s")

if __name__ == "__main__":
    main()
