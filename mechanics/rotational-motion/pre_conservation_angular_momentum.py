def conservation_angular_momentum(I1, w1, I2):
    # I1*w1 = I2*w2 => w2 = (I1*w1)/I2
    return (I1 * w1) / I2

def main():
    I1 = float(input("Enter initial moment of inertia (kg·m^2): "))
    w1 = float(input("Enter initial angular velocity (rad/s): "))
    I2 = float(input("Enter final moment of inertia (kg·m^2): "))
    w2 = conservation_angular_momentum(I1, w1, I2)
    print(f"Final angular velocity: {w2:.2f} rad/s")

if __name__ == "__main__":
    main()
