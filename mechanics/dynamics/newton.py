import numpy as np
import matplotlib.pyplot as plt

def second_law_simulation(mass, force, t_max):
    t = np.linspace(0, t_max, 100)
    a = force / mass
    v = a * t
    x = 0.5 * a * t**2
    return t, a, v, x

def plot_second_law(t, a, v, x, mass, force):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, v)
    plt.title("Velocity vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(t, x)
    plt.title("Position vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.suptitle(f"Newton's 2nd Law: m={mass} kg, F={force} N", y=1.05)
    plt.show()

def third_law_collision(m1, m2, v1_initial, v2_initial):
    """
    Newton's Third Law: For every action, there is an equal and opposite reaction
    Example: Two objects colliding (elastic collision)
    
    Args:
        m1, m2: masses of objects (kg)
        v1_initial, v2_initial: initial velocities (m/s)
    
    Returns:
        Final velocities after collision
    """
    # Conservation of momentum and energy (elastic collision)
    # F12 = -F21 (Third Law: forces are equal and opposite)
    
    v1_final = ((m1 - m2) * v1_initial + 2 * m2 * v2_initial) / (m1 + m2)
    v2_final = ((m2 - m1) * v2_initial + 2 * m1 * v1_initial) / (m1 + m2)
    
    return v1_final, v2_final

def rocket_propulsion_example(m_rocket_initial, m_fuel, v_exhaust, dt):
    """
    Newton's Third Law: Rocket propulsion example
    Rocket pushes exhaust down, exhaust pushes rocket up
    
    Args:
        m_rocket_initial: initial rocket mass (kg)
        m_fuel: fuel mass burned per second (kg/s)
        v_exhaust: exhaust velocity relative to rocket (m/s)
        dt: time step (s)
    
    Returns:
        rocket velocity change
    """
    # Action: Rocket exerts force on exhaust (downward)
    # Reaction: Exhaust exerts equal force on rocket (upward)
    
    dm = m_fuel * dt  # mass of fuel burned
    m_rocket_final = m_rocket_initial - dm
    
    # Momentum conservation (Newton's 3rd law in action)
    dv_rocket = (v_exhaust * dm) / m_rocket_final
    
    return dv_rocket, m_rocket_final

def demonstrate_third_law():
    """
    Demonstrate Newton's Third Law with two practical examples
    """
    print("=== Newton's Third Law Demonstrations ===")
    print("Law: For every action, there is an equal and opposite reaction\n")
    
    # Example 1: Collision
    print("Example 1: Collision of two cars")
    print("Car A (1500 kg) at 20 m/s hits Car B (1000 kg) at -10 m/s")
    
    m1, m2 = 1500, 1000  # masses in kg
    v1_i, v2_i = 20, -10  # initial velocities in m/s
    
    v1_f, v2_f = third_law_collision(m1, m2, v1_i, v2_i)
    
    print(f"Before collision: Car A = {v1_i} m/s, Car B = {v2_i} m/s")
    print(f"After collision:  Car A = {v1_f:.2f} m/s, Car B = {v2_f:.2f} m/s")
    
    # Calculate forces (equal and opposite)
    impulse_A = m1 * (v1_f - v1_i)
    impulse_B = m2 * (v2_f - v2_i)
    print(f"Impulse on Car A: {impulse_A:.2f} kg⋅m/s")
    print(f"Impulse on Car B: {impulse_B:.2f} kg⋅m/s")
    print(f"Forces are equal and opposite: {abs(impulse_A) == abs(impulse_B)}\n")
    
    # Example 2: Rocket propulsion
    print("Example 2: Rocket propulsion")
    print("Rocket burns fuel to create exhaust, exhaust pushes rocket forward")
    
    m_rocket = 10000  # kg
    fuel_rate = 100   # kg/s
    v_exhaust = 3000  # m/s (relative to rocket)
    time_burn = 5     # seconds
    
    total_dv = 0
    current_mass = m_rocket
    
    print(f"Initial rocket mass: {m_rocket} kg")
    print(f"Fuel burn rate: {fuel_rate} kg/s")
    print(f"Exhaust velocity: {v_exhaust} m/s")
    
    for t in range(time_burn):
        dv, current_mass = rocket_propulsion_example(current_mass, fuel_rate, v_exhaust, 1.0)
        total_dv += dv
        print(f"Time {t+1}s: ΔV = +{dv:.2f} m/s, Mass = {current_mass:.0f} kg")
    
    print(f"Total velocity change: {total_dv:.2f} m/s")
    print("Action: Rocket pushes exhaust downward")
    print("Reaction: Exhaust pushes rocket upward (Newton's 3rd Law)")

def main():
    print("Newton's Laws Demonstration")
    print("1. Second Law Simulation")
    print("2. Third Law Examples")
    choice = input("Choose (1 or 2): ")
    
    if choice == "1":
        mass = float(input("Enter mass (kg): "))
        force = float(input("Enter net force (N): "))
        t_max = float(input("Enter total time (s): "))
        t, a, v, x = second_law_simulation(mass, force, t_max)
        plot_second_law(t, a, v, x, mass, force)
    elif choice == "2":
        demonstrate_third_law()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
