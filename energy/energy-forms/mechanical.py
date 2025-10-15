"""
Mechanical Energy - Kinetic and Potential Energy Calculations
Comprehensive implementation of mechanical energy concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class ParticleState:
    """Represents the state of a particle with position and velocity"""
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray, mass: float, time: float = 0.0):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.time = time

class MechanicalEnergy:
    """Class for calculating and analyzing mechanical energy"""
    
    def __init__(self, g: float = 9.81):
        self.g = g  # gravitational acceleration (m/s²)
    
    def kinetic_energy(self, mass: float, velocity: np.ndarray) -> float:
        """
        Calculate kinetic energy: KE = (1/2)mv²
        
        Args:
            mass: Mass in kg
            velocity: Velocity vector in m/s
            
        Returns:
            Kinetic energy in Joules
        """
        v_magnitude = np.linalg.norm(velocity)
        return 0.5 * mass * v_magnitude**2
    
    def gravitational_potential_energy(self, mass: float, height: float, reference_height: float = 0.0) -> float:
        """
        Calculate gravitational potential energy: PE = mgh
        
        Args:
            mass: Mass in kg
            height: Height in m
            reference_height: Reference level for zero PE
            
        Returns:
            Potential energy in Joules
        """
        return mass * self.g * (height - reference_height)
    
    def elastic_potential_energy(self, k: float, displacement: float) -> float:
        """
        Calculate elastic potential energy: PE = (1/2)kx²
        
        Args:
            k: Spring constant in N/m
            displacement: Displacement from equilibrium in m
            
        Returns:
            Elastic potential energy in Joules
        """
        return 0.5 * k * displacement**2
    
    def gravitational_potential_energy_general(self, m1: float, m2: float, r: float, G: float = 6.674e-11) -> float:
        """
        Calculate gravitational potential energy for two masses: PE = -Gm₁m₂/r
        
        Args:
            m1, m2: Masses in kg
            r: Separation distance in m
            G: Gravitational constant
            
        Returns:
            Gravitational potential energy in Joules
        """
        return -G * m1 * m2 / r
    
    def total_mechanical_energy(self, particle: ParticleState, potential_energy: float) -> float:
        """
        Calculate total mechanical energy: E = KE + PE
        
        Args:
            particle: ParticleState object
            potential_energy: Potential energy in J
            
        Returns:
            Total mechanical energy in Joules
        """
        ke = self.kinetic_energy(particle.mass, particle.velocity)
        return ke + potential_energy
    
    def escape_velocity(self, M: float, R: float, G: float = 6.674e-11) -> float:
        """
        Calculate escape velocity from a massive body: v_esc = √(2GM/R)
        
        Args:
            M: Mass of the body in kg
            R: Radius of the body in m
            G: Gravitational constant
            
        Returns:
            Escape velocity in m/s
        """
        return np.sqrt(2 * G * M / R)
    
    def pendulum_energy_analysis(self, L: float, theta_max: float, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze energy in a simple pendulum
        
        Args:
            L: Pendulum length in m
            theta_max: Maximum angle in radians
            num_points: Number of analysis points
            
        Returns:
            (angles, kinetic_energy, potential_energy, total_energy)
        """
        # Create angle array from -theta_max to +theta_max
        theta = np.linspace(-theta_max, theta_max, num_points)
        
        # For a pendulum, energy conservation gives us the speed at each angle
        # E_total = mgL(1 - cos(theta_max)) = constant
        E_total = self.g * L * (1 - np.cos(theta_max))  # per unit mass
        
        # At angle theta: mgL(1 - cos(theta)) + (1/2)v² = E_total
        # Therefore: v² = 2gL(cos(theta) - cos(theta_max))
        v_squared = 2 * self.g * L * (np.cos(theta) - np.cos(theta_max))
        v_squared = np.maximum(v_squared, 0)  # Ensure non-negative
        
        # Calculate energies (per unit mass)
        KE = 0.5 * v_squared
        PE = self.g * L * (1 - np.cos(theta))
        total_energy = KE + PE
        
        return theta, KE, PE, total_energy
    
    def projectile_motion_energy(self, v0: float, angle: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze energy in projectile motion
        
        Args:
            v0: Initial velocity in m/s
            angle: Launch angle in degrees
            num_points: Number of trajectory points
            
        Returns:
            (time, x, y, kinetic_energy, potential_energy)
        """
        angle_rad = np.radians(angle)
        
        # Calculate flight time
        t_flight = 2 * v0 * np.sin(angle_rad) / self.g
        t = np.linspace(0, t_flight, num_points)
        
        # Position components
        x = v0 * np.cos(angle_rad) * t
        y = v0 * np.sin(angle_rad) * t - 0.5 * self.g * t**2
        
        # Velocity components
        vx = v0 * np.cos(angle_rad) * np.ones_like(t)
        vy = v0 * np.sin(angle_rad) - self.g * t
        
        # Energies (per unit mass)
        KE = 0.5 * (vx**2 + vy**2)
        PE = self.g * y
        
        return t, x, y, KE, PE

def demonstrate_mechanical_energy():
    """Demonstrate various aspects of mechanical energy"""
    
    energy_calc = MechanicalEnergy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Mechanical Energy Analysis', fontsize=16)
    
    # 1. Pendulum Energy
    theta, KE_pendulum, PE_pendulum, E_total_pendulum = energy_calc.pendulum_energy_analysis(L=1.0, theta_max=np.pi/3)
    
    axes[0,0].plot(np.degrees(theta), KE_pendulum, 'r-', label='Kinetic Energy', linewidth=2)
    axes[0,0].plot(np.degrees(theta), PE_pendulum, 'b-', label='Potential Energy', linewidth=2)
    axes[0,0].plot(np.degrees(theta), E_total_pendulum, 'g--', label='Total Energy', linewidth=2)
    axes[0,0].set_title('Pendulum Energy vs. Angle')
    axes[0,0].set_xlabel('Angle (degrees)')
    axes[0,0].set_ylabel('Energy (J/kg)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Projectile Motion Energy
    t, x, y, KE_proj, PE_proj = energy_calc.projectile_motion_energy(v0=30, angle=45)
    
    axes[0,1].plot(t, KE_proj, 'r-', label='Kinetic Energy', linewidth=2)
    axes[0,1].plot(t, PE_proj, 'b-', label='Potential Energy', linewidth=2)
    axes[0,1].plot(t, KE_proj + PE_proj, 'g--', label='Total Energy', linewidth=2)
    axes[0,1].set_title('Projectile Motion Energy vs. Time')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Energy (J/kg)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Spring Potential Energy
    x_spring = np.linspace(-0.1, 0.1, 100)  # displacement in m
    k_values = [100, 200, 500]  # spring constants
    
    for k in k_values:
        PE_spring = energy_calc.elastic_potential_energy(k, x_spring)
        axes[1,0].plot(x_spring * 100, PE_spring, label=f'k = {k} N/m', linewidth=2)
    
    axes[1,0].set_title('Elastic Potential Energy')
    axes[1,0].set_xlabel('Displacement (cm)')
    axes[1,0].set_ylabel('Potential Energy (J)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Gravitational Potential Energy
    heights = np.linspace(0, 100, 100)  # height in m
    masses = [1, 5, 10]  # masses in kg
    
    for mass in masses:
        PE_grav = energy_calc.gravitational_potential_energy(mass, heights)
        axes[1,1].plot(heights, PE_grav, label=f'm = {mass} kg', linewidth=2)
    
    axes[1,1].set_title('Gravitational Potential Energy')
    axes[1,1].set_xlabel('Height (m)')
    axes[1,1].set_ylabel('Potential Energy (J)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some calculations
    print("Mechanical Energy Calculations")
    print("=" * 40)
    
    # Kinetic energy example
    mass = 2.0  # kg
    velocity = np.array([10, 5, 0])  # m/s
    ke = energy_calc.kinetic_energy(mass, velocity)
    print(f"Kinetic Energy: m={mass}kg, v={np.linalg.norm(velocity):.1f}m/s → KE={ke:.1f}J")
    
    # Potential energy examples
    height = 10  # m
    pe_grav = energy_calc.gravitational_potential_energy(mass, height)
    print(f"Gravitational PE: m={mass}kg, h={height}m → PE={pe_grav:.1f}J")
    
    k = 200  # N/m
    x = 0.05  # m
    pe_elastic = energy_calc.elastic_potential_energy(k, x)
    print(f"Elastic PE: k={k}N/m, x={x}m → PE={pe_elastic:.3f}J")
    
    # Escape velocity
    M_earth = 5.97e24  # kg
    R_earth = 6.37e6   # m
    v_esc = energy_calc.escape_velocity(M_earth, R_earth)
    print(f"Earth escape velocity: {v_esc/1000:.1f} km/s")

def interactive_energy_calculator():
    """Interactive calculator for mechanical energy"""
    
    print("Mechanical Energy Calculator")
    print("=" * 30)
    
    energy_calc = MechanicalEnergy()
    
    while True:
        print("\nChoose calculation type:")
        print("1. Kinetic Energy")
        print("2. Gravitational Potential Energy")
        print("3. Elastic Potential Energy")
        print("4. Total Mechanical Energy")
        print("5. Escape Velocity")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            mass = float(input("Enter mass (kg): "))
            print("Enter velocity components:")
            vx = float(input("  vx (m/s): "))
            vy = float(input("  vy (m/s): "))
            vz = float(input("  vz (m/s): "))
            
            velocity = np.array([vx, vy, vz])
            ke = energy_calc.kinetic_energy(mass, velocity)
            
            print(f"Kinetic Energy: {ke:.3f} J")
            print(f"Speed: {np.linalg.norm(velocity):.3f} m/s")
            
        elif choice == '2':
            mass = float(input("Enter mass (kg): "))
            height = float(input("Enter height (m): "))
            ref_height = float(input("Enter reference height (m, default 0): ") or "0")
            
            pe = energy_calc.gravitational_potential_energy(mass, height, ref_height)
            print(f"Gravitational Potential Energy: {pe:.3f} J")
            
        elif choice == '3':
            k = float(input("Enter spring constant (N/m): "))
            x = float(input("Enter displacement from equilibrium (m): "))
            
            pe = energy_calc.elastic_potential_energy(k, x)
            print(f"Elastic Potential Energy: {pe:.6f} J")
            
        elif choice == '4':
            mass = float(input("Enter mass (kg): "))
            print("Enter velocity components:")
            vx = float(input("  vx (m/s): "))
            vy = float(input("  vy (m/s): "))
            vz = float(input("  vz (m/s): "))
            velocity = np.array([vx, vy, vz])
            
            pe = float(input("Enter potential energy (J): "))
            
            particle = ParticleState(np.array([0, 0, 0]), velocity, mass)
            total_energy = energy_calc.total_mechanical_energy(particle, pe)
            ke = energy_calc.kinetic_energy(mass, velocity)
            
            print(f"Kinetic Energy: {ke:.3f} J")
            print(f"Potential Energy: {pe:.3f} J")
            print(f"Total Mechanical Energy: {total_energy:.3f} J")
            
        elif choice == '5':
            M = float(input("Enter mass of central body (kg): "))
            R = float(input("Enter radius of central body (m): "))
            
            v_esc = energy_calc.escape_velocity(M, R)
            print(f"Escape velocity: {v_esc:.1f} m/s ({v_esc/1000:.2f} km/s)")
            
        elif choice == '6':
            print("Thanks for using the mechanical energy calculator!")
            break
            
        else:
            print("Invalid choice. Please try again.")

class WorkEnergyPrinciple:
    """
    Demonstrates the Work-Energy Principle: W = F·d·cos(θ)
    Shows how work done by a force changes an object's kinetic energy
    """
    
    def __init__(self):
        self.energy_calc = MechanicalEnergy()
    
    def calculate_work(self, force: float, distance: float, angle_degrees: float) -> dict:
        """
        Calculate work done by a force: W = F·d·cos(θ)
        
        Args:
            force: Magnitude of force in Newtons
            distance: Distance moved in meters  
            angle_degrees: Angle between force and displacement in degrees
            
        Returns:
            Dictionary with work calculation details
        """
        angle_radians = np.radians(angle_degrees)
        work = force * distance * np.cos(angle_radians)
        
        return {
            'work': work,
            'force': force,
            'distance': distance, 
            'angle_degrees': angle_degrees,
            'angle_radians': angle_radians,
            'cos_angle': np.cos(angle_radians),
            'force_component': force * np.cos(angle_radians)
        }
    
    def work_energy_theorem_demo(self, mass: float, initial_velocity: float, 
                                force: float, distance: float, angle_degrees: float):
        """
        Demonstrate Work-Energy Theorem: W_net = ΔKE = KE_f - KE_i
        """
        print(f"\n🔧 Work-Energy Theorem Demonstration")
        print("=" * 50)
        
        # Calculate initial kinetic energy
        v_i = np.array([initial_velocity, 0])
        ke_initial = self.energy_calc.kinetic_energy(mass, v_i)
        
        # Calculate work done
        work_result = self.calculate_work(force, distance, angle_degrees)
        work_done = work_result['work']
        
        # Calculate final kinetic energy using work-energy theorem
        ke_final = ke_initial + work_done
        
        # Calculate final velocity
        v_final = np.sqrt(2 * ke_final / mass) if ke_final >= 0 else 0
        
        print(f"📊 Initial Conditions:")
        print(f"   Mass: {mass:.1f} kg")
        print(f"   Initial velocity: {initial_velocity:.1f} m/s")
        print(f"   Initial KE: {ke_initial:.1f} J")
        
        print(f"\n🔨 Applied Force:")
        print(f"   Force magnitude: {force:.1f} N") 
        print(f"   Distance moved: {distance:.1f} m")
        print(f"   Angle with displacement: {angle_degrees:.1f}°")
        print(f"   Force component along motion: {work_result['force_component']:.1f} N")
        
        print(f"\n⚡ Work-Energy Analysis:")
        print(f"   Work done: W = F·d·cos(θ) = {force:.1f} × {distance:.1f} × {work_result['cos_angle']:.3f} = {work_done:.1f} J")
        print(f"   Final KE: KE_f = KE_i + W = {ke_initial:.1f} + {work_done:.1f} = {ke_final:.1f} J")
        print(f"   Final velocity: {v_final:.1f} m/s")
        print(f"   Change in KE: ΔKE = {ke_final - ke_initial:.1f} J")
        
        return {
            'initial_ke': ke_initial,
            'final_ke': ke_final, 
            'work_done': work_done,
            'final_velocity': v_final
        }
    
    def plot_work_vs_angle(self, force: float = 100, distance: float = 10):
        """
        Plot how work varies with the angle between force and displacement
        """
        angles = np.linspace(0, 180, 181)
        work_values = []
        
        for angle in angles:
            work = self.calculate_work(force, distance, angle)['work']
            work_values.append(work)
        
        plt.figure(figsize=(12, 8))
        
        # Main plot
        plt.subplot(2, 2, 1)
        plt.plot(angles, work_values, 'b-', linewidth=2, label=f'F = {force} N, d = {distance} m')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Angle θ (degrees)')
        plt.ylabel('Work Done (J)')
        plt.title('Work vs Angle: W = F·d·cos(θ)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Highlight key angles
        key_angles = [0, 90, 180]
        key_works = [self.calculate_work(force, distance, angle)['work'] for angle in key_angles]
        plt.scatter(key_angles, key_works, color='red', s=100, zorder=5)
        
        for i, (angle, work) in enumerate(zip(key_angles, key_works)):
            plt.annotate(f'θ={angle}°\nW={work:.0f}J', 
                        xy=(angle, work), xytext=(10, 10),
                        textcoords='offset points', ha='left',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        # Force component plot
        plt.subplot(2, 2, 2)
        force_components = [force * np.cos(np.radians(angle)) for angle in angles]
        plt.plot(angles, force_components, 'r-', linewidth=2, label='F·cos(θ)')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Angle θ (degrees)')
        plt.ylabel('Force Component (N)')
        plt.title('Force Component Along Motion')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Vector diagrams for key angles
        for i, angle in enumerate([0, 60, 90, 120]):
            plt.subplot(2, 4, 5 + i)
            
            # Draw displacement vector
            plt.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.05, fc='blue', ec='blue', label='d')
            
            # Draw force vector
            angle_rad = np.radians(angle)
            fx = np.cos(angle_rad)
            fy = np.sin(angle_rad)
            plt.arrow(0, 0, fx, fy, head_width=0.05, head_length=0.05, fc='red', ec='red', label='F')
            
            # Draw force component
            plt.arrow(0, 0, fx, 0, head_width=0.03, head_length=0.03, fc='green', ec='green', 
                     linestyle='--', alpha=0.7, label='F·cos(θ)')
            
            plt.xlim(-0.2, 1.2)
            plt.ylim(-0.2, 1.2)
            plt.grid(True, alpha=0.3)
            plt.title(f'θ = {angle}°')
            plt.axis('equal')
            
            if i == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def interactive_work_calculator(self):
        """
        Interactive calculator for work-energy problems
        """
        print(f"\n🧮 Interactive Work-Energy Calculator")
        print("=" * 40)
        
        while True:
            print(f"\nChoose calculation type:")
            print("1. Calculate work done by a force")
            print("2. Work-Energy theorem demonstration")
            print("3. Plot work vs angle")
            print("4. Return to main menu")
            
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                try:
                    force = float(input("Enter force magnitude (N): "))
                    distance = float(input("Enter distance moved (m): "))
                    angle = float(input("Enter angle between force and displacement (degrees): "))
                    
                    result = self.calculate_work(force, distance, angle)
                    
                    print(f"\n📊 Work Calculation Results:")
                    print(f"   Force: {result['force']:.1f} N")
                    print(f"   Distance: {result['distance']:.1f} m") 
                    print(f"   Angle: {result['angle_degrees']:.1f}° = {result['angle_radians']:.3f} rad")
                    print(f"   cos({result['angle_degrees']:.1f}°) = {result['cos_angle']:.3f}")
                    print(f"   Force component along motion: {result['force_component']:.1f} N")
                    print(f"   Work done: W = F·d·cos(θ) = {result['work']:.1f} J")
                    
                    if result['work'] > 0:
                        print(f"   ✅ Positive work: Force adds energy to the object")
                    elif result['work'] < 0:
                        print(f"   ❌ Negative work: Force removes energy from the object")
                    else:
                        print(f"   ⚪ Zero work: Force perpendicular to motion")
                        
                except ValueError:
                    print("Please enter valid numbers.")
            
            elif choice == '2':
                try:
                    mass = float(input("Enter object mass (kg): "))
                    v_initial = float(input("Enter initial velocity (m/s): "))
                    force = float(input("Enter applied force (N): "))
                    distance = float(input("Enter distance moved (m): "))
                    angle = float(input("Enter angle between force and displacement (degrees): "))
                    
                    self.work_energy_theorem_demo(mass, v_initial, force, distance, angle)
                    
                except ValueError:
                    print("Please enter valid numbers.")
            
            elif choice == '3':
                try:
                    force = float(input("Enter force magnitude (N) [default: 100]: ") or "100")
                    distance = float(input("Enter distance (m) [default: 10]: ") or "10")
                    print("Generating work vs angle plot...")
                    self.plot_work_vs_angle(force, distance)
                except ValueError:
                    print("Please enter valid numbers.")
            
            elif choice == '4':
                break
            
            else:
                print("Invalid choice. Please try again.")

def demonstrate_work_energy_principle():
    """
    Comprehensive demonstration of the work-energy principle
    """
    work_demo = WorkEnergyPrinciple()
    
    print(f"\n🔬 Work-Energy Principle Demonstration")
    print("=" * 50)
    print("The Work-Energy Principle states:")
    print("W_net = ΔKE = KE_final - KE_initial")
    print("Where W = F·d·cos(θ) for a constant force")
    
    # Example 1: Force in direction of motion
    print(f"\n📖 Example 1: Force in Direction of Motion")
    work_demo.work_energy_theorem_demo(mass=5.0, initial_velocity=2.0, 
                                     force=20.0, distance=3.0, angle_degrees=0.0)
    
    # Example 2: Force at an angle
    print(f"\n📖 Example 2: Force at an Angle")
    work_demo.work_energy_theorem_demo(mass=2.0, initial_velocity=5.0,
                                     force=15.0, distance=4.0, angle_degrees=60.0)
    
    # Example 3: Force opposing motion
    print(f"\n📖 Example 3: Force Opposing Motion (Friction)")
    work_demo.work_energy_theorem_demo(mass=3.0, initial_velocity=8.0,
                                     force=-10.0, distance=2.0, angle_degrees=180.0)
    
    # Generate visualization
    print(f"\n📊 Generating Work vs Angle Visualization...")
    work_demo.plot_work_vs_angle(force=50, distance=5)

if __name__ == "__main__":
    print("🔧 Mechanical Energy & Work-Energy Principle Analysis")
    print("=" * 60)
    
    print("\nSelect demonstration:")
    print("1. Mechanical Energy Analysis")
    print("2. Work-Energy Principle") 
    print("3. Interactive Calculators")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        demonstrate_mechanical_energy()
    elif choice == '2':  
        demonstrate_work_energy_principle()
    elif choice == '3':
        print("\nChoose calculator:")
        print("1. Mechanical Energy Calculator")
        print("2. Work-Energy Calculator")
        calc_choice = input("Enter choice (1-2): ")
        
        if calc_choice == '1':
            interactive_energy_calculator()
        elif calc_choice == '2':
            work_demo = WorkEnergyPrinciple()
            work_demo.interactive_work_calculator()
    else:
        # Run both demonstrations
        demonstrate_mechanical_energy()
        demonstrate_work_energy_principle()