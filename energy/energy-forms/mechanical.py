"""
Mechanical Energy - Kinetic and Potential Energy Calculations
Comprehensive implementation of mechanical energy concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class ParticleState:
    """Represents the state of a particle with position and velocity"""
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    time: float = 0.0

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

if __name__ == "__main__":
    print("Mechanical Energy Analysis")
    print("=" * 30)
    
    # Run demonstrations
    demonstrate_mechanical_energy()
    
    # Interactive calculator
    use_calculator = input("\nWould you like to use the interactive calculator? (y/n): ").lower()
    if use_calculator == 'y':
        interactive_energy_calculator()