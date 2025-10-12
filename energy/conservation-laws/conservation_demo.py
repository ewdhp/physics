"""
Energy Conservation Laws - Python Implementations
Demonstrates energy conservation principles across different physical systems
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class EnergyConservation:
    """Class for demonstrating energy conservation in various systems"""
    
    def __init__(self):
        self.g = 9.81  # gravitational acceleration (m/s²)
    
    def mechanical_energy_pendulum(self, L: float, theta_0: float, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Demonstrate energy conservation in a simple pendulum
        
        Args:
            L: Length of pendulum (m)
            theta_0: Initial angle (radians)
            num_points: Number of calculation points
            
        Returns:
            (time, kinetic_energy, potential_energy, total_energy)
        """
        # Calculate period and create time array
        T = 2 * np.pi * np.sqrt(L / self.g)  # Small angle approximation
        t = np.linspace(0, 2*T, num_points)
        
        # For small angles: θ(t) = θ₀ cos(ωt), where ω = √(g/L)
        omega = np.sqrt(self.g / L)
        theta = theta_0 * np.cos(omega * t)
        theta_dot = -theta_0 * omega * np.sin(omega * t)
        
        # Calculate energies (per unit mass)
        # Kinetic energy: KE = (1/2) * L² * θ̇²
        KE = 0.5 * L**2 * theta_dot**2
        
        # Potential energy: PE = gL(1 - cos θ) ≈ (1/2) * gL * θ² for small angles
        PE = self.g * L * (1 - np.cos(theta))
        
        # Total energy
        E_total = KE + PE
        
        return t, KE, PE, E_total
    
    def projectile_with_air_resistance(self, v0: float, angle: float, drag_coeff: float, 
                                     mass: float, dt: float = 0.01) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Demonstrate energy loss due to air resistance in projectile motion
        
        Args:
            v0: Initial velocity (m/s)
            angle: Launch angle (degrees)
            drag_coeff: Drag coefficient (kg/s)
            mass: Projectile mass (kg)
            dt: Time step (s)
            
        Returns:
            (time, kinetic_energy, potential_energy, total_energy)
        """
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Initial conditions
        vx = v0 * np.cos(angle_rad)
        vy = v0 * np.sin(angle_rad)
        x, y = 0.0, 0.0
        
        # Lists to store results
        t_list = [0]
        KE_list = [0.5 * mass * v0**2]
        PE_list = [0]
        E_total_list = [KE_list[0]]
        
        t = 0
        while y >= 0:  # Continue until projectile hits ground
            # Calculate drag force
            v_magnitude = np.sqrt(vx**2 + vy**2)
            if v_magnitude > 0:
                drag_x = -drag_coeff * vx * v_magnitude / mass
                drag_y = -drag_coeff * vy * v_magnitude / mass
            else:
                drag_x = drag_y = 0
                
            # Update velocities (including gravity and drag)
            vx += drag_x * dt
            vy += (-self.g + drag_y) * dt
            
            # Update position
            x += vx * dt
            y += vy * dt
            
            # Update time
            t += dt
            
            if y < 0:  # Stop if below ground
                break
                
            # Calculate energies
            KE = 0.5 * mass * (vx**2 + vy**2)
            PE = mass * self.g * y
            E_total = KE + PE
            
            # Store results
            t_list.append(t)
            KE_list.append(KE)
            PE_list.append(PE)
            E_total_list.append(E_total)
        
        return t_list, KE_list, PE_list, E_total_list
    
    def spring_mass_system(self, k: float, mass: float, A: float, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Demonstrate energy conservation in a spring-mass system
        
        Args:
            k: Spring constant (N/m)
            mass: Mass (kg)
            A: Amplitude (m)
            num_points: Number of points
            
        Returns:
            (time, kinetic_energy, potential_energy, total_energy)
        """
        # Calculate angular frequency and period
        omega = np.sqrt(k / mass)
        T = 2 * np.pi / omega
        
        # Create time array
        t = np.linspace(0, 2*T, num_points)
        
        # Position and velocity for SHM: x = A cos(ωt), v = -Aω sin(ωt)
        x = A * np.cos(omega * t)
        v = -A * omega * np.sin(omega * t)
        
        # Calculate energies
        KE = 0.5 * mass * v**2
        PE = 0.5 * k * x**2
        E_total = KE + PE
        
        return t, KE, PE, E_total

def demonstrate_conservation():
    """Demonstrate energy conservation in various systems"""
    
    conservation = EnergyConservation()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Energy Conservation in Physical Systems', fontsize=16)
    
    # 1. Pendulum
    t, KE, PE, E_total = conservation.mechanical_energy_pendulum(L=1.0, theta_0=0.3)
    
    axes[0,0].plot(t, KE, 'r-', label='Kinetic Energy', linewidth=2)
    axes[0,0].plot(t, PE, 'b-', label='Potential Energy', linewidth=2)
    axes[0,0].plot(t, E_total, 'g--', label='Total Energy', linewidth=2)
    axes[0,0].set_title('Simple Pendulum (Conservative System)')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Energy (J/kg)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Spring-Mass System
    t, KE, PE, E_total = conservation.spring_mass_system(k=100, mass=2.0, A=0.1)
    
    axes[0,1].plot(t, KE, 'r-', label='Kinetic Energy', linewidth=2)
    axes[0,1].plot(t, PE, 'b-', label='Potential Energy', linewidth=2)
    axes[0,1].plot(t, E_total, 'g--', label='Total Energy', linewidth=2)
    axes[0,1].set_title('Spring-Mass System (Conservative)')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Energy (J)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Projectile without air resistance
    # Simple parabolic trajectory calculation
    v0, angle = 30, 45
    angle_rad = np.radians(angle)
    t_flight = 2 * v0 * np.sin(angle_rad) / conservation.g
    t_proj = np.linspace(0, t_flight, 100)
    
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad) - conservation.g * t_proj
    y = v0 * np.sin(angle_rad) * t_proj - 0.5 * conservation.g * t_proj**2
    
    mass = 1.0  # kg
    KE_no_drag = 0.5 * mass * (vx**2 + vy**2)
    PE_no_drag = mass * conservation.g * y
    E_total_no_drag = KE_no_drag + PE_no_drag
    
    axes[1,0].plot(t_proj, KE_no_drag, 'r-', label='Kinetic Energy', linewidth=2)
    axes[1,0].plot(t_proj, PE_no_drag, 'b-', label='Potential Energy', linewidth=2)
    axes[1,0].plot(t_proj, E_total_no_drag, 'g--', label='Total Energy', linewidth=2)
    axes[1,0].set_title('Projectile Motion (No Air Resistance)')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Energy (J)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Projectile with air resistance
    t_drag, KE_drag, PE_drag, E_total_drag = conservation.projectile_with_air_resistance(
        v0=30, angle=45, drag_coeff=0.1, mass=1.0
    )
    
    axes[1,1].plot(t_drag, KE_drag, 'r-', label='Kinetic Energy', linewidth=2)
    axes[1,1].plot(t_drag, PE_drag, 'b-', label='Potential Energy', linewidth=2)
    axes[1,1].plot(t_drag, E_total_drag, 'g--', label='Total Energy', linewidth=2)
    axes[1,1].set_title('Projectile Motion (With Air Resistance)')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Energy (J)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print energy conservation analysis
    print("Energy Conservation Analysis:")
    print("=" * 50)
    
    # Pendulum analysis
    E_var_pendulum = np.var(E_total)
    print(f"Pendulum - Energy variation: {E_var_pendulum:.2e} J²/kg²")
    print(f"Energy conservation: {'Excellent' if E_var_pendulum < 1e-10 else 'Good' if E_var_pendulum < 1e-6 else 'Poor'}")
    
    # Energy loss in projectile with drag
    E_initial_drag = E_total_drag[0]
    E_final_drag = E_total_drag[-1]
    energy_loss = E_initial_drag - E_final_drag
    print(f"\nProjectile with air resistance:")
    print(f"Initial energy: {E_initial_drag:.2f} J")
    print(f"Final energy: {E_final_drag:.2f} J")
    print(f"Energy lost to air resistance: {energy_loss:.2f} J ({100*energy_loss/E_initial_drag:.1f}%)")

def interactive_conservation_demo():
    """Interactive demonstration of energy conservation"""
    
    print("Energy Conservation Interactive Demo")
    print("=" * 40)
    
    conservation = EnergyConservation()
    
    while True:
        print("\nChoose a system to analyze:")
        print("1. Simple Pendulum")
        print("2. Spring-Mass System")
        print("3. Projectile Motion")
        print("4. Show all demonstrations")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            L = float(input("Enter pendulum length (m): "))
            theta_0 = float(input("Enter initial angle (degrees): "))
            theta_0_rad = np.radians(theta_0)
            
            t, KE, PE, E_total = conservation.mechanical_energy_pendulum(L, theta_0_rad)
            
            plt.figure(figsize=(10, 6))
            plt.plot(t, KE, 'r-', label='Kinetic Energy', linewidth=2)
            plt.plot(t, PE, 'b-', label='Potential Energy', linewidth=2)
            plt.plot(t, E_total, 'g--', label='Total Energy', linewidth=2)
            plt.title(f'Pendulum: L={L}m, θ₀={theta_0}°')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy (J/kg)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Energy variation: {np.var(E_total):.2e} J²/kg²")
            
        elif choice == '2':
            k = float(input("Enter spring constant (N/m): "))
            mass = float(input("Enter mass (kg): "))
            A = float(input("Enter amplitude (m): "))
            
            t, KE, PE, E_total = conservation.spring_mass_system(k, mass, A)
            
            plt.figure(figsize=(10, 6))
            plt.plot(t, KE, 'r-', label='Kinetic Energy', linewidth=2)
            plt.plot(t, PE, 'b-', label='Potential Energy', linewidth=2)
            plt.plot(t, E_total, 'g--', label='Total Energy', linewidth=2)
            plt.title(f'Spring-Mass: k={k}N/m, m={mass}kg, A={A}m')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy (J)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Total energy: {np.mean(E_total):.3f} J")
            print(f"Energy variation: {np.var(E_total):.2e} J²")
            
        elif choice == '3':
            v0 = float(input("Enter initial velocity (m/s): "))
            angle = float(input("Enter launch angle (degrees): "))
            drag_coeff = float(input("Enter drag coefficient (kg/s): "))
            mass = float(input("Enter projectile mass (kg): "))
            
            t_list, KE_list, PE_list, E_total_list = conservation.projectile_with_air_resistance(
                v0, angle, drag_coeff, mass
            )
            
            plt.figure(figsize=(10, 6))
            plt.plot(t_list, KE_list, 'r-', label='Kinetic Energy', linewidth=2)
            plt.plot(t_list, PE_list, 'b-', label='Potential Energy', linewidth=2)
            plt.plot(t_list, E_total_list, 'g--', label='Total Energy', linewidth=2)
            plt.title(f'Projectile: v₀={v0}m/s, θ={angle}°, drag={drag_coeff}')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy (J)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            energy_loss = E_total_list[0] - E_total_list[-1]
            print(f"Energy lost to air resistance: {energy_loss:.2f} J ({100*energy_loss/E_total_list[0]:.1f}%)")
            
        elif choice == '4':
            demonstrate_conservation()
            
        elif choice == '5':
            print("Thanks for exploring energy conservation!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Energy Conservation Laws Demonstration")
    print("=" * 45)
    
    # Run demonstrations
    demonstrate_conservation()
    
    # Interactive demo
    interactive_demo = input("\nWould you like to run the interactive demo? (y/n): ").lower()
    if interactive_demo == 'y':
        interactive_conservation_demo()