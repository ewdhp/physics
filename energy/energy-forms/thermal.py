"""
Thermal Energy - Heat, Temperature, and Internal Energy
Implementation of thermodynamic energy concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class ThermalEnergy:
    """Class for thermal energy calculations and thermodynamic processes"""
    
    def __init__(self):
        self.k_B = 1.381e-23  # Boltzmann constant (J/K)
        self.R = 8.314        # Gas constant (J/(mol·K))
        self.N_A = 6.022e23   # Avogadro's number
        
    def thermal_energy_ideal_gas(self, n_moles: float, T: float, degrees_of_freedom: int = 3) -> float:
        """
        Calculate thermal energy of ideal gas: U = (f/2)nRT
        
        Args:
            n_moles: Number of moles
            T: Temperature in Kelvin
            degrees_of_freedom: Number of degrees of freedom (3 for monatomic, 5 for diatomic)
            
        Returns:
            Internal energy in Joules
        """
        return (degrees_of_freedom / 2) * n_moles * self.R * T
    
    def heat_capacity(self, n_moles: float, degrees_of_freedom: int = 3, process: str = 'constant_volume') -> float:
        """
        Calculate heat capacity for ideal gas
        
        Args:
            n_moles: Number of moles
            degrees_of_freedom: Degrees of freedom
            process: 'constant_volume' or 'constant_pressure'
            
        Returns:
            Heat capacity in J/K
        """
        C_v = (degrees_of_freedom / 2) * n_moles * self.R
        
        if process == 'constant_volume':
            return C_v
        elif process == 'constant_pressure':
            return C_v + n_moles * self.R  # C_p = C_v + nR
        else:
            raise ValueError("Process must be 'constant_volume' or 'constant_pressure'")
    
    def heat_transfer_conduction(self, k: float, A: float, delta_T: float, thickness: float) -> float:
        """
        Calculate heat transfer by conduction: q = kA(ΔT)/d
        
        Args:
            k: Thermal conductivity (W/(m·K))
            A: Cross-sectional area (m²)
            delta_T: Temperature difference (K)
            thickness: Material thickness (m)
            
        Returns:
            Heat transfer rate in Watts
        """
        return k * A * delta_T / thickness
    
    def heat_transfer_convection(self, h: float, A: float, delta_T: float) -> float:
        """
        Calculate heat transfer by convection: q = hA(ΔT)
        
        Args:
            h: Convective heat transfer coefficient (W/(m²·K))
            A: Surface area (m²)
            delta_T: Temperature difference (K)
            
        Returns:
            Heat transfer rate in Watts
        """
        return h * A * delta_T
    
    def heat_transfer_radiation(self, epsilon: float, sigma: float, A: float, T_hot: float, T_cold: float) -> float:
        """
        Calculate heat transfer by radiation: q = εσA(T₁⁴ - T₂⁴)
        
        Args:
            epsilon: Emissivity (0-1)
            sigma: Stefan-Boltzmann constant (5.67e-8 W/(m²·K⁴))
            A: Surface area (m²)
            T_hot: Hot temperature (K)
            T_cold: Cold temperature (K)
            
        Returns:
            Heat transfer rate in Watts
        """
        return epsilon * sigma * A * (T_hot**4 - T_cold**4)
    
    def entropy_change_isothermal(self, n_moles: float, V_initial: float, V_final: float) -> float:
        """
        Calculate entropy change for isothermal process: ΔS = nR ln(V₂/V₁)
        
        Args:
            n_moles: Number of moles
            V_initial: Initial volume (m³)
            V_final: Final volume (m³)
            
        Returns:
            Entropy change in J/K
        """
        return n_moles * self.R * np.log(V_final / V_initial)
    
    def entropy_change_temperature(self, mass: float, c_p: float, T_initial: float, T_final: float) -> float:
        """
        Calculate entropy change due to temperature change: ΔS = mc_p ln(T₂/T₁)
        
        Args:
            mass: Mass in kg
            c_p: Specific heat capacity (J/(kg·K))
            T_initial: Initial temperature (K)
            T_final: Final temperature (K)
            
        Returns:
            Entropy change in J/K
        """
        return mass * c_p * np.log(T_final / T_initial)
    
    def carnot_efficiency(self, T_hot: float, T_cold: float) -> float:
        """
        Calculate Carnot efficiency: η = 1 - T_cold/T_hot
        
        Args:
            T_hot: Hot reservoir temperature (K)
            T_cold: Cold reservoir temperature (K)
            
        Returns:
            Efficiency (0-1)
        """
        return 1 - T_cold / T_hot
    
    def maxwell_boltzmann_distribution(self, v: np.ndarray, mass: float, T: float) -> np.ndarray:
        """
        Calculate Maxwell-Boltzmann speed distribution
        
        Args:
            v: Speed array (m/s)
            mass: Particle mass (kg)
            T: Temperature (K)
            
        Returns:
            Probability density
        """
        prefactor = 4 * np.pi * (mass / (2 * np.pi * self.k_B * T))**(3/2)
        exponential = np.exp(-mass * v**2 / (2 * self.k_B * T))
        return prefactor * v**2 * exponential

def demonstrate_thermal_energy():
    """Demonstrate thermal energy concepts"""
    
    thermal = ThermalEnergy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Thermal Energy Analysis', fontsize=16)
    
    # 1. Internal energy vs temperature for different gases
    T_range = np.linspace(200, 800, 100)  # Temperature range (K)
    n_moles = 1.0  # 1 mole
    
    # Different types of gases
    gas_types = {'Monatomic (He, Ar)': 3, 'Diatomic (N₂, O₂)': 5, 'Linear molecules': 5, 'Nonlinear molecules': 6}
    
    for gas_type, dof in gas_types.items():
        U = thermal.thermal_energy_ideal_gas(n_moles, T_range, dof)
        axes[0,0].plot(T_range, U, label=gas_type, linewidth=2)
    
    axes[0,0].set_title('Internal Energy vs Temperature')
    axes[0,0].set_xlabel('Temperature (K)')
    axes[0,0].set_ylabel('Internal Energy (J)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Heat transfer mechanisms
    delta_T_range = np.linspace(10, 100, 50)  # Temperature difference (K)
    
    # Parameters for different heat transfer modes
    k = 50  # Thermal conductivity (W/(m·K)) - copper
    A = 0.01  # Area (m²)
    thickness = 0.001  # Thickness (m)
    h = 100  # Convection coefficient (W/(m²·K))
    epsilon = 0.8  # Emissivity
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    T_cold = 300  # Cold temperature (K)
    
    q_conduction = [thermal.heat_transfer_conduction(k, A, dT, thickness) for dT in delta_T_range]
    q_convection = [thermal.heat_transfer_convection(h, A, dT) for dT in delta_T_range]
    q_radiation = [thermal.heat_transfer_radiation(epsilon, sigma, A, T_cold + dT, T_cold) for dT in delta_T_range]
    
    axes[0,1].plot(delta_T_range, q_conduction, 'r-', label='Conduction', linewidth=2)
    axes[0,1].plot(delta_T_range, q_convection, 'b-', label='Convection', linewidth=2)
    axes[0,1].plot(delta_T_range, q_radiation, 'g-', label='Radiation', linewidth=2)
    axes[0,1].set_title('Heat Transfer Rates')
    axes[0,1].set_xlabel('Temperature Difference (K)')
    axes[0,1].set_ylabel('Heat Transfer Rate (W)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Maxwell-Boltzmann distribution
    v_range = np.linspace(0, 2000, 1000)  # Speed range (m/s)
    mass_N2 = 28.014e-3 / thermal.N_A  # Mass of N₂ molecule (kg)
    
    temperatures = [200, 300, 500, 800]  # Different temperatures (K)
    
    for T in temperatures:
        f_v = thermal.maxwell_boltzmann_distribution(v_range, mass_N2, T)
        axes[1,0].plot(v_range, f_v, label=f'T = {T} K', linewidth=2)
    
    axes[1,0].set_title('Maxwell-Boltzmann Distribution (N₂)')
    axes[1,0].set_xlabel('Speed (m/s)')
    axes[1,0].set_ylabel('Probability Density (s/m)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Carnot efficiency vs temperature ratio
    T_hot_range = np.linspace(400, 1000, 100)  # Hot temperature range (K)
    T_cold_values = [200, 250, 300, 350]  # Different cold temperatures (K)
    
    for T_cold in T_cold_values:
        efficiency = [thermal.carnot_efficiency(T_hot, T_cold) for T_hot in T_hot_range]
        axes[1,1].plot(T_hot_range, np.array(efficiency) * 100, label=f'T_cold = {T_cold} K', linewidth=2)
    
    axes[1,1].set_title('Carnot Efficiency')
    axes[1,1].set_xlabel('Hot Temperature (K)')
    axes[1,1].set_ylabel('Efficiency (%)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some thermal calculations
    print("Thermal Energy Calculations")
    print("=" * 40)
    
    # Internal energy
    n = 2.0  # moles
    T = 300  # K
    U_monatomic = thermal.thermal_energy_ideal_gas(n, T, 3)
    U_diatomic = thermal.thermal_energy_ideal_gas(n, T, 5)
    print(f"Internal energy at {T}K for {n} moles:")
    print(f"  Monatomic gas: {U_monatomic:.1f} J")
    print(f"  Diatomic gas: {U_diatomic:.1f} J")
    
    # Heat capacity
    C_v_mono = thermal.heat_capacity(n, 3, 'constant_volume')
    C_p_mono = thermal.heat_capacity(n, 3, 'constant_pressure')
    print(f"\nHeat capacities for {n} moles monatomic gas:")
    print(f"  C_v = {C_v_mono:.1f} J/K")
    print(f"  C_p = {C_p_mono:.1f} J/K")
    print(f"  γ = C_p/C_v = {C_p_mono/C_v_mono:.3f}")
    
    # Carnot efficiency
    T_hot, T_cold = 600, 300  # K
    eta_carnot = thermal.carnot_efficiency(T_hot, T_cold)
    print(f"\nCarnot engine ({T_hot}K → {T_cold}K):")
    print(f"  Maximum efficiency: {eta_carnot*100:.1f}%")

def interactive_thermal_calculator():
    """Interactive calculator for thermal energy"""
    
    print("Thermal Energy Calculator")
    print("=" * 30)
    
    thermal = ThermalEnergy()
    
    while True:
        print("\nChoose calculation type:")
        print("1. Internal Energy (Ideal Gas)")
        print("2. Heat Transfer - Conduction")
        print("3. Heat Transfer - Convection") 
        print("4. Heat Transfer - Radiation")
        print("5. Carnot Efficiency")
        print("6. Entropy Change")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            n_moles = float(input("Enter number of moles: "))
            T = float(input("Enter temperature (K): "))
            print("Select gas type:")
            print("  1. Monatomic (3 DOF)")
            print("  2. Diatomic (5 DOF)")
            print("  3. Nonlinear molecule (6 DOF)")
            gas_choice = input("Enter choice (1-3): ")
            
            dof_map = {'1': 3, '2': 5, '3': 6}
            dof = dof_map.get(gas_choice, 3)
            
            U = thermal.thermal_energy_ideal_gas(n_moles, T, dof)
            print(f"Internal Energy: {U:.1f} J")
            
        elif choice == '2':
            k = float(input("Enter thermal conductivity (W/(m·K)): "))
            A = float(input("Enter area (m²): "))
            delta_T = float(input("Enter temperature difference (K): "))
            thickness = float(input("Enter thickness (m): "))
            
            q = thermal.heat_transfer_conduction(k, A, delta_T, thickness)
            print(f"Heat transfer rate: {q:.1f} W")
            
        elif choice == '3':
            h = float(input("Enter convection coefficient (W/(m²·K)): "))
            A = float(input("Enter area (m²): "))
            delta_T = float(input("Enter temperature difference (K): "))
            
            q = thermal.heat_transfer_convection(h, A, delta_T)
            print(f"Heat transfer rate: {q:.1f} W")
            
        elif choice == '4':
            epsilon = float(input("Enter emissivity (0-1): "))
            A = float(input("Enter area (m²): "))
            T_hot = float(input("Enter hot temperature (K): "))
            T_cold = float(input("Enter cold temperature (K): "))
            sigma = 5.67e-8  # Stefan-Boltzmann constant
            
            q = thermal.heat_transfer_radiation(epsilon, sigma, A, T_hot, T_cold)
            print(f"Heat transfer rate: {q:.1f} W")
            
        elif choice == '5':
            T_hot = float(input("Enter hot reservoir temperature (K): "))
            T_cold = float(input("Enter cold reservoir temperature (K): "))
            
            eta = thermal.carnot_efficiency(T_hot, T_cold)
            print(f"Carnot efficiency: {eta*100:.1f}%")
            
        elif choice == '6':
            print("Select entropy change type:")
            print("  1. Isothermal volume change")
            print("  2. Temperature change at constant pressure")
            sub_choice = input("Enter choice (1-2): ")
            
            if sub_choice == '1':
                n_moles = float(input("Enter number of moles: "))
                V_i = float(input("Enter initial volume (m³): "))
                V_f = float(input("Enter final volume (m³): "))
                
                delta_S = thermal.entropy_change_isothermal(n_moles, V_i, V_f)
                print(f"Entropy change: {delta_S:.3f} J/K")
                
            elif sub_choice == '2':
                mass = float(input("Enter mass (kg): "))
                c_p = float(input("Enter specific heat capacity (J/(kg·K)): "))
                T_i = float(input("Enter initial temperature (K): "))
                T_f = float(input("Enter final temperature (K): "))
                
                delta_S = thermal.entropy_change_temperature(mass, c_p, T_i, T_f)
                print(f"Entropy change: {delta_S:.3f} J/K")
            
        elif choice == '7':
            print("Thanks for using the thermal energy calculator!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Thermal Energy Analysis")
    print("=" * 25)
    
    # Run demonstrations
    demonstrate_thermal_energy()
    
    # Interactive calculator
    use_calculator = input("\nWould you like to use the interactive calculator? (y/n): ").lower()
    if use_calculator == 'y':
        interactive_thermal_calculator()