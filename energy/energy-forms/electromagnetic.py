"""
Electromagnetic Energy - Electric and Magnetic Field Energy
Implementation of electromagnetic energy concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class ElectromagneticEnergy:
    """Class for electromagnetic energy calculations"""
    
    def __init__(self):
        self.epsilon_0 = 8.854e-12  # Permittivity of free space (F/m)
        self.mu_0 = 4*np.pi*1e-7   # Permeability of free space (H/m)
        self.c = 2.998e8           # Speed of light (m/s)
        self.e = 1.602e-19         # Elementary charge (C)
        self.h = 6.626e-34         # Planck constant (J·s)
        
    def electric_field_energy_density(self, E: float, epsilon_r: float = 1.0) -> float:
        """
        Calculate electric field energy density: u_E = (1/2)ε₀εᵣE²
        
        Args:
            E: Electric field magnitude (V/m)
            epsilon_r: Relative permittivity
            
        Returns:
            Energy density in J/m³
        """
        epsilon = self.epsilon_0 * epsilon_r
        return 0.5 * epsilon * E**2
    
    def magnetic_field_energy_density(self, B: float, mu_r: float = 1.0) -> float:
        """
        Calculate magnetic field energy density: u_B = (1/2μ₀μᵣ)B²
        
        Args:
            B: Magnetic field magnitude (T)
            mu_r: Relative permeability
            
        Returns:
            Energy density in J/m³
        """
        mu = self.mu_0 * mu_r
        return B**2 / (2 * mu)
    
    def electromagnetic_wave_energy_density(self, E_0: float, phase: float = 0) -> Tuple[float, float, float]:
        """
        Calculate energy density in electromagnetic wave
        
        Args:
            E_0: Electric field amplitude (V/m)
            phase: Phase of the wave (radians)
            
        Returns:
            (electric_energy_density, magnetic_energy_density, total_energy_density)
        """
        E = E_0 * np.cos(phase)
        B = E_0 / self.c * np.cos(phase)
        
        u_E = self.electric_field_energy_density(E)
        u_B = self.magnetic_field_energy_density(B)
        u_total = u_E + u_B
        
        return u_E, u_B, u_total
    
    def poynting_vector_magnitude(self, E: float, B: float) -> float:
        """
        Calculate Poynting vector magnitude: S = (1/μ₀)EB
        
        Args:
            E: Electric field magnitude (V/m)
            B: Magnetic field magnitude (T)
            
        Returns:
            Poynting vector magnitude (W/m²)
        """
        return E * B / self.mu_0
    
    def electromagnetic_wave_intensity(self, E_0: float) -> float:
        """
        Calculate time-averaged intensity of EM wave: ⟨S⟩ = (1/2μ₀c)E₀²
        
        Args:
            E_0: Electric field amplitude (V/m)
            
        Returns:
            Intensity in W/m²
        """
        return E_0**2 / (2 * self.mu_0 * self.c)
    
    def photon_energy(self, frequency: float = None, wavelength: float = None) -> float:
        """
        Calculate photon energy: E = hf = hc/λ
        
        Args:
            frequency: Frequency in Hz (provide either frequency or wavelength)
            wavelength: Wavelength in m
            
        Returns:
            Photon energy in Joules
        """
        if frequency is not None:
            return self.h * frequency
        elif wavelength is not None:
            return self.h * self.c / wavelength
        else:
            raise ValueError("Provide either frequency or wavelength")
    
    def capacitor_energy(self, C: float, V: float = None, Q: float = None) -> float:
        """
        Calculate energy stored in capacitor: U = (1/2)CV² = (1/2)QV = Q²/(2C)
        
        Args:
            C: Capacitance in F
            V: Voltage in V (optional)
            Q: Charge in C (optional)
            
        Returns:
            Energy in Joules
        """
        if V is not None:
            return 0.5 * C * V**2
        elif Q is not None:
            return Q**2 / (2 * C)
        else:
            raise ValueError("Provide either voltage V or charge Q")
    
    def inductor_energy(self, L: float, I: float) -> float:
        """
        Calculate energy stored in inductor: U = (1/2)LI²
        
        Args:
            L: Inductance in H
            I: Current in A
            
        Returns:
            Energy in Joules
        """
        return 0.5 * L * I**2
    
    def radiation_pressure(self, I: float, reflection_coefficient: float = 0) -> float:
        """
        Calculate radiation pressure: P = I(1 + r)/c
        
        Args:
            I: Intensity (W/m²)
            reflection_coefficient: 0 for absorption, 1 for perfect reflection
            
        Returns:
            Pressure in Pa
        """
        return I * (1 + reflection_coefficient) / self.c

def demonstrate_electromagnetic_energy():
    """Demonstrate electromagnetic energy concepts"""
    
    em_energy = ElectromagneticEnergy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Electromagnetic Energy Analysis', fontsize=16)
    
    # 1. Electric field energy density vs field strength
    E_range = np.logspace(2, 8, 100)  # Electric field range (V/m)
    
    materials = {'Vacuum': 1.0, 'Water': 81, 'Silicon': 12, 'Titanium dioxide': 100}
    
    for material, epsilon_r in materials.items():
        u_E = [em_energy.electric_field_energy_density(E, epsilon_r) for E in E_range]
        axes[0,0].loglog(E_range, u_E, label=material, linewidth=2)
    
    axes[0,0].set_title('Electric Field Energy Density')
    axes[0,0].set_xlabel('Electric Field (V/m)')
    axes[0,0].set_ylabel('Energy Density (J/m³)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Electromagnetic wave energy densities
    phase_range = np.linspace(0, 4*np.pi, 1000)
    E_0 = 1000  # V/m (amplitude)
    
    u_E_list, u_B_list, u_total_list = [], [], []
    for phase in phase_range:
        u_E, u_B, u_total = em_energy.electromagnetic_wave_energy_density(E_0, phase)
        u_E_list.append(u_E)
        u_B_list.append(u_B)
        u_total_list.append(u_total)
    
    axes[0,1].plot(phase_range/(2*np.pi), u_E_list, 'r-', label='Electric', linewidth=2)
    axes[0,1].plot(phase_range/(2*np.pi), u_B_list, 'b-', label='Magnetic', linewidth=2)
    axes[0,1].plot(phase_range/(2*np.pi), u_total_list, 'g--', label='Total', linewidth=2)
    axes[0,1].set_title(f'EM Wave Energy Density (E₀ = {E_0} V/m)')
    axes[0,1].set_xlabel('Phase (cycles)')
    axes[0,1].set_ylabel('Energy Density (J/m³)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Photon energy vs wavelength
    wavelength_range = np.logspace(-12, -3, 1000)  # Wavelength range (m)
    photon_energies = [em_energy.photon_energy(wavelength=wl) for wl in wavelength_range]
    photon_energies_eV = np.array(photon_energies) / em_energy.e  # Convert to eV
    
    # Mark different regions of electromagnetic spectrum
    regions = {
        'Gamma rays': (1e-12, 1e-10),
        'X-rays': (1e-11, 1e-8),
        'UV': (1e-8, 4e-7),
        'Visible': (4e-7, 7e-7),
        'IR': (7e-7, 1e-3),
        'Microwave': (1e-3, 1e-1),
        'Radio': (1e-1, 1e3)
    }
    
    axes[1,0].loglog(wavelength_range * 1e9, photon_energies_eV, 'k-', linewidth=2)
    axes[1,0].set_title('Photon Energy vs Wavelength')
    axes[1,0].set_xlabel('Wavelength (nm)')
    axes[1,0].set_ylabel('Photon Energy (eV)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add visible light region
    visible_range = np.array([400, 700]) * 1e-9  # m
    visible_energies = [em_energy.photon_energy(wavelength=wl)/em_energy.e for wl in visible_range]
    axes[1,0].axvspan(400, 700, alpha=0.3, color='yellow', label='Visible')
    axes[1,0].legend()
    
    # 4. EM wave intensity vs electric field amplitude
    E_0_range = np.linspace(100, 10000, 100)  # V/m
    intensities = [em_energy.electromagnetic_wave_intensity(E_0) for E_0 in E_0_range]
    
    axes[1,1].plot(E_0_range, np.array(intensities) * 1e-3, 'r-', linewidth=2)
    axes[1,1].set_title('EM Wave Intensity vs Field Amplitude')
    axes[1,1].set_xlabel('Electric Field Amplitude (V/m)')
    axes[1,1].set_ylabel('Intensity (kW/m²)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add some reference levels
    solar_constant = 1.361  # kW/m²
    axes[1,1].axhline(y=solar_constant, color='orange', linestyle='--', 
                     label=f'Solar constant ({solar_constant} kW/m²)')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print electromagnetic energy calculations
    print("Electromagnetic Energy Calculations")
    print("=" * 45)
    
    # Field energy densities
    E_field = 1000  # V/m
    B_field = E_field / em_energy.c  # T (for EM wave)
    
    u_E = em_energy.electric_field_energy_density(E_field)
    u_B = em_energy.magnetic_field_energy_density(B_field)
    print(f"Electric field energy density (E = {E_field} V/m): {u_E:.2e} J/m³")
    print(f"Magnetic field energy density (B = {B_field:.2e} T): {u_B:.2e} J/m³")
    
    # Photon energies
    wavelengths = {
        'Gamma ray': 1e-12,    # m
        'X-ray': 1e-10,       # m  
        'UV': 200e-9,         # m
        'Visible (green)': 550e-9,  # m
        'IR': 10e-6,          # m
        'Microwave': 1e-2,    # m
        'Radio': 1.0          # m
    }
    
    print(f"\nPhoton energies:")
    for name, wavelength in wavelengths.items():
        E_photon = em_energy.photon_energy(wavelength=wavelength)
        E_photon_eV = E_photon / em_energy.e
        print(f"  {name:15}: {E_photon_eV:.2e} eV")
    
    # Circuit energy storage
    C = 1e-6  # F (1 μF capacitor)
    V_cap = 100  # V
    U_cap = em_energy.capacitor_energy(C, V=V_cap)
    
    L = 1e-3  # H (1 mH inductor)
    I_ind = 1.0  # A
    U_ind = em_energy.inductor_energy(L, I_ind)
    
    print(f"\nCircuit energy storage:")
    print(f"  Capacitor ({C*1e6:.0f}μF at {V_cap}V): {U_cap*1e3:.1f} mJ")
    print(f"  Inductor ({L*1e3:.0f}mH at {I_ind}A): {U_ind*1e3:.1f} mJ")

def interactive_electromagnetic_calculator():
    """Interactive calculator for electromagnetic energy"""
    
    print("Electromagnetic Energy Calculator")
    print("=" * 35)
    
    em_energy = ElectromagneticEnergy()
    
    while True:
        print("\nChoose calculation type:")
        print("1. Electric Field Energy Density")
        print("2. Magnetic Field Energy Density")
        print("3. EM Wave Intensity")
        print("4. Photon Energy")
        print("5. Capacitor Energy")
        print("6. Inductor Energy")
        print("7. Radiation Pressure")
        print("8. Exit")
        
        choice = input("Enter your choice (1-8): ").strip()
        
        if choice == '1':
            E = float(input("Enter electric field (V/m): "))
            epsilon_r = float(input("Enter relative permittivity (default 1.0): ") or "1.0")
            
            u_E = em_energy.electric_field_energy_density(E, epsilon_r)
            print(f"Electric field energy density: {u_E:.2e} J/m³")
            
        elif choice == '2':
            B = float(input("Enter magnetic field (T): "))
            mu_r = float(input("Enter relative permeability (default 1.0): ") or "1.0")
            
            u_B = em_energy.magnetic_field_energy_density(B, mu_r)
            print(f"Magnetic field energy density: {u_B:.2e} J/m³")
            
        elif choice == '3':
            E_0 = float(input("Enter electric field amplitude (V/m): "))
            
            intensity = em_energy.electromagnetic_wave_intensity(E_0)
            print(f"EM wave intensity: {intensity:.2e} W/m² = {intensity*1e-3:.3f} kW/m²")
            
        elif choice == '4':
            print("Choose input method:")
            print("  1. Frequency (Hz)")
            print("  2. Wavelength (m)")
            method = input("Enter choice (1-2): ")
            
            if method == '1':
                freq = float(input("Enter frequency (Hz): "))
                E_photon = em_energy.photon_energy(frequency=freq)
            elif method == '2':
                wavelength = float(input("Enter wavelength (m): "))
                E_photon = em_energy.photon_energy(wavelength=wavelength)
            else:
                print("Invalid choice")
                continue
                
            E_photon_eV = E_photon / em_energy.e
            print(f"Photon energy: {E_photon:.2e} J = {E_photon_eV:.3f} eV")
            
        elif choice == '5':
            C = float(input("Enter capacitance (F): "))
            print("Provide either voltage or charge:")
            V_input = input("Voltage (V) - press enter to skip: ")
            
            if V_input:
                V = float(V_input)
                U = em_energy.capacitor_energy(C, V=V)
            else:
                Q = float(input("Charge (C): "))
                U = em_energy.capacitor_energy(C, Q=Q)
                
            print(f"Capacitor energy: {U:.2e} J = {U*1e3:.3f} mJ")
            
        elif choice == '6':
            L = float(input("Enter inductance (H): "))
            I = float(input("Enter current (A): "))
            
            U = em_energy.inductor_energy(L, I)
            print(f"Inductor energy: {U:.2e} J = {U*1e3:.3f} mJ")
            
        elif choice == '7':
            I = float(input("Enter intensity (W/m²): "))
            print("Surface type:")
            print("  1. Perfect absorber (r = 0)")
            print("  2. Perfect reflector (r = 1)")
            print("  3. Custom reflection coefficient")
            surface_type = input("Enter choice (1-3): ")
            
            if surface_type == '1':
                r = 0
            elif surface_type == '2':
                r = 1
            elif surface_type == '3':
                r = float(input("Enter reflection coefficient (0-1): "))
            else:
                print("Invalid choice")
                continue
                
            pressure = em_energy.radiation_pressure(I, r)
            print(f"Radiation pressure: {pressure:.2e} Pa = {pressure*1e6:.3f} μPa")
            
        elif choice == '8':
            print("Thanks for using the electromagnetic energy calculator!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Electromagnetic Energy Analysis")
    print("=" * 35)
    
    # Run demonstrations
    demonstrate_electromagnetic_energy()
    
    # Interactive calculator
    use_calculator = input("\nWould you like to use the interactive calculator? (y/n): ").lower()
    if use_calculator == 'y':
        interactive_electromagnetic_calculator()