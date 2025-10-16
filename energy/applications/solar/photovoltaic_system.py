"""
Photovoltaic System Simulation

This script demonstrates the physics of photovoltaic (PV) solar panels,
including efficiency calculations, I-V characteristics, and power output
under various conditions. Models realistic ~20% efficiency systems.

Key Physics:
- Photovoltaic effect: photons → electron-hole pairs → current
- I-V characteristics of solar cells
- Maximum Power Point Tracking (MPPT)
- Temperature and irradiance effects
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# Set plotting style
try:
    plt.style.use('seaborn')
except OSError:
    plt.style.use('default')

# Simple dataclass replacement for Python 3.6 compatibility
class DataClass:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class PVCell:
    """Represents a single photovoltaic cell with physical parameters"""
    
    def __init__(self, area=0.0156, efficiency_stc=0.20, open_circuit_voltage=0.65, 
                 short_circuit_current=8.5, voltage_temp_coeff=-0.0025, 
                 current_temp_coeff=0.0005, ideality_factor=1.2, 
                 series_resistance=0.005, shunt_resistance=1000):
        # Standard Test Conditions (STC): 1000 W/m², 25°C, AM 1.5
        self.area = area  # m² (typical 125mm x 125mm cell)
        self.efficiency_stc = efficiency_stc  # 20% efficiency at STC
        self.open_circuit_voltage = open_circuit_voltage  # Voc in volts
        self.short_circuit_current = short_circuit_current  # Isc in amperes
        self.voltage_temp_coeff = voltage_temp_coeff  # V/°C
        self.current_temp_coeff = current_temp_coeff  # A/°C
        self.ideality_factor = ideality_factor  # Diode ideality factor
        self.series_resistance = series_resistance  # Ohms
        self.shunt_resistance = shunt_resistance  # Ohms
        
        # Calculate derived parameters
        self.thermal_voltage = 0.0259  # kT/q at 25°C in volts
        self.saturation_current = self.calculate_saturation_current()
        self.photocurrent_stc = self.short_circuit_current
    
    def calculate_saturation_current(self) -> float:
        """Calculate diode saturation current from STC parameters"""
        # Simplified model: I0 ≈ Isc / (exp(Voc/(n*Vt)) - 1)
        return self.short_circuit_current / (np.exp(self.open_circuit_voltage / 
                                                   (self.ideality_factor * self.thermal_voltage)) - 1)

class PVSystem:
    """Complete photovoltaic system with multiple cells/modules"""
    
    def __init__(self, cell: PVCell, cells_series: int = 60, cells_parallel: int = 1):
        self.cell = cell
        self.cells_series = cells_series
        self.cells_parallel = cells_parallel
        self.total_area = cell.area * cells_series * cells_parallel
        
    def photocurrent(self, irradiance: float, temperature: float = 25) -> float:
        """
        Calculate photocurrent based on irradiance and temperature
        
        Args:
            irradiance: Solar irradiance in W/m²
            temperature: Cell temperature in °C
            
        Returns:
            Photocurrent in amperes
        """
        # Temperature correction
        temp_diff = temperature - 25
        current_correction = 1 + self.cell.current_temp_coeff * temp_diff
        
        # Irradiance is linear with current
        return (self.cell.photocurrent_stc * irradiance / 1000 * 
                current_correction * self.cells_parallel)
    
    def diode_current(self, voltage: float, temperature: float = 25) -> float:
        """Calculate diode current based on voltage and temperature"""
        temp_kelvin = temperature + 273.15
        thermal_voltage = 8.617e-5 * temp_kelvin / 1.602e-19  # kT/q
        
        # Temperature scaling of saturation current
        i_sat = self.cell.saturation_current * ((temp_kelvin / 298.15) ** 3) * \
                np.exp(1.12 * 1.602e-19 / (1.381e-23) * (1/298.15 - 1/temp_kelvin))
        
        # Single diode equation for series-connected cells
        v_diode = voltage / self.cells_series
        return i_sat * self.cells_parallel * (np.exp(v_diode / 
                                                     (self.cell.ideality_factor * thermal_voltage)) - 1)
    
    def iv_characteristic(self, voltage: float, irradiance: float, temperature: float = 25) -> float:
        """
        Calculate current for given voltage using single-diode model
        
        Args:
            voltage: Terminal voltage in volts
            irradiance: Solar irradiance in W/m²
            temperature: Cell temperature in °C
            
        Returns:
            Current in amperes
        """
        # Photocurrent (light-generated current)
        i_ph = self.photocurrent(irradiance, temperature)
        
        # Diode current
        i_d = self.diode_current(voltage, temperature)
        
        # Shunt current
        i_sh = voltage / (self.cell.shunt_resistance * self.cells_series / self.cells_parallel)
        
        # Series resistance effect (simplified)
        # I = Iph - Id - Ish - V*Rs (approximation)
        series_loss = voltage * self.cell.series_resistance / (self.cells_series / self.cells_parallel)
        
        return i_ph - i_d - i_sh - series_loss
    
    def find_mpp(self, irradiance: float, temperature: float = 25) -> Tuple[float, float, float]:
        """
        Find Maximum Power Point using simple sweep method
        
        Returns:
            (voltage_mpp, current_mpp, power_mpp)
        """
        voltages = np.linspace(0, self.cells_series * self.cell.open_circuit_voltage * 0.9, 1000)
        currents = [max(0, self.iv_characteristic(v, irradiance, temperature)) for v in voltages]
        powers = voltages * currents
        
        max_idx = np.argmax(powers)
        return voltages[max_idx], currents[max_idx], powers[max_idx]
    
    def efficiency(self, irradiance: float, temperature: float = 25) -> float:
        """Calculate system efficiency under given conditions"""
        _, _, power_mpp = self.find_mpp(irradiance, temperature)
        incident_power = irradiance * self.total_area
        return power_mpp / incident_power if incident_power > 0 else 0

def plot_iv_curves():
    """Plot I-V and P-V curves under different conditions"""
    # Create PV system
    cell = PVCell()
    system = PVSystem(cell)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Different irradiance levels
    irradiances = [200, 400, 600, 800, 1000]  # W/m²
    voltages = np.linspace(0, system.cells_series * cell.open_circuit_voltage, 300)
    
    for irr in irradiances:
        currents = [max(0, system.iv_characteristic(v, irr)) for v in voltages]
        powers = [v * i for v, i in zip(voltages, currents)]
        
        ax1.plot(voltages, currents, label=f'{irr} W/m²', linewidth=2)
        ax2.plot(voltages, powers, label=f'{irr} W/m²', linewidth=2)
        
        # Mark MPP
        v_mpp, i_mpp, p_mpp = system.find_mpp(irr)
        ax1.plot(v_mpp, i_mpp, 'o', markersize=8, color='red')
        ax2.plot(v_mpp, p_mpp, 'o', markersize=8, color='red')
    
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current (A)')
    ax1.set_title('I-V Characteristics vs Irradiance\n(Temperature = 25°C)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('P-V Characteristics vs Irradiance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Different temperatures
    temperatures = [0, 25, 50, 75]  # °C
    irradiance_fixed = 1000  # W/m²
    
    for temp in temperatures:
        currents = [max(0, system.iv_characteristic(v, irradiance_fixed, temp)) for v in voltages]
        powers = [v * i for v, i in zip(voltages, currents)]
        
        ax3.plot(voltages, currents, label=f'{temp}°C', linewidth=2)
        ax4.plot(voltages, powers, label=f'{temp}°C', linewidth=2)
        
        # Mark MPP
        v_mpp, i_mpp, p_mpp = system.find_mpp(irradiance_fixed, temp)
        ax3.plot(v_mpp, i_mpp, 'o', markersize=8, color='red')
        ax4.plot(v_mpp, p_mpp, 'o', markersize=8, color='red')
    
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Current (A)')
    ax3.set_title('I-V Characteristics vs Temperature\n(Irradiance = 1000 W/m²)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Power (W)')
    ax4.set_title('P-V Characteristics vs Temperature')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def daily_energy_production():
    """Simulate daily energy production with varying conditions"""
    cell = PVCell()
    system = PVSystem(cell, cells_series=60, cells_parallel=4)  # Small residential system
    
    # Simulate one day (hourly data)
    hours = np.arange(0, 24, 0.5)
    
    # Solar irradiance model (simplified)
    sunrise = 6
    sunset = 18
    peak_irradiance = 1000  # W/m²
    
    irradiances = []
    for hour in hours:
        if sunrise <= hour <= sunset:
            # Sinusoidal model for solar irradiance
            solar_angle = np.pi * (hour - sunrise) / (sunset - sunrise)
            irr = peak_irradiance * np.sin(solar_angle) ** 1.5
        else:
            irr = 0
        irradiances.append(irr)
    
    # Temperature model (simplified)
    ambient_temp = 25 + 10 * np.sin(2 * np.pi * (hours - 6) / 24)  # °C
    cell_temp = ambient_temp + 0.03 * np.array(irradiances)  # Cell heating
    
    # Calculate power output
    powers = []
    efficiencies = []
    
    for irr, temp in zip(irradiances, cell_temp):
        if irr > 0:
            _, _, power = system.find_mpp(irr, temp)
            eff = system.efficiency(irr, temp)
        else:
            power = 0
            eff = 0
        powers.append(power)
        efficiencies.append(eff)
    
    # Calculate daily energy
    dt = 0.5  # hours
    daily_energy = np.sum(powers) * dt / 1000  # kWh
    
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Irradiance
    ax1.fill_between(hours, irradiances, alpha=0.7, color='orange')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Solar Irradiance (W/m²)')
    ax1.set_title('Daily Solar Irradiance Profile')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    
    # Temperature
    ax2.plot(hours, ambient_temp, label='Ambient', linewidth=2, color='blue')
    ax2.plot(hours, cell_temp, label='Cell', linewidth=2, color='red')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Temperature Profiles')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 24)
    
    # Power output
    ax3.fill_between(hours, powers, alpha=0.7, color='green')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Power Output (W)')
    ax3.set_title(f'PV Power Output\nDaily Energy: {daily_energy:.2f} kWh')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)
    
    # Efficiency
    ax4.plot(hours, np.array(efficiencies) * 100, linewidth=2, color='purple')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Efficiency (%)')
    ax4.set_title('PV System Efficiency Throughout Day')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 24)
    ax4.set_ylim(0, 25)
    
    plt.tight_layout()
    plt.show()
    
    return daily_energy

def pv_system_comparison():
    """Compare different PV technologies and configurations"""
    
    # Different cell technologies
    technologies = {
        'Monocrystalline Si': {'efficiency': 0.22, 'temp_coeff': -0.0035},
        'Polycrystalline Si': {'efficiency': 0.18, 'temp_coeff': -0.0040},
        'Thin Film CdTe': {'efficiency': 0.16, 'temp_coeff': -0.0025},
        'Perovskite (emerging)': {'efficiency': 0.25, 'temp_coeff': -0.0020}
    }
    
    temperatures = np.linspace(0, 60, 50)
    irradiance = 1000  # W/m²
    
    plt.figure(figsize=(12, 8))
    
    for tech_name, params in technologies.items():
        # Create cell with specific parameters
        cell = PVCell(efficiency_stc=params['efficiency'],
                     voltage_temp_coeff=params['temp_coeff'])
        system = PVSystem(cell)
        
        efficiencies = []
        for temp in temperatures:
            eff = system.efficiency(irradiance, temp)
            efficiencies.append(eff * 100)
        
        plt.plot(temperatures, efficiencies, linewidth=3, 
                label=f'{tech_name} (STC: {params["efficiency"]*100:.1f}%)')
    
    plt.xlabel('Cell Temperature (°C)')
    plt.ylabel('Efficiency (%)')
    plt.title('PV Technology Comparison: Efficiency vs Temperature\n(Irradiance = 1000 W/m²)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def main():
    """Main demonstration of photovoltaic systems"""
    
    print("=== Photovoltaic System Simulation ===\n")
    
    # Create a typical residential PV system
    cell = PVCell()
    system = PVSystem(cell, cells_series=60, cells_parallel=10)  # ~3kW system
    
    print(f"System Configuration:")
    print(f"- Cell area: {cell.area:.4f} m²")
    print(f"- Cells in series: {system.cells_series}")
    print(f"- Parallel strings: {system.cells_parallel}")
    print(f"- Total area: {system.total_area:.2f} m²")
    print(f"- STC efficiency: {cell.efficiency_stc*100:.1f}%")
    
    # Standard Test Conditions performance
    v_mpp, i_mpp, p_mpp = system.find_mpp(1000, 25)
    efficiency_stc = system.efficiency(1000, 25)
    
    print(f"\nStandard Test Conditions (1000 W/m², 25°C):")
    print(f"- MPP Voltage: {v_mpp:.1f} V")
    print(f"- MPP Current: {i_mpp:.1f} A")
    print(f"- MPP Power: {p_mpp:.1f} W ({p_mpp/1000:.2f} kW)")
    print(f"- System efficiency: {efficiency_stc*100:.1f}%")
    
    # Different operating conditions
    conditions = [
        (800, 35, "Partly cloudy, warm"),
        (600, 20, "Cloudy, cool"),
        (1200, 45, "Very sunny, hot")
    ]
    
    print(f"\nPerformance under different conditions:")
    for irr, temp, desc in conditions:
        v_mpp, i_mpp, p_mpp = system.find_mpp(irr, temp)
        eff = system.efficiency(irr, temp)
        print(f"- {desc} ({irr} W/m², {temp}°C):")
        print(f"  Power: {p_mpp:.0f} W, Efficiency: {eff*100:.1f}%")
    
    print(f"\nGenerating visualizations...")
    
    # Generate plots
    plot_iv_curves()
    daily_energy = daily_energy_production()
    pv_system_comparison()
    
    print(f"\nKey Insights:")
    print(f"- Modern PV systems achieve ~20% efficiency under ideal conditions")
    print(f"- Temperature significantly affects voltage (and thus power)")
    print(f"- Irradiance linearly affects current (and thus power)")
    print(f"- Daily energy production depends on both weather and sun angle")
    print(f"- Technology choice affects temperature sensitivity")

if __name__ == "__main__":
    main()