"""
Concentrated Solar Power (CSP) System Simulation

This script demonstrates the physics of concentrated solar power systems,
modeling the energy conversion chain: light → thermal → mechanical → electrical
with realistic ~35% overall efficiency.

Key Physics:
- Solar concentration using mirrors/lenses
- Heat transfer and thermal energy storage
- Thermodynamic cycles (Rankine, Brayton)
- Energy conversion efficiency chain
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import math

# Set plotting style
try:
    plt.style.use('seaborn')
except OSError:
    plt.style.use('default')

class SolarConcentrator:
    """Models solar concentration system (parabolic trough, tower, dish)"""
    
    def __init__(self, concentrator_type: str = "parabolic_trough"):
        self.type = concentrator_type
        
        # Concentration ratios and efficiencies by type
        self.parameters = {
            "parabolic_trough": {
                "concentration_ratio": 80,
                "optical_efficiency": 0.75,
                "max_temperature": 400,  # °C
                "tracking": "single_axis"
            },
            "solar_tower": {
                "concentration_ratio": 600,
                "optical_efficiency": 0.65,
                "max_temperature": 600,  # °C
                "tracking": "dual_axis"
            },
            "parabolic_dish": {
                "concentration_ratio": 3000,
                "optical_efficiency": 0.85,
                "max_temperature": 800,  # °C
                "tracking": "dual_axis"
            }
        }
        
        self.specs = self.parameters[concentrator_type]
    
    def concentrated_flux(self, dni: float, cosine_losses: float = 1.0) -> float:
        """
        Calculate concentrated solar flux
        
        Args:
            dni: Direct Normal Irradiance (W/m²)
            cosine_losses: Cosine losses due to sun angle (0-1)
            
        Returns:
            Concentrated flux (W/m²)
        """
        return (dni * self.specs["concentration_ratio"] * 
                self.specs["optical_efficiency"] * cosine_losses)
    
    def tracking_efficiency(self, sun_elevation: float, sun_azimuth: float, 
                          collector_azimuth: float = 180) -> float:
        """Calculate tracking efficiency based on sun position"""
        
        if self.specs["tracking"] == "dual_axis":
            return 1.0  # Perfect tracking
        elif self.specs["tracking"] == "single_axis":
            # East-West tracking, cosine losses for elevation
            return np.cos(np.radians(sun_elevation - 90)) if sun_elevation > 10 else 0
        else:
            # Fixed collector
            angle_diff = abs(sun_azimuth - collector_azimuth)
            return max(0, np.cos(np.radians(angle_diff)) * 
                      np.cos(np.radians(90 - sun_elevation)))

class ThermalReceiver:
    """Models thermal receiver and heat transfer fluid system"""
    
    def __init__(self, fluid_type: str = "molten_salt"):
        self.fluid_type = fluid_type
        
        # Heat transfer fluid properties
        self.fluids = {
            "molten_salt": {
                "specific_heat": 1520,  # J/kg·K
                "density": 1900,  # kg/m³
                "max_temp": 600,  # °C
                "min_temp": 290,  # °C
                "thermal_conductivity": 0.57  # W/m·K
            },
            "synthetic_oil": {
                "specific_heat": 2300,  # J/kg·K
                "density": 850,  # kg/m³
                "max_temp": 400,  # °C
                "min_temp": 200,  # °C
                "thermal_conductivity": 0.11  # W/m·K
            },
            "water_steam": {
                "specific_heat": 4180,  # J/kg·K (liquid)
                "density": 1000,  # kg/m³
                "max_temp": 300,  # °C
                "min_temp": 100,  # °C
                "thermal_conductivity": 0.60  # W/m·K
            }
        }
        
        self.properties = self.fluids[fluid_type]
        
    def thermal_efficiency(self, flux: float, ambient_temp: float, 
                          receiver_temp: float) -> float:
        """
        Calculate thermal receiver efficiency
        
        Args:
            flux: Concentrated solar flux (W/m²)
            ambient_temp: Ambient temperature (°C)
            receiver_temp: Receiver temperature (°C)
            
        Returns:
            Thermal efficiency (0-1)
        """
        if flux <= 0:
            return 0
            
        # Heat losses (convection + radiation + conduction)
        temp_diff = receiver_temp - ambient_temp
        
        # Convective losses (simplified)
        h_conv = 10  # W/m²·K (natural convection)
        q_conv = h_conv * temp_diff
        
        # Radiative losses (Stefan-Boltzmann)
        emissivity = 0.85
        stefan_boltzmann = 5.67e-8  # W/m²·K⁴
        t_hot = receiver_temp + 273.15  # K
        t_amb = ambient_temp + 273.15  # K
        q_rad = emissivity * stefan_boltzmann * (t_hot**4 - t_amb**4)
        
        # Total losses
        q_loss = q_conv + q_rad
        
        # Efficiency = (input - losses) / input
        efficiency = max(0, (flux - q_loss) / flux)
        return min(1.0, efficiency)
    
    def heat_capacity_rate(self, mass_flow: float) -> float:
        """Calculate heat capacity rate (mass flow × specific heat)"""
        return mass_flow * self.properties["specific_heat"]

class ThermalStorage:
    """Models thermal energy storage system"""
    
    def __init__(self, capacity_kwh: float = 1000, storage_type: str = "two_tank"):
        self.capacity = capacity_kwh * 3.6e6  # Convert to Joules
        self.storage_type = storage_type
        self.stored_energy = self.capacity * 0.5  # Start half full
        self.efficiency = 0.95  # Round-trip efficiency
        
    def charge(self, power_thermal: float, dt: float) -> float:
        """
        Charge thermal storage
        
        Args:
            power_thermal: Thermal power input (W)
            dt: Time step (hours)
            
        Returns:
            Actual power stored (W)
        """
        energy_in = power_thermal * dt * 3600  # Convert to Joules
        available_capacity = self.capacity - self.stored_energy
        
        energy_stored = min(energy_in * self.efficiency, available_capacity)
        self.stored_energy += energy_stored
        
        return energy_stored / (dt * 3600)  # Return power
    
    def discharge(self, power_demand: float, dt: float) -> float:
        """
        Discharge thermal storage
        
        Args:
            power_demand: Thermal power demand (W)
            dt: Time step (hours)
            
        Returns:
            Actual power delivered (W)
        """
        energy_demand = power_demand * dt * 3600  # Convert to Joules
        available_energy = self.stored_energy
        
        energy_delivered = min(energy_demand, available_energy * self.efficiency)
        self.stored_energy -= energy_delivered / self.efficiency
        
        return energy_delivered / (dt * 3600)  # Return power
    
    def state_of_charge(self) -> float:
        """Return state of charge (0-1)"""
        return self.stored_energy / self.capacity

class PowerBlock:
    """Models thermodynamic power cycle (steam turbine, gas turbine, etc.)"""
    
    def __init__(self, cycle_type: str = "rankine", rated_power_mw: float = 50):
        self.cycle_type = cycle_type
        self.rated_power = rated_power_mw * 1e6  # Convert to Watts
        
        # Cycle parameters
        self.cycles = {
            "rankine": {
                "efficiency_design": 0.42,  # Steam cycle
                "min_load": 0.25,
                "startup_time": 4,  # hours
                "hot_temp": 540,  # °C
                "cold_temp": 40   # °C
            },
            "brayton": {
                "efficiency_design": 0.45,  # Gas turbine cycle
                "min_load": 0.40,
                "startup_time": 0.5,  # hours
                "hot_temp": 800,  # °C
                "cold_temp": 40   # °C
            },
            "stirling": {
                "efficiency_design": 0.40,  # Stirling engine
                "min_load": 0.10,
                "startup_time": 0.1,  # hours
                "hot_temp": 600,  # °C
                "cold_temp": 40   # °C
            }
        }
        
        self.specs = self.cycles[cycle_type]
        self.is_online = False
        self.current_load = 0.0
        
    def carnot_efficiency(self, hot_temp: float, cold_temp: float) -> float:
        """Calculate theoretical Carnot efficiency"""
        t_hot = hot_temp + 273.15  # K
        t_cold = cold_temp + 273.15  # K
        return 1 - (t_cold / t_hot)
    
    def actual_efficiency(self, load_fraction: float, hot_temp: float) -> float:
        """
        Calculate actual cycle efficiency
        
        Args:
            load_fraction: Fraction of rated power (0-1)
            hot_temp: Hot reservoir temperature (°C)
            
        Returns:
            Cycle efficiency (0-1)
        """
        if load_fraction < self.specs["min_load"]:
            return 0
            
        # Base efficiency
        carnot_eff = self.carnot_efficiency(hot_temp, self.specs["cold_temp"])
        relative_eff = self.specs["efficiency_design"] / \
                      self.carnot_efficiency(self.specs["hot_temp"], 
                                           self.specs["cold_temp"])
        
        design_eff = carnot_eff * relative_eff
        
        # Part-load efficiency curve (quadratic approximation)
        part_load_factor = 0.8 + 0.2 * load_fraction
        
        return design_eff * part_load_factor
    
    def electrical_power(self, thermal_power: float, hot_temp: float) -> float:
        """
        Convert thermal power to electrical power
        
        Args:
            thermal_power: Input thermal power (W)
            hot_temp: Hot reservoir temperature (°C)
            
        Returns:
            Electrical power output (W)
        """
        load_fraction = thermal_power / self.rated_power
        
        if load_fraction < self.specs["min_load"]:
            return 0
            
        efficiency = self.actual_efficiency(min(1.0, load_fraction), hot_temp)
        return thermal_power * efficiency

class CSPPlant:
    """Complete CSP plant integrating all components"""
    
    def __init__(self, plant_type: str = "parabolic_trough", size_mw: float = 50):
        self.concentrator = SolarConcentrator(plant_type.replace("_plant", ""))
        self.receiver = ThermalReceiver()
        self.storage = ThermalStorage(capacity_kwh=size_mw * 8)  # 8h storage
        self.power_block = PowerBlock(rated_power_mw=size_mw)
        
        # Plant specifications
        self.collector_area = size_mw * 5000  # m² (rough estimate)
        self.size_mw = size_mw
        
    def instantaneous_performance(self, dni: float, ambient_temp: float, 
                                sun_elevation: float, demand_fraction: float = 1.0) -> Dict:
        """
        Calculate instantaneous plant performance
        
        Args:
            dni: Direct Normal Irradiance (W/m²)
            ambient_temp: Ambient temperature (°C)
            sun_elevation: Sun elevation angle (degrees)
            demand_fraction: Electrical demand as fraction of rated power
            
        Returns:
            Dictionary with performance metrics
        """
        # Solar collection
        tracking_eff = self.concentrator.tracking_efficiency(sun_elevation, 180)
        concentrated_flux = self.concentrator.concentrated_flux(dni, tracking_eff)
        
        # Thermal collection
        receiver_temp = min(self.concentrator.specs["max_temperature"], 
                          ambient_temp + concentrated_flux / 50)  # Simplified
        thermal_eff = self.receiver.thermal_efficiency(concentrated_flux, 
                                                     ambient_temp, receiver_temp)
        
        thermal_power_collected = (concentrated_flux * thermal_eff * 
                                 self.collector_area)
        
        # Power generation demand
        electrical_demand = demand_fraction * self.power_block.rated_power
        thermal_demand = electrical_demand / 0.35  # Assume 35% efficiency
        
        # Energy balance
        thermal_surplus = thermal_power_collected - thermal_demand
        
        if thermal_surplus > 0:
            # Store excess energy
            stored_power = self.storage.charge(thermal_surplus, 0.1)  # 6-min timestep
            thermal_to_power = thermal_demand
        else:
            # Discharge storage to meet demand
            storage_contribution = self.storage.discharge(-thermal_surplus, 0.1)
            thermal_to_power = thermal_power_collected + storage_contribution
            stored_power = 0
        
        # Electrical generation
        electrical_power = self.power_block.electrical_power(thermal_to_power, receiver_temp)
        
        # Overall efficiency
        solar_input = dni * self.collector_area
        overall_efficiency = electrical_power / solar_input if solar_input > 0 else 0
        
        return {
            "solar_input_mw": solar_input / 1e6,
            "thermal_collected_mw": thermal_power_collected / 1e6,
            "thermal_to_power_mw": thermal_to_power / 1e6,
            "electrical_output_mw": electrical_power / 1e6,
            "storage_soc": self.storage.state_of_charge(),
            "receiver_temp": receiver_temp,
            "overall_efficiency": overall_efficiency,
            "thermal_efficiency": thermal_eff,
            "tracking_efficiency": tracking_eff
        }

def plot_csp_performance():
    """Plot CSP plant performance under various conditions"""
    
    # Create different CSP technologies
    plants = {
        "Parabolic Trough": CSPPlant("parabolic_trough", 50),
        "Solar Tower": CSPPlant("solar_tower", 50),
        "Parabolic Dish": CSPPlant("parabolic_dish", 10)
    }
    
    # DNI variation
    dni_values = np.linspace(0, 1000, 100)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance vs DNI
    for plant_name, plant in plants.items():
        efficiencies = []
        power_outputs = []
        
        for dni in dni_values:
            result = plant.instantaneous_performance(dni, 25, 45, 1.0)
            efficiencies.append(result["overall_efficiency"] * 100)
            power_outputs.append(result["electrical_output_mw"])
        
        ax1.plot(dni_values, efficiencies, linewidth=2, label=plant_name)
        ax2.plot(dni_values, power_outputs, linewidth=2, label=plant_name)
    
    ax1.set_xlabel('Direct Normal Irradiance (W/m²)')
    ax1.set_ylabel('Overall Efficiency (%)')
    ax1.set_title('CSP Efficiency vs Solar Irradiance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1000)
    
    ax2.set_xlabel('Direct Normal Irradiance (W/m²)')
    ax2.set_ylabel('Electrical Power (MW)')
    ax2.set_title('CSP Power Output vs Solar Irradiance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 1000)
    
    # Daily operation with storage
    hours = np.arange(0, 24, 0.1)
    
    # Solar resource model
    dni_profile = []
    for hour in hours:
        if 6 <= hour <= 18:
            solar_angle = np.pi * (hour - 6) / 12
            dni = 800 * np.sin(solar_angle) ** 1.5
        else:
            dni = 0
        dni_profile.append(dni)
    
    # Sun elevation approximation
    sun_elevation = []
    for hour in hours:
        if 6 <= hour <= 18:
            elevation = 60 * np.sin(np.pi * (hour - 6) / 12)
        else:
            elevation = 0
        sun_elevation.append(elevation)
    
    # Electrical demand profile (peak in evening)
    demand_profile = 0.7 + 0.3 * np.sin(2 * np.pi * (hours - 6) / 24)
    demand_profile = np.maximum(0.3, demand_profile)  # Minimum 30% demand
    
    # Simulate daily operation
    plant = plants["Parabolic Trough"]
    plant.storage.stored_energy = plant.storage.capacity * 0.5  # Reset storage
    
    results = []
    for dni, elevation, demand in zip(dni_profile, sun_elevation, demand_profile):
        result = plant.instantaneous_performance(dni, 25, elevation, demand)
        results.append(result)
    
    # Extract time series
    electrical_output = [r["electrical_output_mw"] for r in results]
    storage_soc = [r["storage_soc"] * 100 for r in results]
    thermal_collected = [r["thermal_collected_mw"] for r in results]
    
    # Plot daily operation
    ax3.plot(hours, dni_profile, label='DNI', color='orange', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(hours, electrical_output, label='Power Output', color='blue', linewidth=2)
    ax3_twin.plot(hours, np.array(demand_profile) * 50, label='Demand', 
                  color='red', linestyle='--', linewidth=2)
    
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('DNI (W/m²)', color='orange')
    ax3_twin.set_ylabel('Power (MW)', color='blue')
    ax3.set_title('Daily CSP Operation with Storage')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # Storage state of charge
    ax4.fill_between(hours, storage_soc, alpha=0.7, color='green')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Storage State of Charge (%)')
    ax4.set_title('Thermal Energy Storage Throughout Day')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

def efficiency_analysis():
    """Analyze the efficiency chain in CSP systems"""
    
    # Create a representative CSP plant
    plant = CSPPlant("parabolic_trough", 50)
    
    # Operating conditions
    dni = 800  # W/m²
    ambient_temp = 25  # °C
    sun_elevation = 45  # degrees
    
    # Calculate step-by-step efficiencies
    tracking_eff = plant.concentrator.tracking_efficiency(sun_elevation, 180)
    concentrated_flux = plant.concentrator.concentrated_flux(dni, tracking_eff)
    
    receiver_temp = ambient_temp + concentrated_flux / 50
    thermal_eff = plant.receiver.thermal_efficiency(concentrated_flux, ambient_temp, receiver_temp)
    
    # Power cycle efficiency
    cycle_eff = plant.power_block.actual_efficiency(1.0, receiver_temp)
    
    # Overall efficiency
    overall_eff = (plant.concentrator.specs["optical_efficiency"] * 
                  tracking_eff * thermal_eff * cycle_eff)
    
    # Create efficiency waterfall chart
    efficiencies = [
        ("Solar Input", 1.0),
        ("Optical", plant.concentrator.specs["optical_efficiency"]),
        ("Tracking", tracking_eff),
        ("Thermal", thermal_eff),
        ("Power Cycle", cycle_eff),
        ("Overall", overall_eff)
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Efficiency waterfall
    labels = [e[0] for e in efficiencies]
    values = [e[1] * 100 for e in efficiencies]
    
    colors = ['lightblue', 'orange', 'green', 'red', 'purple', 'darkblue']
    bars = ax1.bar(labels, values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Efficiency (%)')
    ax1.set_title('CSP Efficiency Chain Analysis\n(DNI=800 W/m², T=25°C)')
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Sankey-style efficiency flow
    stages = ['Solar\nInput', 'Optical\nLosses', 'Tracking\nLosses', 
              'Thermal\nLosses', 'Cycle\nLosses', 'Electrical\nOutput']
    
    y_positions = np.arange(len(stages))
    cumulative_eff = [1.0]
    
    current_eff = 1.0
    for _, eff in efficiencies[1:-1]:
        current_eff *= eff
        cumulative_eff.append(current_eff)
    
    widths = [eff * 100 for eff in cumulative_eff]
    
    for i, (stage, width) in enumerate(zip(stages, widths)):
        ax2.barh(i, width, height=0.6, alpha=0.7, 
                color=colors[min(i, len(colors)-1)])
        ax2.text(width/2, i, f'{width:.1f}%', 
                ha='center', va='center', fontweight='bold')
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(stages)
    ax2.set_xlabel('Cumulative Efficiency (%)')
    ax2.set_title('Cumulative Efficiency Through CSP System')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 110)
    
    plt.tight_layout()
    plt.show()
    
    print(f"CSP Efficiency Analysis:")
    print(f"- Optical efficiency: {plant.concentrator.specs['optical_efficiency']*100:.1f}%")
    print(f"- Tracking efficiency: {tracking_eff*100:.1f}%")
    print(f"- Thermal efficiency: {thermal_eff*100:.1f}%")
    print(f"- Power cycle efficiency: {cycle_eff*100:.1f}%")
    print(f"- Overall efficiency: {overall_eff*100:.1f}%")

def main():
    """Main demonstration of CSP systems"""
    
    print("=== Concentrated Solar Power (CSP) System Simulation ===\n")
    
    # Create different CSP technologies
    technologies = ["parabolic_trough", "solar_tower", "parabolic_dish"]
    
    print("CSP Technology Comparison:")
    for tech in technologies:
        plant = CSPPlant(tech, 50)
        conc = plant.concentrator
        
        print(f"\n{tech.replace('_', ' ').title()}:")
        print(f"- Concentration ratio: {conc.specs['concentration_ratio']}x")
        print(f"- Optical efficiency: {conc.specs['optical_efficiency']*100:.1f}%")
        print(f"- Max temperature: {conc.specs['max_temperature']}°C")
        print(f"- Tracking: {conc.specs['tracking'].replace('_', ' ')}")
        
        # Performance at standard conditions
        result = plant.instantaneous_performance(800, 25, 45, 1.0)
        print(f"- Overall efficiency: {result['overall_efficiency']*100:.1f}%")
        print(f"- Power output: {result['electrical_output_mw']:.1f} MW")
    
    print(f"\nKey CSP Advantages:")
    print(f"- Higher efficiency than PV (~35% vs ~20%)")
    print(f"- Built-in thermal storage capability")
    print(f"- Dispatchable power generation")
    print(f"- Good capacity factors in sunny regions")
    
    print(f"\nGenerating performance analysis...")
    
    # Generate plots
    plot_csp_performance()
    efficiency_analysis()
    
    print(f"\nKey Insights:")
    print(f"- CSP achieves ~35% efficiency through multiple conversion steps")
    print(f"- Thermal storage enables power generation after sunset")
    print(f"- Different technologies suit different applications")
    print(f"- Higher operating temperatures improve cycle efficiency")

if __name__ == "__main__":
    main()