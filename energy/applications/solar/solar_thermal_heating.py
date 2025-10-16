"""
Solar Thermal Heating System Simulation

This script demonstrates solar thermal collectors for direct heating
applications in buildings and industrial processes. Models flat-plate
and evacuated tube collectors for domestic hot water and space heating.

Key Physics:
- Solar radiation absorption and heat transfer
- Collector thermal efficiency
- Heat exchanger performance
- Thermal mass and storage effects
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import math

# Set plotting style
try:
    plt.style.use('seaborn')
except OSError:
    plt.style.use('default')

class SolarCollector:
    """Models solar thermal collector performance"""
    
    def __init__(self, collector_type: str = "flat_plate"):
        self.type = collector_type
        
        # Collector parameters based on type
        self.parameters = {
            "flat_plate": {
                "optical_efficiency": 0.75,  # F'(τα) - optical efficiency
                "heat_loss_coeff_1": 3.5,     # a1 - linear heat loss coefficient (W/m²K)
                "heat_loss_coeff_2": 0.015,   # a2 - quadratic heat loss coefficient (W/m²K²)
                "incidence_modifier": 0.1,    # b0 - incidence angle modifier
                "cost_per_m2": 300,           # $/m²
                "lifetime": 25,               # years
                "glazing_layers": 1
            },
            "evacuated_tube": {
                "optical_efficiency": 0.70,
                "heat_loss_coeff_1": 1.5,
                "heat_loss_coeff_2": 0.008,
                "incidence_modifier": 0.05,
                "cost_per_m2": 500,
                "lifetime": 20,
                "glazing_layers": 2
            },
            "unglazed": {
                "optical_efficiency": 0.85,
                "heat_loss_coeff_1": 15.0,
                "heat_loss_coeff_2": 0.0,
                "incidence_modifier": 0.15,
                "cost_per_m2": 150,
                "lifetime": 15,
                "glazing_layers": 0
            }
        }
        
        self.specs = self.parameters[collector_type]
    
    def incidence_angle_modifier(self, incidence_angle: float) -> float:
        """
        Calculate incidence angle modifier (IAM)
        
        Args:
            incidence_angle: Angle between sun and collector normal (degrees)
            
        Returns:
            Incidence angle modifier (0-1)
        """
        angle_rad = np.radians(incidence_angle)
        b0 = self.specs["incidence_modifier"]
        
        # Common IAM model: IAM = 1 - b0(1/cos(θ) - 1)
        if incidence_angle >= 90:
            return 0
        
        iam = 1 - b0 * (1/np.cos(angle_rad) - 1)
        return max(0, iam)
    
    def thermal_efficiency(self, irradiance: float, inlet_temp: float, 
                          ambient_temp: float, incidence_angle: float = 0) -> float:
        """
        Calculate instantaneous thermal efficiency using Hottel-Whillier equation
        
        Args:
            irradiance: Solar irradiance (W/m²)
            inlet_temp: Fluid inlet temperature (°C)
            ambient_temp: Ambient temperature (°C)
            incidence_angle: Solar incidence angle (degrees)
            
        Returns:
            Thermal efficiency (0-1)
        """
        if irradiance <= 0:
            return 0
        
        # Temperature difference
        delta_t = inlet_temp - ambient_temp
        
        # Incidence angle modifier
        iam = self.incidence_angle_modifier(incidence_angle)
        
        # Hottel-Whillier efficiency equation
        # η = F'(τα) * IAM - UL * ΔT / G
        a0 = self.specs["optical_efficiency"] * iam
        a1 = self.specs["heat_loss_coeff_1"]
        a2 = self.specs["heat_loss_coeff_2"]
        
        efficiency = a0 - (a1 * delta_t + a2 * delta_t**2) / irradiance
        
        return max(0, min(1, efficiency))
    
    def useful_heat_gain(self, area: float, irradiance: float, inlet_temp: float,
                        ambient_temp: float, incidence_angle: float = 0) -> float:
        """
        Calculate useful heat gain from collector
        
        Args:
            area: Collector area (m²)
            irradiance: Solar irradiance (W/m²)
            inlet_temp: Fluid inlet temperature (°C)
            ambient_temp: Ambient temperature (°C)
            incidence_angle: Solar incidence angle (degrees)
            
        Returns:
            Useful heat gain (W)
        """
        efficiency = self.thermal_efficiency(irradiance, inlet_temp, 
                                           ambient_temp, incidence_angle)
        return area * irradiance * efficiency

class ThermalStorage:
    """Models thermal storage tank for solar heating systems"""
    
    def __init__(self, volume: float = 300, height: float = 2.0):
        """
        Initialize thermal storage tank
        
        Args:
            volume: Tank volume (liters)
            height: Tank height (m)
        """
        self.volume = volume / 1000  # Convert to m³
        self.height = height
        self.diameter = np.sqrt(4 * self.volume / (np.pi * height))
        
        # Water properties
        self.density = 1000  # kg/m³
        self.specific_heat = 4180  # J/kg·K
        self.thermal_conductivity = 0.6  # W/m·K
        
        # Tank properties
        self.insulation_thickness = 0.1  # m
        self.insulation_conductivity = 0.04  # W/m·K
        
        # Initialize stratified temperature profile
        self.n_layers = 10
        self.layer_height = height / self.n_layers
        self.layer_volume = self.volume / self.n_layers
        self.layer_mass = self.density * self.layer_volume
        
        # Initial temperature distribution (stratified)
        self.temperatures = np.linspace(60, 20, self.n_layers)  # °C, hot at top
    
    def heat_loss_coefficient(self, ambient_temp: float = 20) -> float:
        """Calculate overall heat loss coefficient"""
        # Surface area
        side_area = np.pi * self.diameter * self.height
        top_bottom_area = 2 * np.pi * (self.diameter/2)**2
        total_area = side_area + top_bottom_area
        
        # Thermal resistance of insulation
        r_insulation = self.insulation_thickness / (self.insulation_conductivity * total_area)
        
        # Convective resistance (simplified)
        h_conv = 10  # W/m²K
        r_convection = 1 / (h_conv * total_area)
        
        # Overall heat transfer coefficient
        u_overall = 1 / (r_insulation + r_convection)
        
        return u_overall
    
    def add_heat(self, power: float, inlet_temp: float, dt: float, 
                position: str = "bottom") -> float:
        """
        Add heat to tank at specified position
        
        Args:
            power: Heat input rate (W)
            inlet_temp: Inlet fluid temperature (°C)
            dt: Time step (hours)
            position: "top", "middle", or "bottom"
            
        Returns:
            Average tank temperature (°C)
        """
        energy = power * dt * 3600  # Convert to Joules
        
        # Determine injection layer
        if position == "top":
            layer_idx = 0
        elif position == "middle":
            layer_idx = self.n_layers // 2
        else:  # bottom
            layer_idx = self.n_layers - 1
        
        # Add energy to specific layer
        temp_rise = energy / (self.layer_mass * self.specific_heat)
        self.temperatures[layer_idx] += temp_rise
        
        # Apply buoyancy effects (hot water rises)
        self.apply_stratification()
        
        return np.mean(self.temperatures)
    
    def remove_heat(self, power: float, dt: float, position: str = "top") -> Tuple[float, float]:
        """
        Remove heat from tank
        
        Args:
            power: Heat removal rate (W)
            dt: Time step (hours)
            position: "top", "middle", or "bottom"
            
        Returns:
            (outlet_temperature, average_tank_temperature)
        """
        energy = power * dt * 3600  # Convert to Joules
        
        # Determine extraction layer
        if position == "top":
            layer_idx = 0
        elif position == "middle":
            layer_idx = self.n_layers // 2
        else:  # bottom
            layer_idx = self.n_layers - 1
        
        # Extract energy from specific layer
        outlet_temp = self.temperatures[layer_idx]
        temp_drop = energy / (self.layer_mass * self.specific_heat)
        self.temperatures[layer_idx] -= temp_drop
        
        # Apply stratification
        self.apply_stratification()
        
        return outlet_temp, np.mean(self.temperatures)
    
    def apply_stratification(self):
        """Apply thermal stratification effects"""
        # Simple mixing model - hot water floats to top
        self.temperatures = np.sort(self.temperatures)[::-1]
    
    def ambient_losses(self, ambient_temp: float, dt: float):
        """Calculate and apply ambient heat losses"""
        dt_seconds = dt * 3600
        u_loss = self.heat_loss_coefficient()
        
        for i in range(self.n_layers):
            temp_diff = self.temperatures[i] - ambient_temp
            heat_loss = u_loss * temp_diff * dt_seconds / self.n_layers
            temp_drop = heat_loss / (self.layer_mass * self.specific_heat)
            self.temperatures[i] -= temp_drop

class SolarHeatingSystem:
    """Complete solar heating system with collector and storage"""
    
    def __init__(self, collector_type: str = "flat_plate", 
                 collector_area: float = 10, storage_volume: float = 300):
        self.collector = SolarCollector(collector_type)
        self.storage = ThermalStorage(storage_volume)
        self.collector_area = collector_area
        
        # System parameters
        self.pump_power = 100  # W (circulation pump)
        self.heat_exchanger_efficiency = 0.85
        self.pipe_heat_loss = 0.05  # 5% loss in piping
        
        # Control parameters
        self.collector_on_temp_diff = 5  # °C (turn on when collector > tank + 5°C)
        self.collector_off_temp_diff = 2  # °C (turn off when collector < tank + 2°C)
        self.pump_running = False
        
    def collector_control(self, collector_temp: float, tank_temp: float) -> bool:
        """Determine if collector pump should run"""
        temp_diff = collector_temp - tank_temp
        
        if not self.pump_running:
            # Turn on pump
            if temp_diff > self.collector_on_temp_diff:
                self.pump_running = True
        else:
            # Turn off pump
            if temp_diff < self.collector_off_temp_diff:
                self.pump_running = False
                
        return self.pump_running
    
    def system_performance(self, irradiance: float, ambient_temp: float,
                          hot_water_demand: float, dt: float = 0.1) -> Dict:
        """
        Calculate system performance for given conditions
        
        Args:
            irradiance: Solar irradiance (W/m²)
            ambient_temp: Ambient temperature (°C)
            hot_water_demand: Hot water energy demand (W)
            dt: Time step (hours)
            
        Returns:
            Dictionary with performance metrics
        """
        # Average tank temperature for collector inlet
        tank_avg_temp = np.mean(self.storage.temperatures)
        
        # Collector performance
        collector_efficiency = self.collector.thermal_efficiency(
            irradiance, tank_avg_temp, ambient_temp)
        
        heat_collected = self.collector.useful_heat_gain(
            self.collector_area, irradiance, tank_avg_temp, ambient_temp)
        
        # Control logic
        collector_outlet_temp = tank_avg_temp + (heat_collected / 
                                               (2000 * 4180)) * 3600  # Simplified
        pump_on = self.collector_control(collector_outlet_temp, tank_avg_temp)
        
        # Energy flows
        if pump_on and heat_collected > 0:
            # Heat added to storage (accounting for heat exchanger and pipe losses)
            net_heat_to_storage = (heat_collected * self.heat_exchanger_efficiency * 
                                 (1 - self.pipe_heat_loss))
            self.storage.add_heat(net_heat_to_storage, collector_outlet_temp, dt)
            pump_energy = self.pump_power * dt  # Wh
        else:
            net_heat_to_storage = 0
            pump_energy = 0
        
        # Hot water demand
        if hot_water_demand > 0:
            outlet_temp, new_tank_temp = self.storage.remove_heat(
                hot_water_demand, dt, "top")
        else:
            outlet_temp = self.storage.temperatures[0]
            new_tank_temp = np.mean(self.storage.temperatures)
        
        # Ambient losses
        self.storage.ambient_losses(ambient_temp, dt)
        
        # Calculate solar fraction
        total_heat_demand = hot_water_demand * dt
        solar_contribution = min(net_heat_to_storage * dt, total_heat_demand)
        solar_fraction = solar_contribution / total_heat_demand if total_heat_demand > 0 else 0
        
        return {
            "collector_efficiency": collector_efficiency,
            "heat_collected_w": heat_collected,
            "net_heat_to_storage_w": net_heat_to_storage,
            "tank_avg_temp": new_tank_temp,
            "outlet_temp": outlet_temp,
            "pump_running": pump_on,
            "pump_energy_wh": pump_energy,
            "solar_fraction": solar_fraction
        }

def plot_collector_performance():
    """Plot collector efficiency curves for different types"""
    
    collectors = ["flat_plate", "evacuated_tube", "unglazed"]
    irradiance = 800  # W/m²
    ambient_temp = 20  # °C
    
    # Temperature differences
    temp_diffs = np.linspace(0, 80, 100)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Efficiency vs temperature difference
    for collector_type in collectors:
        collector = SolarCollector(collector_type)
        efficiencies = []
        
        for dt in temp_diffs:
            inlet_temp = ambient_temp + dt
            eff = collector.thermal_efficiency(irradiance, inlet_temp, ambient_temp)
            efficiencies.append(eff * 100)
        
        ax1.plot(temp_diffs, efficiencies, linewidth=2, 
                label=collector_type.replace('_', ' ').title())
    
    ax1.set_xlabel('Temperature Difference (°C)')
    ax1.set_ylabel('Efficiency (%)')
    ax1.set_title(f'Collector Efficiency vs Temperature\n(Irradiance = {irradiance} W/m²)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 90)
    
    # Efficiency vs irradiance
    irradiances = np.linspace(100, 1000, 100)
    temp_diff = 40  # °C
    
    for collector_type in collectors:
        collector = SolarCollector(collector_type)
        efficiencies = []
        
        for irr in irradiances:
            inlet_temp = ambient_temp + temp_diff
            eff = collector.thermal_efficiency(irr, inlet_temp, ambient_temp)
            efficiencies.append(eff * 100)
        
        ax2.plot(irradiances, efficiencies, linewidth=2,
                label=collector_type.replace('_', ' ').title())
    
    ax2.set_xlabel('Solar Irradiance (W/m²)')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title(f'Collector Efficiency vs Irradiance\n(ΔT = {temp_diff}°C)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Incidence angle effects
    angles = np.linspace(0, 80, 100)
    
    for collector_type in collectors:
        collector = SolarCollector(collector_type)
        modifiers = []
        
        for angle in angles:
            iam = collector.incidence_angle_modifier(angle)
            modifiers.append(iam)
        
        ax3.plot(angles, modifiers, linewidth=2,
                label=collector_type.replace('_', ' ').title())
    
    ax3.set_xlabel('Incidence Angle (degrees)')
    ax3.set_ylabel('Incidence Angle Modifier')
    ax3.set_title('Incidence Angle Effects')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, 80)
    ax3.set_ylim(0, 1)
    
    # Cost-effectiveness comparison
    collector_areas = np.linspace(5, 50, 20)
    annual_irradiation = 1500  # kWh/m²/year (typical)
    
    for collector_type in collectors:
        collector = SolarCollector(collector_type)
        annual_yields = []
        
        for area in collector_areas:
            # Simplified annual calculation
            avg_efficiency = 0.5  # Rough average
            annual_yield = area * annual_irradiation * avg_efficiency
            annual_yields.append(annual_yield)
        
        costs = collector_areas * collector.specs["cost_per_m2"]
        cost_per_kwh = costs / np.array(annual_yields)
        
        ax4.plot(collector_areas, cost_per_kwh, linewidth=2,
                label=collector_type.replace('_', ' ').title())
    
    ax4.set_xlabel('Collector Area (m²)')
    ax4.set_ylabel('Cost per Annual kWh ($/kWh)')
    ax4.set_title('Cost Effectiveness vs System Size')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def simulate_daily_operation():
    """Simulate daily operation of solar heating system"""
    
    # Create system
    system = SolarHeatingSystem("flat_plate", collector_area=15, storage_volume=500)
    
    # Time array (24 hours, 10-minute intervals)
    hours = np.arange(0, 24, 1/6)
    
    # Weather and load profiles
    irradiances = []
    ambient_temps = []
    hot_water_demands = []
    
    for hour in hours:
        # Solar irradiance (clear day)
        if 6 <= hour <= 18:
            solar_angle = np.pi * (hour - 6) / 12
            irr = 900 * np.sin(solar_angle) ** 1.5
        else:
            irr = 0
        irradiances.append(irr)
        
        # Ambient temperature
        temp = 15 + 10 * np.sin(2 * np.pi * (hour - 8) / 24)
        ambient_temps.append(temp)
        
        # Hot water demand (morning and evening peaks)
        demand = 3000 * (np.exp(-((hour - 7)**2) / 2) + 
                        0.7 * np.exp(-((hour - 19)**2) / 4))
        hot_water_demands.append(demand)
    
    # Simulate system operation
    results = []
    dt = 1/6  # 10-minute time steps
    
    for irr, temp, demand in zip(irradiances, ambient_temps, hot_water_demands):
        result = system.system_performance(irr, temp, demand, dt)
        results.append(result)
    
    # Extract time series data
    collector_efficiency = [r["collector_efficiency"] * 100 for r in results]
    heat_collected = [r["heat_collected_w"] / 1000 for r in results]  # kW
    tank_temps = [r["tank_avg_temp"] for r in results]
    outlet_temps = [r["outlet_temp"] for r in results]
    solar_fractions = [r["solar_fraction"] * 100 for r in results]
    pump_status = [r["pump_running"] for r in results]
    
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Solar resource and collection
    ax1.fill_between(hours, irradiances, alpha=0.3, color='orange', label='Irradiance')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(hours, heat_collected, color='red', linewidth=2, label='Heat Collected')
    
    ax1.set_ylabel('Solar Irradiance (W/m²)', color='orange')
    ax1_twin.set_ylabel('Heat Collected (kW)', color='red')
    ax1.set_title('Solar Resource and Heat Collection')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    
    # Storage temperatures
    ax2.plot(hours, tank_temps, linewidth=2, label='Tank Average', color='blue')
    ax2.plot(hours, outlet_temps, linewidth=2, label='Hot Water Outlet', color='red')
    ax2.axhline(y=60, color='green', linestyle='--', label='Target Temperature')
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Storage Tank Temperatures')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 24)
    
    # Hot water demand and solar fraction
    ax3.fill_between(hours, np.array(hot_water_demands)/1000, alpha=0.3, 
                     color='blue', label='Hot Water Demand')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(hours, solar_fractions, color='green', linewidth=2, 
                  label='Solar Fraction')
    
    ax3.set_ylabel('Hot Water Demand (kW)', color='blue')
    ax3_twin.set_ylabel('Solar Fraction (%)', color='green')
    ax3.set_title('Load Profile and Solar Contribution')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)
    
    # System efficiency and pump operation
    ax4.plot(hours, collector_efficiency, linewidth=2, color='purple', 
             label='Collector Efficiency')
    
    # Show pump operation as background
    pump_periods = []
    for i, status in enumerate(pump_status):
        if status:
            pump_periods.append(hours[i])
        else:
            pump_periods.append(np.nan)
    
    ax4.fill_between(hours, 0, 100, where=[p for p in pump_status], 
                     alpha=0.2, color='gray', label='Pump On')
    
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Collector Efficiency (%)')
    ax4.set_title('System Efficiency and Pump Operation')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(0, 24)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate daily performance metrics
    daily_irradiation = np.sum(irradiances) * dt / 1000  # kWh/m²
    daily_heat_collected = np.sum(heat_collected) * dt  # kWh
    daily_hot_water_demand = np.sum(hot_water_demands) * dt / 1000  # kWh
    daily_solar_contribution = np.sum([r["net_heat_to_storage_w"] * dt / 1000 
                                      for r in results])  # kWh
    
    overall_solar_fraction = daily_solar_contribution / daily_hot_water_demand * 100
    
    print(f"\nDaily Performance Summary:")
    print(f"- Solar irradiation: {daily_irradiation:.1f} kWh/m²")
    print(f"- Heat collected: {daily_heat_collected:.1f} kWh")
    print(f"- Hot water demand: {daily_hot_water_demand:.1f} kWh")
    print(f"- Solar contribution: {daily_solar_contribution:.1f} kWh")
    print(f"- Overall solar fraction: {overall_solar_fraction:.1f}%")

def main():
    """Main demonstration of solar thermal systems"""
    
    print("=== Solar Thermal Heating System Simulation ===\n")
    
    # Compare collector technologies
    collectors = ["flat_plate", "evacuated_tube", "unglazed"]
    
    print("Solar Thermal Collector Comparison:")
    for collector_type in collectors:
        collector = SolarCollector(collector_type)
        specs = collector.specs
        
        print(f"\n{collector_type.replace('_', ' ').title()} Collector:")
        print(f"- Optical efficiency: {specs['optical_efficiency']*100:.1f}%")
        print(f"- Heat loss coeff (linear): {specs['heat_loss_coeff_1']:.1f} W/m²K")
        print(f"- Heat loss coeff (quadratic): {specs['heat_loss_coeff_2']:.3f} W/m²K²")
        print(f"- Cost: ${specs['cost_per_m2']}/m²")
        print(f"- Lifetime: {specs['lifetime']} years")
        
        # Performance at standard conditions
        eff_low = collector.thermal_efficiency(800, 30, 20) * 100
        eff_high = collector.thermal_efficiency(800, 70, 20) * 100
        print(f"- Efficiency at ΔT=10°C: {eff_low:.1f}%")
        print(f"- Efficiency at ΔT=50°C: {eff_high:.1f}%")
    
    print(f"\nTypical Applications:")
    print(f"- Flat Plate: Domestic hot water, space heating")
    print(f"- Evacuated Tube: High-temperature applications, cold climates")
    print(f"- Unglazed: Pool heating, low-temperature processes")
    
    print(f"\nGenerating performance analysis...")
    
    # Generate plots and simulations
    plot_collector_performance()
    simulate_daily_operation()
    
    print(f"\nKey Insights:")
    print(f"- Solar thermal systems can provide 50-80% of hot water needs")
    print(f"- Collector efficiency decreases with operating temperature")
    print(f"- Thermal storage enables load shifting and system optimization")
    print(f"- Proper sizing critical for cost-effectiveness")

if __name__ == "__main__":
    main()