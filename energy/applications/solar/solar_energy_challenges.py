"""
Solar Energy Challenges: Intermittency, Storage, and Weather Dependence

This script demonstrates the key challenges facing solar energy systems:
intermittency, energy storage requirements, and weather dependence impacts.
Models grid integration, variability, and mitigation strategies.

Key Physics:
- Solar resource variability and forecasting
- Energy storage sizing and dispatch
- Grid stability and power quality
- Economic impacts of intermittency
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import random

# Set plotting style
try:
    plt.style.use('seaborn')
except OSError:
    plt.style.use('default')

class WeatherCondition:
    """Represents weather conditions affecting solar generation"""
    def __init__(self, cloud_cover=0, visibility=50, humidity=0.5, wind_speed=5, temperature=25):
        self.cloud_cover = cloud_cover  # 0-1 (0 = clear, 1 = overcast)
        self.visibility = visibility   # km
        self.humidity = humidity     # 0-1
        self.wind_speed = wind_speed   # m/s
        self.temperature = temperature  # °C
    
class SolarResource:
    """Models solar irradiance with weather variability"""
    
    def __init__(self, latitude: float = 35.0):
        self.latitude = latitude
        self.clear_sky_model = "simplified"
        
    def clear_sky_irradiance(self, day_of_year: int, hour: float) -> float:
        """
        Calculate clear sky irradiance using simplified model
        
        Args:
            day_of_year: Day of year (1-365)
            hour: Hour of day (0-24)
            
        Returns:
            Clear sky irradiance (W/m²)
        """
        # Solar declination angle
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation angle
        lat_rad = np.radians(self.latitude)
        dec_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        
        elevation = np.arcsin(np.sin(lat_rad) * np.sin(dec_rad) + 
                             np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad))
        
        elevation_deg = np.degrees(elevation)
        
        if elevation_deg <= 0:
            return 0
        
        # Air mass
        air_mass = 1 / np.sin(elevation) if elevation > 0 else 0
        air_mass = min(air_mass, 10)  # Limit extreme values
        
        # Clear sky irradiance (simplified Iqbal model)
        extraterrestrial = 1367 * (1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365)))
        
        # Atmospheric attenuation
        tau_beam = 0.56 * (np.exp(-0.65 * air_mass) + np.exp(-0.095 * air_mass))
        tau_diffuse = 0.271 - 0.294 * tau_beam
        
        dni = extraterrestrial * tau_beam
        dhi = extraterrestrial * tau_diffuse * np.sin(elevation)
        ghi = dni * np.sin(elevation) + dhi
        
        return max(0, ghi)
    
    def cloud_impact(self, clear_sky_irr: float, weather: WeatherCondition) -> float:
        """
        Model cloud impact on solar irradiance
        
        Args:
            clear_sky_irr: Clear sky irradiance (W/m²)
            weather: Weather conditions
            
        Returns:
            Actual irradiance (W/m²)
        """
        # Cloud cover effects
        cloud_factor = 1 - 0.75 * weather.cloud_cover**2
        
        # Atmospheric scattering due to humidity
        humidity_factor = 1 - 0.1 * weather.humidity
        
        # Visibility effects (pollution, dust)
        visibility_factor = min(1.0, weather.visibility / 50)  # Normalize to 50km
        
        # Combined effects
        total_factor = cloud_factor * humidity_factor * visibility_factor
        
        return clear_sky_irr * total_factor
    
    def generate_realistic_profile(self, days: int = 7, resolution_hours: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic solar irradiance profile with weather variability
        
        Args:
            days: Number of days to simulate
            resolution_hours: Time resolution in hours
            
        Returns:
            (time_array, irradiance_array)
        """
        hours_total = days * 24
        time_points = int(hours_total / resolution_hours)
        time_array = np.linspace(0, hours_total, time_points)
        irradiance_array = np.zeros(time_points)
        
        for i, time_hour in enumerate(time_array):
            day_of_year = int(time_hour // 24) + 1
            hour_of_day = time_hour % 24
            
            # Clear sky irradiance
            clear_sky = self.clear_sky_irradiance(day_of_year, hour_of_day)
            
            # Generate weather conditions with some persistence
            if i == 0:
                cloud_cover = random.uniform(0, 0.8)
                humidity = random.uniform(0.3, 0.9)
                visibility = random.uniform(10, 50)
            else:
                # Add persistence to weather
                cloud_cover = max(0, min(1, cloud_cover + random.uniform(-0.1, 0.1)))
                humidity = max(0, min(1, humidity + random.uniform(-0.05, 0.05)))
                visibility = max(5, min(50, visibility + random.uniform(-2, 2)))
            
            weather = WeatherCondition(
                cloud_cover=cloud_cover,
                humidity=humidity,
                visibility=visibility,
                wind_speed=random.uniform(2, 15),
                temperature=20 + 10 * np.sin(2 * np.pi * hour_of_day / 24)
            )
            
            # Apply weather effects
            actual_irradiance = self.cloud_impact(clear_sky, weather)
            
            # Add short-term variability (cloud edges, etc.)
            if actual_irradiance > 0:
                variability = 1 + 0.1 * random.uniform(-1, 1)
                actual_irradiance *= variability
            
            irradiance_array[i] = max(0, actual_irradiance)
        
        return time_array, irradiance_array

class EnergyStorage:
    """Models various energy storage technologies for solar integration"""
    
    def __init__(self, technology: str = "lithium_ion", capacity_mwh: float = 100):
        self.technology = technology
        self.capacity = capacity_mwh
        
        # Technology parameters
        self.parameters = {
            "lithium_ion": {
                "round_trip_efficiency": 0.90,
                "max_power_ratio": 0.5,  # C-rate (1/hours for full discharge)
                "self_discharge_rate": 0.0001,  # per hour
                "cycle_life": 5000,
                "cost_per_mwh": 200000,  # $
                "response_time": 0.001  # hours
            },
            "pumped_hydro": {
                "round_trip_efficiency": 0.80,
                "max_power_ratio": 0.25,
                "self_discharge_rate": 0.00001,
                "cycle_life": 25000,
                "cost_per_mwh": 50000,
                "response_time": 0.25
            },
            "compressed_air": {
                "round_trip_efficiency": 0.65,
                "max_power_ratio": 0.20,
                "self_discharge_rate": 0.0001,
                "cycle_life": 20000,
                "cost_per_mwh": 100000,
                "response_time": 0.5
            },
            "flywheel": {
                "round_trip_efficiency": 0.85,
                "max_power_ratio": 4.0,  # Very high power
                "self_discharge_rate": 0.05,  # High self-discharge
                "cycle_life": 100000,
                "cost_per_mwh": 1000000,  # High cost per energy
                "response_time": 0.0001
            }
        }
        
        self.specs = self.parameters[technology]
        self.max_power = capacity_mwh * self.specs["max_power_ratio"]
        
        # State variables
        self.stored_energy = capacity_mwh * 0.5  # Start at 50% SOC
        self.cycles_completed = 0
        
    def charge(self, power_mw: float, duration_hours: float) -> float:
        """
        Attempt to charge storage
        
        Args:
            power_mw: Charging power (MW)
            duration_hours: Charging duration (hours)
            
        Returns:
            Actually charged power (MW)
        """
        # Limit by maximum charging power
        actual_power = min(power_mw, self.max_power)
        
        # Calculate energy to be stored
        energy_in = actual_power * duration_hours
        energy_storable = energy_in * self.specs["round_trip_efficiency"]
        
        # Check capacity limits
        available_capacity = self.capacity - self.stored_energy
        energy_stored = min(energy_storable, available_capacity)
        
        # Update state
        self.stored_energy += energy_stored
        
        # Track cycles (simplified)
        if energy_stored > 0:
            self.cycles_completed += energy_stored / self.capacity
        
        return energy_stored / duration_hours / self.specs["round_trip_efficiency"]
    
    def discharge(self, power_mw: float, duration_hours: float) -> float:
        """
        Attempt to discharge storage
        
        Args:
            power_mw: Discharge power (MW)
            duration_hours: Discharge duration (hours)
            
        Returns:
            Actually discharged power (MW)
        """
        # Limit by maximum discharge power
        actual_power = min(power_mw, self.max_power)
        
        # Calculate energy needed
        energy_demanded = actual_power * duration_hours
        
        # Check available energy
        available_energy = self.stored_energy
        energy_delivered = min(energy_demanded, available_energy)
        
        # Update state
        self.stored_energy -= energy_delivered
        
        # Track cycles
        if energy_delivered > 0:
            self.cycles_completed += energy_delivered / self.capacity
        
        return energy_delivered / duration_hours
    
    def apply_self_discharge(self, duration_hours: float):
        """Apply self-discharge losses"""
        discharge_rate = self.specs["self_discharge_rate"]
        energy_lost = self.stored_energy * discharge_rate * duration_hours
        self.stored_energy = max(0, self.stored_energy - energy_lost)
    
    def state_of_charge(self) -> float:
        """Return state of charge (0-1)"""
        return self.stored_energy / self.capacity
    
    def degradation_factor(self) -> float:
        """Calculate capacity degradation due to cycling"""
        if self.cycles_completed > self.specs["cycle_life"]:
            # Linear degradation after cycle life
            excess_cycles = self.cycles_completed - self.specs["cycle_life"]
            degradation = 0.8 - 0.2 * (excess_cycles / self.specs["cycle_life"])
            return max(0.5, degradation)  # Minimum 50% capacity
        return 1.0

class GridIntegration:
    """Models grid integration challenges and solutions"""
    
    def __init__(self, solar_capacity_mw: float = 100):
        self.solar_capacity = solar_capacity_mw
        self.grid_frequency = 60  # Hz
        self.voltage_regulation_tolerance = 0.05  # ±5%
        
    def power_quality_impact(self, solar_power: float, cloud_ramp_rate: float) -> Dict:
        """
        Assess power quality impacts of solar variability
        
        Args:
            solar_power: Current solar power output (MW)
            cloud_ramp_rate: Rate of power change (MW/minute)
            
        Returns:
            Power quality metrics
        """
        # Frequency deviation (simplified model)
        # Rapid changes in generation affect grid frequency
        frequency_deviation = -cloud_ramp_rate * 0.001  # Hz per MW/min
        actual_frequency = self.grid_frequency + frequency_deviation
        
        # Voltage fluctuation
        # Voltage varies with power injection
        voltage_deviation = (solar_power / self.solar_capacity) * 0.02  # ±2% max
        
        # Power quality indicators
        frequency_stable = abs(frequency_deviation) < 0.1  # ±0.1 Hz acceptable
        voltage_stable = abs(voltage_deviation) < self.voltage_regulation_tolerance
        
        return {
            "frequency_hz": actual_frequency,
            "frequency_deviation_hz": frequency_deviation,
            "voltage_deviation_pu": voltage_deviation,
            "frequency_stable": frequency_stable,
            "voltage_stable": voltage_stable,
            "power_quality_good": frequency_stable and voltage_stable
        }
    
    def calculate_ramp_rates(self, power_profile: np.ndarray, time_array: np.ndarray) -> np.ndarray:
        """Calculate power ramp rates (MW/minute)"""
        dt = (time_array[1] - time_array[0]) * 60  # Convert to minutes
        ramp_rates = np.gradient(power_profile) / dt
        return ramp_rates

def analyze_intermittency():
    """Analyze solar intermittency patterns and impacts"""
    
    # Generate realistic solar profile
    solar_resource = SolarResource(latitude=35)
    time_hours, irradiance = solar_resource.generate_realistic_profile(days=7, resolution_hours=0.1)
    
    # Convert to power output (100 MW solar farm)
    solar_capacity = 100  # MW
    efficiency = 0.20
    power_output = irradiance * solar_capacity * efficiency / 1000  # Convert W/m² to MW
    
    # Calculate variability metrics
    grid = GridIntegration(solar_capacity)
    ramp_rates = grid.calculate_ramp_rates(power_output, time_hours)
    
    # Power quality analysis
    power_quality_results = []
    for i in range(len(power_output)):
        pq = grid.power_quality_impact(power_output[i], ramp_rates[i])
        power_quality_results.append(pq)
    
    frequency_deviations = [pq["frequency_deviation_hz"] for pq in power_quality_results]
    voltage_deviations = [pq["voltage_deviation_pu"] for pq in power_quality_results]
    
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Solar irradiance and power output
    ax1.plot(time_hours, irradiance, alpha=0.7, color='orange', label='Solar Irradiance')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_hours, power_output, color='blue', linewidth=2, label='Power Output')
    
    ax1.set_ylabel('Irradiance (W/m²)', color='orange')
    ax1_twin.set_ylabel('Power Output (MW)', color='blue')
    ax1.set_title('Solar Resource Variability (7 Days)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(time_hours))
    
    # Ramp rate analysis
    ax2.plot(time_hours[1:], ramp_rates[1:], color='red', alpha=0.7)
    ax2.axhline(y=10, color='red', linestyle='--', label='Grid Limit (+10 MW/min)')
    ax2.axhline(y=-10, color='red', linestyle='--', label='Grid Limit (-10 MW/min)')
    ax2.fill_between(time_hours[1:], -10, 10, alpha=0.2, color='green', label='Acceptable Range')
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Ramp Rate (MW/minute)')
    ax2.set_title('Power Ramp Rate Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, max(time_hours))
    
    # Frequency impact
    ax3.plot(time_hours, frequency_deviations, color='purple', linewidth=1)
    ax3.axhline(y=0.1, color='red', linestyle='--', label='Stability Limit')
    ax3.axhline(y=-0.1, color='red', linestyle='--')
    ax3.fill_between(time_hours, -0.1, 0.1, alpha=0.2, color='green')
    
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Frequency Deviation (Hz)')
    ax3.set_title('Grid Frequency Impact')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, max(time_hours))
    
    # Variability statistics
    daily_variability = []
    for day in range(7):
        day_start = day * 24 / 0.1
        day_end = (day + 1) * 24 / 0.1
        day_power = power_output[int(day_start):int(day_end)]
        if len(day_power) > 0:
            daily_var = np.std(day_power) / np.mean(day_power) if np.mean(day_power) > 0 else 0
            daily_variability.append(daily_var)
    
    days = range(1, 8)
    ax4.bar(days, daily_variability, alpha=0.7, color='skyblue')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Daily Solar Variability')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    max_ramp_up = np.max(ramp_rates)
    max_ramp_down = np.min(ramp_rates)
    avg_variability = np.mean(daily_variability)
    
    print(f"Intermittency Analysis Results:")
    print(f"- Maximum ramp up: {max_ramp_up:.1f} MW/minute")
    print(f"- Maximum ramp down: {max_ramp_down:.1f} MW/minute")
    print(f"- Average daily variability: {avg_variability:.3f}")
    print(f"- Frequency excursions: {sum(1 for f in frequency_deviations if abs(f) > 0.1)}")
    
    return time_hours, power_output, ramp_rates

def storage_sizing_analysis():
    """Analyze storage requirements for different penetration levels"""
    
    # Generate annual solar profile (simplified)
    days = 365
    solar_resource = SolarResource()
    time_hours, irradiance = solar_resource.generate_realistic_profile(days=30, resolution_hours=1)
    
    # Different solar penetration levels
    penetration_levels = [10, 20, 30, 40, 50]  # % of peak demand
    peak_demand = 1000  # MW
    
    storage_requirements = []
    
    for penetration in penetration_levels:
        solar_capacity = peak_demand * penetration / 100
        power_output = irradiance * solar_capacity * 0.20 / 1000  # 20% efficiency
        
        # Simple demand profile (daily cycle)
        demand_profile = []
        for hour in time_hours:
            hour_of_day = hour % 24
            # Peak in early evening, minimum at night
            demand_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            demand_profile.append(peak_demand * demand_factor)
        
        demand_profile = np.array(demand_profile)
        
        # Calculate net load (demand - solar)
        net_load = demand_profile - power_output
        
        # Storage sizing based on maximum energy deficit/surplus
        cumulative_energy = np.cumsum(net_load)
        min_cumulative = np.min(cumulative_energy)
        max_cumulative = np.max(cumulative_energy)
        
        # Storage capacity needed
        storage_capacity = max_cumulative - min_cumulative
        storage_requirements.append(storage_capacity)
    
    # Plotting storage requirements
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Storage capacity vs penetration
    ax1.plot(penetration_levels, storage_requirements, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Solar Penetration (%)')
    ax1.set_ylabel('Storage Requirement (MWh)')
    ax1.set_title('Storage Capacity vs Solar Penetration')
    ax1.grid(True, alpha=0.3)
    
    # Storage cost analysis
    storage_technologies = ["lithium_ion", "pumped_hydro", "compressed_air"]
    storage_costs = []
    
    for tech in storage_technologies:
        storage = EnergyStorage(tech, 100)  # 100 MWh reference
        costs = [req * storage.specs["cost_per_mwh"] / 1e6 for req in storage_requirements]  # Million $
        storage_costs.append(costs)
        ax2.plot(penetration_levels, costs, 'o-', linewidth=2, label=tech.replace('_', ' ').title())
    
    ax2.set_xlabel('Solar Penetration (%)')
    ax2.set_ylabel('Storage Cost (Million $)')
    ax2.set_title('Storage Cost vs Solar Penetration')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Storage Sizing Analysis:")
    for i, penetration in enumerate(penetration_levels):
        print(f"- {penetration}% solar: {storage_requirements[i]:.0f} MWh storage needed")

def weather_dependence_analysis():
    """Analyze impact of weather patterns on solar generation"""
    
    # Define different weather scenarios
    weather_scenarios = {
        "Clear Week": {"cloud_avg": 0.1, "cloud_var": 0.05},
        "Variable Week": {"cloud_avg": 0.4, "cloud_var": 0.3},
        "Cloudy Week": {"cloud_avg": 0.8, "cloud_var": 0.1},
        "Storm Pattern": {"cloud_avg": 0.6, "cloud_var": 0.4}
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    scenario_results = {}
    
    for i, (scenario_name, params) in enumerate(weather_scenarios.items()):
        # Generate weather-specific solar profile
        solar_resource = SolarResource()
        
        # Modify cloud generation for this scenario
        time_hours = np.linspace(0, 7*24, 7*24*10)  # 7 days, 6-minute resolution
        irradiance = []
        
        cloud_cover = params["cloud_avg"]
        
        for j, hour in enumerate(time_hours):
            day_of_year = int(hour // 24) + 180  # Summer day
            hour_of_day = hour % 24
            
            # Clear sky irradiance
            clear_sky = solar_resource.clear_sky_irradiance(day_of_year, hour_of_day)
            
            # Weather effects with scenario-specific parameters
            if j % 10 == 0:  # Update weather every hour
                cloud_variation = np.random.normal(0, params["cloud_var"])
                cloud_cover = np.clip(params["cloud_avg"] + cloud_variation, 0, 1)
            
            # Apply cloud effects
            cloud_factor = 1 - 0.75 * cloud_cover**2
            actual_irradiance = clear_sky * cloud_factor
            irradiance.append(max(0, actual_irradiance))
        
        irradiance = np.array(irradiance)
        
        # Convert to power (100 MW solar farm)
        power_output = irradiance * 100 * 0.20 / 1000  # MW
        
        # Calculate daily energy yields
        daily_energies = []
        for day in range(7):
            day_start = day * 24 * 10
            day_end = (day + 1) * 24 * 10
            day_power = power_output[day_start:day_end]
            daily_energy = np.sum(day_power) * 0.1  # MWh (0.1 hour resolution)
            daily_energies.append(daily_energy)
        
        scenario_results[scenario_name] = {
            "daily_energies": daily_energies,
            "total_energy": sum(daily_energies),
            "variability": np.std(daily_energies) / np.mean(daily_energies)
        }
        
        # Plot time series
        ax = axes[i]
        ax.plot(time_hours, power_output, linewidth=1, alpha=0.8)
        ax.set_title(f'{scenario_name}\nTotal: {sum(daily_energies):.0f} MWh, CV: {np.std(daily_energies)/np.mean(daily_energies):.3f}')
        ax.set_ylabel('Power (MW)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 7*24)
        
        if i >= 2:  # Bottom row
            ax.set_xlabel('Time (hours)')
    
    plt.tight_layout()
    plt.show()
    
    # Summary comparison
    print(f"Weather Impact Analysis:")
    for scenario, results in scenario_results.items():
        print(f"\n{scenario}:")
        print(f"- Total weekly energy: {results['total_energy']:.0f} MWh")
        print(f"- Daily variability (CV): {results['variability']:.3f}")
        print(f"- Min daily energy: {min(results['daily_energies']):.1f} MWh")
        print(f"- Max daily energy: {max(results['daily_energies']):.1f} MWh")

def main():
    """Main demonstration of solar energy challenges"""
    
    print("=== Solar Energy Challenges: Intermittency, Storage, and Weather ===\n")
    
    print("Key Challenges in Solar Energy:")
    print("1. Intermittency - Output varies with weather and time of day")
    print("2. Energy Storage - Need to store excess energy for later use")
    print("3. Weather Dependence - Performance highly dependent on conditions")
    print("4. Grid Integration - Maintaining power quality and stability")
    print("5. Economic Impact - Variable output affects system economics\n")
    
    print("Analyzing intermittency patterns...")
    time_hours, power_output, ramp_rates = analyze_intermittency()
    
    print("\nAnalyzing storage requirements...")
    storage_sizing_analysis()
    
    print("\nAnalyzing weather dependence...")
    weather_dependence_analysis()
    
    print(f"\nMitigation Strategies:")
    print(f"- Energy Storage: Batteries, pumped hydro, compressed air")
    print(f"- Grid Flexibility: Demand response, smart grid technologies")
    print(f"- Forecasting: Weather prediction for generation planning")
    print(f"- Geographic Diversity: Distributed solar reduces variability")
    print(f"- Hybrid Systems: Combining solar with other renewables")
    print(f"- Grid Infrastructure: Transmission to balance supply/demand")
    
    print(f"\nKey Insights:")
    print(f"- Solar variability increases with penetration level")
    print(f"- Storage requirements grow exponentially with penetration")
    print(f"- Weather patterns significantly impact system performance")
    print(f"- Advanced forecasting and control systems are essential")

if __name__ == "__main__":
    main()