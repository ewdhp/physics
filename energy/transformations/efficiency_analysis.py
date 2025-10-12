"""
Energy Efficiency Analysis - System Performance and Optimization
Comprehensive tools for analyzing energy conversion efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class EnergySystem:
    """Represents an energy conversion system with inputs, outputs, and losses"""
    name: str
    energy_input: float
    useful_output: float
    waste_heat: float = 0.0
    other_losses: float = 0.0
    
    @property
    def efficiency(self) -> float:
        """Calculate system efficiency"""
        if self.energy_input == 0:
            return 0.0
        return self.useful_output / self.energy_input
    
    @property
    def total_losses(self) -> float:
        """Calculate total energy losses"""
        return self.waste_heat + self.other_losses

class EfficiencyAnalysis:
    """Class for energy efficiency analysis and optimization"""
    
    def __init__(self):
        self.systems = {}
        
    def carnot_efficiency(self, T_hot: float, T_cold: float) -> float:
        """
        Calculate Carnot efficiency: η = 1 - T_cold/T_hot
        
        Args:
            T_hot: Hot reservoir temperature (K)
            T_cold: Cold reservoir temperature (K)
            
        Returns:
            Carnot efficiency (0-1)
        """
        return 1 - T_cold / T_hot
    
    def heat_engine_efficiency(self, Q_hot: float, Q_cold: float) -> float:
        """
        Calculate actual heat engine efficiency: η = 1 - Q_cold/Q_hot
        
        Args:
            Q_hot: Heat input (J)
            Q_cold: Heat rejected (J)
            
        Returns:
            Actual efficiency (0-1)
        """
        return 1 - Q_cold / Q_hot
    
    def coefficient_of_performance_heat_pump(self, T_hot: float, T_cold: float) -> float:
        """
        Calculate COP for heat pump: COP = T_hot / (T_hot - T_cold)
        
        Args:
            T_hot: Hot reservoir temperature (K)
            T_cold: Cold reservoir temperature (K)
            
        Returns:
            Coefficient of Performance
        """
        return T_hot / (T_hot - T_cold)
    
    def coefficient_of_performance_refrigerator(self, T_hot: float, T_cold: float) -> float:
        """
        Calculate COP for refrigerator: COP = T_cold / (T_hot - T_cold)
        
        Args:
            T_hot: Hot reservoir temperature (K)
            T_cold: Cold reservoir temperature (K)
            
        Returns:
            Coefficient of Performance
        """
        return T_cold / (T_hot - T_cold)
    
    def combined_system_efficiency(self, systems: List[EnergySystem]) -> float:
        """
        Calculate overall efficiency of systems in series
        
        Args:
            systems: List of EnergySystem objects in series
            
        Returns:
            Combined efficiency
        """
        total_efficiency = 1.0
        for system in systems:
            total_efficiency *= system.efficiency
        return total_efficiency
    
    def parallel_system_efficiency(self, systems: List[EnergySystem], weights: List[float] = None) -> float:
        """
        Calculate weighted efficiency of parallel systems
        
        Args:
            systems: List of EnergySystem objects in parallel
            weights: Relative weights/usage of each system
            
        Returns:
            Weighted average efficiency
        """
        if weights is None:
            weights = [1.0] * len(systems)
        
        total_weight = sum(weights)
        weighted_efficiency = sum(w * s.efficiency for w, s in zip(weights, systems))
        
        return weighted_efficiency / total_weight
    
    def energy_cascade_analysis(self, primary_energy: float, conversion_steps: List[float]) -> Dict:
        """
        Analyze energy flow through cascade of conversion steps
        
        Args:
            primary_energy: Initial energy input (J)
            conversion_steps: List of efficiency values for each step
            
        Returns:
            Dictionary with energy flow analysis
        """
        energy_flow = [primary_energy]
        cumulative_losses = [0.0]
        
        current_energy = primary_energy
        total_losses = 0.0
        
        for i, efficiency in enumerate(conversion_steps):
            useful_energy = current_energy * efficiency
            losses = current_energy * (1 - efficiency)
            
            energy_flow.append(useful_energy)
            total_losses += losses
            cumulative_losses.append(total_losses)
            
            current_energy = useful_energy
        
        overall_efficiency = current_energy / primary_energy
        
        return {
            'energy_flow': energy_flow,
            'cumulative_losses': cumulative_losses,
            'step_efficiencies': conversion_steps,
            'overall_efficiency': overall_efficiency,
            'final_useful_energy': current_energy,
            'total_losses': total_losses
        }
    
    def second_law_efficiency(self, actual_work: float, reversible_work: float) -> float:
        """
        Calculate second law (exergy) efficiency
        
        Args:
            actual_work: Actual work output (J)
            reversible_work: Maximum theoretical work (J)
            
        Returns:
            Second law efficiency (0-1)
        """
        return actual_work / reversible_work
    
    def compare_energy_systems(self, systems_data: Dict) -> Dict:
        """
        Compare different energy systems
        
        Args:
            systems_data: Dictionary with system names and performance data
            
        Returns:
            Comparison analysis
        """
        comparison = {}
        
        for name, data in systems_data.items():
            system = EnergySystem(**data)
            comparison[name] = {
                'efficiency': system.efficiency,
                'energy_input': system.energy_input,
                'useful_output': system.useful_output,
                'losses': system.total_losses,
                'loss_fraction': system.total_losses / system.energy_input if system.energy_input > 0 else 0
            }
        
        # Rank by efficiency
        ranked_systems = sorted(comparison.items(), key=lambda x: x[1]['efficiency'], reverse=True)
        comparison['ranking'] = ranked_systems
        
        return comparison

def demonstrate_efficiency_analysis():
    """Demonstrate energy efficiency analysis concepts"""
    
    analyzer = EfficiencyAnalysis()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Energy Efficiency Analysis', fontsize=16)
    
    # 1. Carnot efficiency vs temperature ratio
    T_cold_values = [200, 250, 300, 350]  # K
    T_hot_range = np.linspace(400, 1000, 100)  # K
    
    for T_cold in T_cold_values:
        eta_carnot = [analyzer.carnot_efficiency(T_hot, T_cold) for T_hot in T_hot_range]
        axes[0,0].plot(T_hot_range, np.array(eta_carnot) * 100, 
                      label=f'T_cold = {T_cold} K', linewidth=2)
    
    axes[0,0].set_title('Carnot Efficiency vs Hot Temperature')
    axes[0,0].set_xlabel('Hot Temperature (K)')
    axes[0,0].set_ylabel('Carnot Efficiency (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Add practical engine efficiency ranges
    axes[0,0].axhspan(25, 35, alpha=0.3, color='red', label='ICE Range')
    axes[0,0].axhspan(35, 45, alpha=0.3, color='blue', label='Gas Turbine Range')
    
    # 2. Energy cascade analysis - comparing different power systems
    primary_energy = 1000  # MJ
    
    systems = {
        'Coal Plant': [0.85, 0.38, 0.98],  # Combustion, Thermal-to-mechanical, Generator
        'Gas Turbine': [0.92, 0.42, 0.98],  # Combustion, Thermal-to-mechanical, Generator
        'Solar PV': [1.0, 0.22],  # Direct conversion, DC-AC conversion
        'Hydroelectric': [1.0, 0.90, 0.98],  # Potential-to-kinetic, Turbine, Generator
        'Nuclear': [0.95, 0.33, 0.98]  # Fission-to-thermal, Thermal cycle, Generator
    }
    
    system_colors = ['brown', 'orange', 'gold', 'blue', 'purple']
    
    for i, (system_name, efficiencies) in enumerate(systems.items()):
        analysis = analyzer.energy_cascade_analysis(primary_energy, efficiencies)
        steps = range(len(analysis['energy_flow']))
        
        axes[0,1].plot(steps, analysis['energy_flow'], 'o-', 
                      color=system_colors[i], label=system_name, linewidth=2, markersize=6)
    
    axes[0,1].set_title('Energy Flow Through Conversion Systems')
    axes[0,1].set_xlabel('Conversion Step')
    axes[0,1].set_ylabel('Energy (MJ)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. System efficiency comparison
    system_names = list(systems.keys())
    overall_efficiencies = []
    
    for efficiencies in systems.values():
        analysis = analyzer.energy_cascade_analysis(primary_energy, efficiencies)
        overall_efficiencies.append(analysis['overall_efficiency'] * 100)
    
    bars = axes[1,0].bar(system_names, overall_efficiencies, 
                        color=system_colors, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Overall System Efficiencies')
    axes[1,0].set_ylabel('Efficiency (%)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, efficiency in zip(bars, overall_efficiencies):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{efficiency:.1f}%', ha='center', va='bottom')
    
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # 4. Heat pump COP vs temperature difference
    T_cold_hp = 280  # K (7°C)
    T_hot_range_hp = np.linspace(290, 350, 100)  # K
    
    cop_carnot = [analyzer.coefficient_of_performance_heat_pump(T_hot, T_cold_hp) 
                  for T_hot in T_hot_range_hp]
    cop_realistic = np.array(cop_carnot) * 0.5  # Realistic COP is ~50% of Carnot
    
    axes[1,1].plot(T_hot_range_hp - 273.15, cop_carnot, 'r-', 
                  label='Carnot COP', linewidth=2)
    axes[1,1].plot(T_hot_range_hp - 273.15, cop_realistic, 'b-', 
                  label='Realistic COP', linewidth=2)
    
    axes[1,1].set_title('Heat Pump Coefficient of Performance')
    axes[1,1].set_xlabel('Hot Side Temperature (°C)')
    axes[1,1].set_ylabel('COP')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print efficiency analysis
    print("Energy Efficiency Analysis")
    print("=" * 40)
    
    # Coal power plant detailed analysis
    coal_analysis = analyzer.energy_cascade_analysis(1000, [0.85, 0.38, 0.98])
    print(f"Coal Power Plant Analysis:")
    print(f"  Primary energy: {coal_analysis['energy_flow'][0]:.0f} MJ")
    print(f"  After combustion: {coal_analysis['energy_flow'][1]:.0f} MJ (85% efficient)")
    print(f"  After thermal cycle: {coal_analysis['energy_flow'][2]:.0f} MJ (38% efficient)")
    print(f"  Final electricity: {coal_analysis['energy_flow'][3]:.0f} MJ (98% efficient)")
    print(f"  Overall efficiency: {coal_analysis['overall_efficiency']*100:.1f}%")
    
    # Heat engine vs heat pump comparison
    T_hot, T_cold = 600, 300  # K
    eta_carnot = analyzer.carnot_efficiency(T_hot, T_cold)
    cop_hp = analyzer.coefficient_of_performance_heat_pump(T_hot, T_cold)
    
    print(f"\nThermodynamic Cycle Comparison (600K → 300K):")
    print(f"  Heat engine (Carnot): {eta_carnot*100:.1f}% efficiency")
    print(f"  Heat pump (Carnot): {cop_hp:.1f} COP")
    print(f"  Note: COP > 1 means heat pump delivers more thermal energy than electrical input")

def interactive_efficiency_calculator():
    """Interactive efficiency calculator and system designer"""
    
    print("Energy Efficiency Calculator")
    print("=" * 30)
    
    analyzer = EfficiencyAnalysis()
    
    while True:
        print("\nChoose analysis type:")
        print("1. Carnot Efficiency")
        print("2. Heat Engine Analysis")
        print("3. Heat Pump/Refrigerator COP")
        print("4. Energy Cascade Analysis")
        print("5. System Comparison")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            T_hot = float(input("Enter hot reservoir temperature (K): "))
            T_cold = float(input("Enter cold reservoir temperature (K): "))
            
            eta_carnot = analyzer.carnot_efficiency(T_hot, T_cold)
            print(f"Carnot efficiency: {eta_carnot*100:.1f}%")
            print(f"Maximum work per unit heat input: {eta_carnot:.3f}")
            
        elif choice == '2':
            Q_hot = float(input("Enter heat input (J or MJ): "))
            Q_cold = float(input("Enter heat rejected (J or MJ): "))
            
            eta_actual = analyzer.heat_engine_efficiency(Q_hot, Q_cold)
            work_output = Q_hot - Q_cold
            
            print(f"Actual efficiency: {eta_actual*100:.1f}%")
            print(f"Work output: {work_output:.1f}")
            print(f"Heat utilization: {(1-Q_cold/Q_hot)*100:.1f}%")
            
        elif choice == '3':
            device_type = input("Enter device type (heat_pump/refrigerator): ").lower()
            T_hot = float(input("Enter hot side temperature (K): "))
            T_cold = float(input("Enter cold side temperature (K): "))
            
            if device_type == 'heat_pump':
                cop = analyzer.coefficient_of_performance_heat_pump(T_hot, T_cold)
                print(f"Heat pump COP: {cop:.2f}")
                print(f"Heat delivered per unit work: {cop:.2f}")
            elif device_type == 'refrigerator':
                cop = analyzer.coefficient_of_performance_refrigerator(T_hot, T_cold)
                print(f"Refrigerator COP: {cop:.2f}")
                print(f"Heat removed per unit work: {cop:.2f}")
            else:
                print("Invalid device type")
                
        elif choice == '4':
            primary_energy = float(input("Enter primary energy input: "))
            num_steps = int(input("Enter number of conversion steps: "))
            
            efficiencies = []
            for i in range(num_steps):
                eff = float(input(f"Enter efficiency for step {i+1} (0-1): "))
                efficiencies.append(eff)
            
            analysis = analyzer.energy_cascade_analysis(primary_energy, efficiencies)
            
            print(f"\nEnergy Cascade Analysis:")
            print(f"Primary energy: {analysis['energy_flow'][0]:.1f}")
            for i, (energy, eff) in enumerate(zip(analysis['energy_flow'][1:], efficiencies)):
                print(f"After step {i+1}: {energy:.1f} (η = {eff*100:.1f}%)")
            print(f"Overall efficiency: {analysis['overall_efficiency']*100:.1f}%")
            print(f"Total losses: {analysis['total_losses']:.1f}")
            
        elif choice == '5':
            print("System Comparison Tool")
            num_systems = int(input("Enter number of systems to compare: "))
            
            systems_data = {}
            for i in range(num_systems):
                print(f"\nSystem {i+1}:")
                name = input("System name: ")
                energy_input = float(input("Energy input: "))
                useful_output = float(input("Useful output: "))
                waste_heat = float(input("Waste heat (optional, default 0): ") or "0")
                
                systems_data[name] = {
                    'name': name,
                    'energy_input': energy_input,
                    'useful_output': useful_output,
                    'waste_heat': waste_heat,
                    'other_losses': energy_input - useful_output - waste_heat
                }
            
            comparison = analyzer.compare_energy_systems(systems_data)
            
            print("\nSystem Comparison Results:")
            print("-" * 50)
            for rank, (name, data) in enumerate(comparison['ranking'], 1):
                print(f"{rank}. {name}")
                print(f"   Efficiency: {data['efficiency']*100:.1f}%")
                print(f"   Losses: {data['loss_fraction']*100:.1f}% of input")
                
        elif choice == '6':
            print("Thanks for using the efficiency calculator!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Energy Efficiency Analysis")
    print("=" * 30)
    
    # Run demonstrations
    demonstrate_efficiency_analysis()
    
    # Interactive calculator
    use_calculator = input("\nWould you like to use the interactive calculator? (y/n): ").lower()
    if use_calculator == 'y':
        interactive_efficiency_calculator()