#!/usr/bin/env python3
"""
Work-Energy Principle Demonstration
===================================

This script demonstrates the fundamental work-energy principle:
W = F · d · cos(θ)

Where:
- W = Work done (Joules)
- F = Applied force (Newtons)  
- d = Distance moved (meters)
- θ = Angle between force and displacement vectors

The work-energy theorem states: W_net = ΔKE = KE_final - KE_initial

Author: Physics Education Project
Date: October 2025
"""

import numpy as np
import matplotlib
# Set backend for GUI display
try:
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')
    print("⚠️  Using Agg backend - plots will be saved but not displayed")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import math

class WorkEnergyCalculator:
    """
    Comprehensive calculator for work-energy principle problems
    """
    
    def __init__(self):
        self.g = 9.81  # gravitational acceleration (m/s²)
    
    def calculate_work(self, force: float, distance: float, angle_degrees: float) -> Dict:
        """
        Calculate work done by a force: W = F·d·cos(θ)
        
        Args:
            force: Magnitude of force in Newtons
            distance: Distance moved in meters  
            angle_degrees: Angle between force and displacement in degrees
            
        Returns:
            Dictionary with comprehensive work calculation details
        """
        angle_radians = np.radians(angle_degrees)
        cos_theta = np.cos(angle_radians)
        work = force * distance * cos_theta
        force_parallel = force * cos_theta
        force_perpendicular = force * np.sin(angle_radians)
        
        return {
            'work': work,
            'force': force,
            'distance': distance, 
            'angle_degrees': angle_degrees,
            'angle_radians': angle_radians,
            'cos_theta': cos_theta,
            'sin_theta': np.sin(angle_radians),
            'force_parallel': force_parallel,
            'force_perpendicular': force_perpendicular,
            'work_type': self._classify_work(work)
        }
    
    def _classify_work(self, work: float) -> str:
        """Classify the type of work done"""
        if work > 0:
            return "Positive (Energy added to system)"
        elif work < 0:
            return "Negative (Energy removed from system)"
        else:
            return "Zero (Force perpendicular to motion)"
    
    def kinetic_energy(self, mass: float, velocity: float) -> float:
        """Calculate kinetic energy: KE = (1/2)mv²"""
        return 0.5 * mass * velocity**2
    
    def work_energy_theorem(self, mass: float, initial_velocity: float, 
                          force: float, distance: float, angle_degrees: float) -> Dict:
        """
        Apply work-energy theorem: W_net = ΔKE
        """
        # Calculate initial kinetic energy
        ke_initial = self.kinetic_energy(mass, initial_velocity)
        
        # Calculate work done
        work_result = self.calculate_work(force, distance, angle_degrees)
        work_done = work_result['work']
        
        # Apply work-energy theorem
        ke_final = ke_initial + work_done
        
        # Calculate final velocity (handle negative KE case)
        if ke_final >= 0:
            final_velocity = math.sqrt(2 * ke_final / mass)
        else:
            final_velocity = 0  # Object comes to rest
            ke_final = 0
        
        delta_ke = ke_final - ke_initial
        
        return {
            'mass': mass,
            'initial_velocity': initial_velocity,
            'final_velocity': final_velocity,
            'ke_initial': ke_initial,
            'ke_final': ke_final,
            'delta_ke': delta_ke,
            'work_done': work_done,
            'work_result': work_result,
            'energy_conserved': abs(work_done - delta_ke) < 1e-10
        }

class WorkEnergyVisualizer:
    """
    Creates visualizations for work-energy principle concepts
    """
    
    def __init__(self):
        self.calculator = WorkEnergyCalculator()
    
    def plot_work_vs_angle(self, force: float = 100, distance: float = 10):
        """
        Create comprehensive visualization of how work varies with angle
        """
        angles = np.linspace(0, 360, 361)
        work_values = []
        force_parallel = []
        force_perpendicular = []
        
        for angle in angles:
            work_result = self.calculator.calculate_work(force, distance, angle)
            work_values.append(work_result['work'])
            force_parallel.append(work_result['force_parallel'])
            force_perpendicular.append(work_result['force_perpendicular'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Work vs Angle
        ax1.plot(angles, work_values, 'b-', linewidth=3, label=f'W = F·d·cos(θ)')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Angle θ (degrees)', fontsize=12)
        ax1.set_ylabel('Work Done (J)', fontsize=12)
        ax1.set_title(f'Work vs Angle\nF = {force} N, d = {distance} m', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Highlight key angles
        key_angles = [0, 90, 180, 270]
        key_works = [self.calculator.calculate_work(force, distance, angle)['work'] for angle in key_angles]
        key_labels = ['Max +Work', 'Zero Work', 'Max -Work', 'Zero Work']
        colors = ['green', 'orange', 'red', 'orange']
        
        for angle, work, label, color in zip(key_angles, key_works, key_labels, colors):
            ax1.scatter([angle], [work], color=color, s=150, zorder=5, edgecolor='black', linewidth=2)
            ax1.annotate(f'{label}\nθ={angle}°, W={work:.0f}J', 
                        xy=(angle, work), xytext=(10, 20),
                        textcoords='offset points', ha='left',
                        bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black'))
        
        # Plot 2: Force Components
        ax2.plot(angles, force_parallel, 'r-', linewidth=2, label='F∥ = F·cos(θ) (Parallel)')
        ax2.plot(angles, force_perpendicular, 'g-', linewidth=2, label='F⊥ = F·sin(θ) (Perpendicular)')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Angle θ (degrees)', fontsize=12)
        ax2.set_ylabel('Force Component (N)', fontsize=12)
        ax2.set_title('Force Components', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        # Plot 3: Vector Diagrams for Key Angles
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Force Vector Diagrams', fontsize=14, weight='bold')
        
        # Draw vector diagrams for different angles
        demo_angles = [0, 45, 90, 135]
        colors_vec = ['blue', 'green', 'orange', 'red']
        
        for i, (angle, color) in enumerate(zip(demo_angles, colors_vec)):
            angle_rad = np.radians(angle)
            
            # Displacement vector (always horizontal)
            dx = 0.8
            start_x = -1.2 + i * 0.6
            ax3.arrow(start_x, -1.0 + i * 0.3, dx, 0, head_width=0.05, head_length=0.05, 
                     fc='black', ec='black', linewidth=2)
            ax3.text(start_x + dx/2, -1.0 + i * 0.3 - 0.1, 'd', ha='center', fontsize=10, weight='bold')
            
            # Force vector
            fx = dx * np.cos(angle_rad)
            fy = dx * np.sin(angle_rad)
            ax3.arrow(start_x, -1.0 + i * 0.3, fx, fy, head_width=0.05, head_length=0.05, 
                     fc=color, ec=color, linewidth=2, alpha=0.8)
            ax3.text(start_x + fx/2 - 0.1, -1.0 + i * 0.3 + fy/2 + 0.1, 'F', ha='center', 
                    fontsize=10, weight='bold', color=color)
            
            # Angle arc
            if angle > 0:
                arc_angles = np.linspace(0, angle_rad, 20)
                arc_x = start_x + 0.2 * np.cos(arc_angles)
                arc_y = -1.0 + i * 0.3 + 0.2 * np.sin(arc_angles)
                ax3.plot(arc_x, arc_y, color=color, linewidth=2)
                ax3.text(start_x + 0.25, -1.0 + i * 0.3 + 0.1, f'{angle}°', 
                        fontsize=9, color=color, weight='bold')
        
        # Plot 4: Work-Energy Example
        # Show a specific example calculation
        example_mass = 2.0
        example_v_initial = 3.0
        example_force = 50.0
        example_distance = 5.0
        example_angle = 30.0
        
        result = self.calculator.work_energy_theorem(example_mass, example_v_initial,
                                                   example_force, example_distance, example_angle)
        
        # Bar chart showing energy transformation
        categories = ['Initial KE', 'Work Done', 'Final KE']
        values = [result['ke_initial'], result['work_done'], result['ke_final']]
        colors_bar = ['skyblue', 'lightgreen', 'salmon']
        
        bars = ax4.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Energy (J)', fontsize=12)
        ax4.set_title('Work-Energy Theorem Example\n' + 
                     f'Mass: {example_mass}kg, Force: {example_force}N @ {example_angle}°', 
                     fontsize=12, weight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.1f}J', ha='center', va='bottom', fontsize=11, weight='bold')
        
        # Add calculation details
        details = (f'Initial: v₀ = {result["initial_velocity"]} m/s\n'
                  f'Final: vf = {result["final_velocity"]:.2f} m/s\n'
                  f'Work: W = F·d·cos({example_angle}°) = {result["work_done"]:.1f} J\n'
                  f'ΔKE = {result["delta_ke"]:.1f} J')
        
        ax4.text(0.02, 0.98, details, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def animate_work_calculation(self, force: float, distance: float, angle_degrees: float):
        """
        Create a step-by-step visual animation of work calculation
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate values
        work_result = self.calculator.calculate_work(force, distance, angle_degrees)
        
        # Set up the plot
        ax.set_xlim(-1, 6)
        ax.set_ylim(-2, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Work Calculation: W = F·d·cos(θ)\nF = {force}N, d = {distance}m, θ = {angle_degrees}°', 
                    fontsize=14, weight='bold')
        
        # Draw displacement vector
        ax.arrow(0, 0, distance, 0, head_width=0.1, head_length=0.2, 
                fc='blue', ec='blue', linewidth=3, label=f'Displacement d = {distance}m')
        
        # Draw force vector
        angle_rad = np.radians(angle_degrees)
        fx = force * np.cos(angle_rad) / 10  # Scale for visualization
        fy = force * np.sin(angle_rad) / 10
        ax.arrow(0, 0, fx, fy, head_width=0.1, head_length=0.2, 
                fc='red', ec='red', linewidth=3, label=f'Force F = {force}N')
        
        # Draw force components
        ax.arrow(0, 0, fx, 0, head_width=0.08, head_length=0.15, 
                fc='green', ec='green', linewidth=2, linestyle='--', alpha=0.8,
                label=f'F∥ = {work_result["force_parallel"]:.1f}N')
        
        if fy != 0:
            ax.arrow(fx, 0, 0, fy, head_width=0.08, head_length=0.1, 
                    fc='orange', ec='orange', linewidth=2, linestyle='--', alpha=0.8,
                    label=f'F⊥ = {work_result["force_perpendicular"]:.1f}N')
        
        # Draw angle arc
        if angle_degrees != 0:
            arc_angles = np.linspace(0, angle_rad, 30)
            arc_x = 0.5 * np.cos(arc_angles)
            arc_y = 0.5 * np.sin(arc_angles)
            ax.plot(arc_x, arc_y, 'purple', linewidth=2)
            ax.text(0.6, 0.2, f'θ = {angle_degrees}°', fontsize=12, color='purple', weight='bold')
        
        # Add calculation steps
        calc_steps = [
            f"Step 1: Identify given values",
            f"   Force F = {force} N",
            f"   Distance d = {distance} m", 
            f"   Angle θ = {angle_degrees}°",
            f"",
            f"Step 2: Calculate cos(θ)",
            f"   cos({angle_degrees}°) = {work_result['cos_theta']:.3f}",
            f"",
            f"Step 3: Apply work formula",
            f"   W = F × d × cos(θ)",
            f"   W = {force} × {distance} × {work_result['cos_theta']:.3f}",
            f"   W = {work_result['work']:.1f} J",
            f"",
            f"Result: {work_result['work_type']}"
        ]
        
        calc_text = '\n'.join(calc_steps)
        ax.text(distance + 0.5, 2, calc_text, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.legend(loc='upper right')
        
        # Remove axis labels for cleaner look
        ax.set_xlabel('Position (m)', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        
        plt.tight_layout()
        plt.show()

def demonstrate_work_energy_examples():
    """
    Run comprehensive work-energy principle demonstrations
    """
    calc = WorkEnergyCalculator()
    viz = WorkEnergyVisualizer()
    
    print("🔧 WORK-ENERGY PRINCIPLE DEMONSTRATIONS")
    print("=" * 60)
    
    print("\n📚 The Work-Energy Principle:")
    print("   W = F · d · cos(θ)")
    print("   Where:")
    print("   • W = Work done (Joules)")
    print("   • F = Applied force (Newtons)")
    print("   • d = Distance moved (meters)")
    print("   • θ = Angle between force and displacement vectors")
    print("\n   Work-Energy Theorem: W_net = ΔKE = KE_final - KE_initial")
    
    # Example calculations
    examples = [
        {"force": 50, "distance": 10, "angle": 0, "name": "Force in direction of motion"},
        {"force": 30, "distance": 8, "angle": 60, "name": "Force at 60° angle"},
        {"force": 20, "distance": 5, "angle": 90, "name": "Force perpendicular to motion"},
        {"force": 40, "distance": 6, "angle": 135, "name": "Force opposing motion"},
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n📖 Example {i}: {ex['name']}")
        print("-" * 50)
        
        result = calc.calculate_work(ex['force'], ex['distance'], ex['angle'])
        
        print(f"Given:")
        print(f"   Force F = {result['force']} N")
        print(f"   Distance d = {result['distance']} m")
        print(f"   Angle θ = {result['angle_degrees']}°")
        
        print(f"\nCalculation:")
        print(f"   cos({result['angle_degrees']}°) = {result['cos_theta']:.3f}")
        print(f"   W = F × d × cos(θ)")
        print(f"   W = {result['force']} × {result['distance']} × {result['cos_theta']:.3f}")
        print(f"   W = {result['work']:.1f} J")
        
        print(f"\nForce Components:")
        print(f"   Parallel to motion: F∥ = {result['force_parallel']:.1f} N")
        print(f"   Perpendicular to motion: F⊥ = {result['force_perpendicular']:.1f} N")
        
        print(f"\nResult: {result['work_type']}")
        
        if i == 1:  # Show detailed visualization for first example
            print("\n📊 Generating detailed visualization...")
            viz.animate_work_calculation(ex['force'], ex['distance'], ex['angle'])
    
    # Work-Energy Theorem Examples
    print(f"\n🔋 WORK-ENERGY THEOREM EXAMPLES")
    print("=" * 50)
    
    theorem_examples = [
        {"mass": 2.0, "v_initial": 5.0, "force": 20.0, "distance": 3.0, "angle": 0},
        {"mass": 1.5, "v_initial": 8.0, "force": 15.0, "distance": 4.0, "angle": 45},
        {"mass": 3.0, "v_initial": 10.0, "force": -25.0, "distance": 2.0, "angle": 180},
    ]
    
    for i, ex in enumerate(theorem_examples, 1):
        print(f"\n🧮 Work-Energy Example {i}:")
        print("-" * 30)
        
        result = calc.work_energy_theorem(ex['mass'], ex['v_initial'], 
                                        ex['force'], ex['distance'], ex['angle'])
        
        print(f"Initial State:")
        print(f"   Mass = {result['mass']} kg")
        print(f"   Initial velocity = {result['initial_velocity']} m/s")
        print(f"   Initial KE = {result['ke_initial']:.1f} J")
        
        print(f"\nApplied Force:")
        print(f"   Force = {ex['force']} N")
        print(f"   Distance = {ex['distance']} m")
        print(f"   Angle = {ex['angle']}°")
        print(f"   Work done = {result['work_done']:.1f} J")
        
        print(f"\nFinal State:")
        print(f"   Final velocity = {result['final_velocity']:.2f} m/s")
        print(f"   Final KE = {result['ke_final']:.1f} J")
        print(f"   Change in KE = {result['delta_ke']:.1f} J")
        
        print(f"\nVerification: W = ΔKE ✓" if result['energy_conserved'] else "⚠️ Energy calculation error")
    
    # Generate comprehensive visualization
    print(f"\n📊 Generating comprehensive work vs angle analysis...")
    viz.plot_work_vs_angle(force=100, distance=5)

def interactive_work_calculator():
    """
    Interactive calculator for work-energy problems
    """
    calc = WorkEnergyCalculator()
    viz = WorkEnergyVisualizer()
    
    print(f"\n🧮 Interactive Work-Energy Calculator")
    print("=" * 45)
    
    while True:
        print(f"\nChoose calculation type:")
        print("1. Calculate work done by a force")
        print("2. Work-Energy theorem analysis")
        print("3. Visualize work vs angle")
        print("4. Animate specific calculation")
        print("5. Exit calculator")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            try:
                print(f"\n📐 Work Calculation: W = F·d·cos(θ)")
                force = float(input("Enter force magnitude (N): "))
                distance = float(input("Enter distance moved (m): "))
                angle = float(input("Enter angle between force and displacement (degrees): "))
                
                result = calc.calculate_work(force, distance, angle)
                
                print(f"\n📊 CALCULATION RESULTS:")
                print(f"{'='*40}")
                print(f"Given values:")
                print(f"   Force F = {result['force']} N")
                print(f"   Distance d = {result['distance']} m") 
                print(f"   Angle θ = {result['angle_degrees']}° = {result['angle_radians']:.3f} rad")
                
                print(f"\nTrigonometric values:")
                print(f"   cos({result['angle_degrees']}°) = {result['cos_theta']:.4f}")
                print(f"   sin({result['angle_degrees']}°) = {result['sin_theta']:.4f}")
                
                print(f"\nForce components:")
                print(f"   Parallel component: F∥ = F·cos(θ) = {result['force_parallel']:.2f} N")
                print(f"   Perpendicular component: F⊥ = F·sin(θ) = {result['force_perpendicular']:.2f} N")
                
                print(f"\nWork calculation:")
                print(f"   W = F × d × cos(θ)")
                print(f"   W = {result['force']} × {result['distance']} × {result['cos_theta']:.4f}")
                print(f"   W = {result['work']:.2f} J")
                
                print(f"\nInterpretation:")
                print(f"   {result['work_type']}")
                
                if result['work'] > 0:
                    print(f"   ✅ The force adds energy to the object")
                elif result['work'] < 0:
                    print(f"   ❌ The force removes energy from the object")
                else:
                    print(f"   ⚪ No energy transfer (force perpendicular to motion)")
                    
            except ValueError:
                print("❌ Please enter valid numbers.")
        
        elif choice == '2':
            try:
                print(f"\n🔋 Work-Energy Theorem: W_net = ΔKE")
                mass = float(input("Enter object mass (kg): "))
                v_initial = float(input("Enter initial velocity (m/s): "))
                force = float(input("Enter applied force (N): "))
                distance = float(input("Enter distance moved (m): "))
                angle = float(input("Enter angle between force and displacement (degrees): "))
                
                result = calc.work_energy_theorem(mass, v_initial, force, distance, angle)
                
                print(f"\n📊 WORK-ENERGY ANALYSIS:")
                print(f"{'='*45}")
                
                print(f"Initial state:")
                print(f"   Mass m = {result['mass']} kg")
                print(f"   Initial velocity v₀ = {result['initial_velocity']} m/s")
                print(f"   Initial kinetic energy KE₀ = ½mv₀² = {result['ke_initial']:.2f} J")
                
                work_details = result['work_result']
                print(f"\nForce analysis:")
                print(f"   Applied force F = {work_details['force']} N")
                print(f"   Distance moved d = {work_details['distance']} m")
                print(f"   Force angle θ = {work_details['angle_degrees']}°")
                print(f"   Effective force F∥ = {work_details['force_parallel']:.2f} N")
                
                print(f"\nWork calculation:")
                print(f"   Work done W = F·d·cos(θ) = {result['work_done']:.2f} J")
                print(f"   Work type: {work_details['work_type']}")
                
                print(f"\nFinal state (using work-energy theorem):")
                print(f"   Final kinetic energy KEf = KE₀ + W = {result['ke_final']:.2f} J")
                print(f"   Final velocity vf = √(2·KEf/m) = {result['final_velocity']:.2f} m/s")
                
                print(f"\nEnergy analysis:")
                print(f"   Change in kinetic energy ΔKE = {result['delta_ke']:.2f} J")
                print(f"   Work done W = {result['work_done']:.2f} J")
                print(f"   Energy conservation check: {'✅ Verified' if result['energy_conserved'] else '❌ Error'}")
                
            except ValueError:
                print("❌ Please enter valid numbers.")
        
        elif choice == '3':
            try:
                print(f"\n📈 Work vs Angle Visualization")
                force = float(input("Enter force magnitude (N) [default: 100]: ") or "100")
                distance = float(input("Enter distance (m) [default: 10]: ") or "10")
                print("Generating comprehensive work analysis...")
                viz.plot_work_vs_angle(force, distance)
            except ValueError:
                print("❌ Please enter valid numbers.")
        
        elif choice == '4':
            try:
                print(f"\n🎬 Animated Work Calculation")
                force = float(input("Enter force magnitude (N): "))
                distance = float(input("Enter distance (m): "))
                angle = float(input("Enter angle (degrees): "))
                print("Creating step-by-step visualization...")
                viz.animate_work_calculation(force, distance, angle)
            except ValueError:
                print("❌ Please enter valid numbers.")
        
        elif choice == '5':
            print("👋 Thank you for using the Work-Energy Calculator!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🔧 WORK-ENERGY PRINCIPLE DEMONSTRATION")
    print("W = F · d · cos(θ)")
    print("=" * 60)
    
    print("\nSelect mode:")
    print("1. Comprehensive demonstrations and examples")
    print("2. Interactive calculator")
    print("3. Both")
    
    mode = input("Enter your choice (1-3): ")
    
    if mode == '1':
        demonstrate_work_energy_examples()
    elif mode == '2':  
        interactive_work_calculator()
    elif mode == '3':
        demonstrate_work_energy_examples()
        interactive_work_calculator()
    else:
        print("Running comprehensive demonstration...")
        demonstrate_work_energy_examples()