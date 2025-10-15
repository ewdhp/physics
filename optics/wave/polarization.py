#!/usr/bin/env python3
"""
Wave Optics: Polarization
========================

This module demonstrates the polarization properties of light:
- Linear, circular, and elliptical polarization
- Malus's law and polarizer analysis
- Brewster's angle and polarization by reflection
- Birefringence and wave plates
- Optical activity and rotation
- Polarization in scattering (Rayleigh scattering)
- LCD displays and polarization applications

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
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Optional, Union
import math


class PolarizedLight:
    """Represents polarized electromagnetic radiation."""
    
    def __init__(self, amplitude_x: float, amplitude_y: float, 
                 phase_x: float = 0, phase_y: float = 0, wavelength: float = 550e-9):
        self.Ex = amplitude_x       # x-component amplitude
        self.Ey = amplitude_y       # y-component amplitude
        self.phi_x = phase_x        # x-component phase
        self.phi_y = phase_y        # y-component phase
        self.wavelength = wavelength
        self.frequency = 3e8 / wavelength
        self.k = 2 * np.pi / wavelength
    
    def electric_field(self, z: float, t: float) -> Tuple[float, float]:
        """Calculate E-field components at position z and time t."""
        phase = self.k * z - 2 * np.pi * self.frequency * t
        Ex = self.Ex * np.cos(phase + self.phi_x)
        Ey = self.Ey * np.cos(phase + self.phi_y)
        return Ex, Ey
    
    def intensity(self) -> float:
        """Calculate total intensity (∝ |E|²)."""
        return self.Ex**2 + self.Ey**2
    
    def degree_of_polarization(self) -> float:
        """Calculate degree of polarization."""
        I_max = (np.sqrt(self.Ex**2) + np.sqrt(self.Ey**2))**2
        I_min = abs(np.sqrt(self.Ex**2) - np.sqrt(self.Ey**2))**2
        if I_max == 0:
            return 0
        return (I_max - I_min) / (I_max + I_min)
    
    def polarization_ellipse_params(self) -> Tuple[float, float, float]:
        """Calculate ellipse parameters: semi-major axis, semi-minor axis, tilt angle."""
        # Phase difference
        delta = self.phi_y - self.phi_x
        
        # Stokes parameters
        S0 = self.Ex**2 + self.Ey**2
        S1 = self.Ex**2 - self.Ey**2
        S2 = 2 * self.Ex * self.Ey * np.cos(delta)
        S3 = 2 * self.Ex * self.Ey * np.sin(delta)
        
        if S0 == 0:
            return 0, 0, 0
        
        # Ellipse parameters
        a = np.sqrt((S0 + np.sqrt(S1**2 + S2**2)) / 2)  # Semi-major axis
        b = np.sqrt((S0 - np.sqrt(S1**2 + S2**2)) / 2)  # Semi-minor axis
        
        # Tilt angle
        if S1 == 0:
            psi = np.pi/4 if S2 > 0 else -np.pi/4
        else:
            psi = 0.5 * np.arctan2(S2, S1)
        
        return a, b, psi


class Polarizer:
    """Represents a linear polarizer."""
    
    def __init__(self, angle: float, transmission: float = 1.0):
        self.angle = angle              # Transmission axis angle (radians)
        self.transmission = transmission # Maximum transmission coefficient
    
    def transmit(self, light: PolarizedLight) -> PolarizedLight:
        """Apply polarizer to incident light (Malus's law)."""
        # Incident field components
        E_parallel = light.Ex * np.cos(self.angle) + light.Ey * np.sin(self.angle)
        
        # Transmitted field (only parallel component)
        Ex_out = E_parallel * np.cos(self.angle) * np.sqrt(self.transmission)
        Ey_out = E_parallel * np.sin(self.angle) * np.sqrt(self.transmission)
        
        # Phase is preserved for the transmitted component
        return PolarizedLight(Ex_out, Ey_out, light.phi_x, light.phi_y, light.wavelength)


class WavePlate:
    """Represents a birefringent wave plate."""
    
    def __init__(self, retardation: float, fast_axis_angle: float = 0):
        self.retardation = retardation      # Phase retardation in radians
        self.fast_axis = fast_axis_angle    # Fast axis orientation
    
    def transmit(self, light: PolarizedLight) -> PolarizedLight:
        """Apply wave plate retardation."""
        # Rotate to wave plate coordinate system
        cos_theta = np.cos(self.fast_axis)
        sin_theta = np.sin(self.fast_axis)
        
        # Components along fast and slow axes
        E_fast = light.Ex * cos_theta + light.Ey * sin_theta
        E_slow = -light.Ex * sin_theta + light.Ey * cos_theta
        
        # Apply retardation to slow axis
        phase_fast = light.phi_x * cos_theta + light.phi_y * sin_theta
        phase_slow = -light.phi_x * sin_theta + light.phi_y * cos_theta + self.retardation
        
        # Rotate back to lab frame
        Ex_out = E_fast * cos_theta - E_slow * sin_theta
        Ey_out = E_fast * sin_theta + E_slow * cos_theta
        
        phi_x_out = phase_fast * cos_theta - phase_slow * sin_theta
        phi_y_out = phase_fast * sin_theta + phase_slow * cos_theta
        
        return PolarizedLight(Ex_out, Ey_out, phi_x_out, phi_y_out, light.wavelength)


def demonstrate_malus_law():
    """Demonstrate Malus's law with crossed polarizers."""
    print("📐 Malus's Law")
    print("=" * 15)
    
    # Create linearly polarized light (vertical)
    incident_light = PolarizedLight(0, 1.0)  # Vertical polarization
    
    # Range of analyzer angles
    angles = np.linspace(0, 2*np.pi, 100)
    transmitted_intensities = []
    
    for angle in angles:
        analyzer = Polarizer(angle)
        transmitted = analyzer.transmit(incident_light)
        transmitted_intensities.append(transmitted.intensity())
    
    # Theoretical Malus's law: I = I₀ cos²(θ)
    theoretical = np.cos(angles)**2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Polar plot of intensity
    ax1 = plt.subplot(1, 2, 1, projection='polar')
    ax1.plot(angles, transmitted_intensities, 'b-', linewidth=3, label='Measured')
    ax1.plot(angles, theoretical, 'r--', linewidth=2, label='Theory')
    ax1.set_title('Malus\'s Law - Polar Plot')
    ax1.legend(loc='upper right')
    
    # Cartesian plot
    ax2.plot(np.degrees(angles), transmitted_intensities, 'b-', linewidth=3, label='Transmitted')
    ax2.plot(np.degrees(angles), theoretical, 'r--', linewidth=2, label='cos²(θ)')
    ax2.set_xlabel('Analyzer Angle (degrees)')
    ax2.set_ylabel('Transmitted Intensity')
    ax2.set_title('Malus\'s Law - I = I₀cos²(θ)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Mark key angles
    key_angles = [0, 45, 90, 135, 180]
    for angle_deg in key_angles:
        angle_rad = np.radians(angle_deg)
        intensity = np.cos(angle_rad)**2
        ax2.plot(angle_deg, intensity, 'ro', markersize=8)
        ax2.annotate(f'{angle_deg}°\nI={intensity:.2f}', 
                    (angle_deg, intensity), textcoords='offset points', 
                    xytext=(10, 10), ha='left')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Malus's Law Results:")
    print(f"   0°: Full transmission (I = I₀)")
    print(f"   45°: Half transmission (I = I₀/2)")
    print(f"   90°: No transmission (I = 0)")
    print(f"   Extinction ratio: {max(transmitted_intensities)/min(transmitted_intensities):.0f}:1")


def demonstrate_brewster_angle():
    """Demonstrate polarization by reflection at Brewster's angle."""
    print("\n☀️ Brewster's Angle and Polarization by Reflection")
    print("=" * 55)
    
    # Material parameters
    n1 = 1.0    # Air
    n2 = 1.5    # Glass
    
    # Calculate Brewster's angle
    brewster_angle = np.arctan(n2 / n1)
    
    # Range of incident angles
    angles = np.linspace(0, np.pi/2 - 0.01, 100)
    
    # Fresnel equations for s and p polarization
    Rs_values = []  # s-polarization reflectance
    Rp_values = []  # p-polarization reflectance
    
    for theta_i in angles:
        # Calculate transmission angle using Snell's law
        sin_theta_t = n1 * np.sin(theta_i) / n2
        if sin_theta_t > 1:  # Total internal reflection
            Rs_values.append(1.0)
            Rp_values.append(1.0)
        else:
            theta_t = np.arcsin(sin_theta_t)
            
            # Fresnel equations
            # s-polarization (perpendicular)
            rs = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / \
                 (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
            Rs = rs**2
            
            # p-polarization (parallel)
            rp = (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / \
                 (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))
            Rp = rp**2
            
            Rs_values.append(Rs)
            Rp_values.append(Rp)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Reflectance vs angle
    ax1.plot(np.degrees(angles), Rs_values, 'b-', linewidth=2, label='s-polarization')
    ax1.plot(np.degrees(angles), Rp_values, 'r-', linewidth=2, label='p-polarization')
    ax1.axvline(np.degrees(brewster_angle), color='green', linestyle='--', 
               linewidth=2, label=f'Brewster angle = {np.degrees(brewster_angle):.1f}°')
    ax1.set_xlabel('Incident Angle (degrees)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Fresnel Reflectance vs Incident Angle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Degree of polarization in reflected light
    # For unpolarized incident light
    degree_polarization = []
    for i in range(len(Rs_values)):
        Rs, Rp = Rs_values[i], Rp_values[i]
        if Rs + Rp > 0:
            dop = abs(Rs - Rp) / (Rs + Rp)
        else:
            dop = 0
        degree_polarization.append(dop)
    
    ax2.plot(np.degrees(angles), degree_polarization, 'purple', linewidth=2)
    ax2.axvline(np.degrees(brewster_angle), color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Incident Angle (degrees)')
    ax2.set_ylabel('Degree of Polarization')
    ax2.set_title('Polarization of Reflected Light')
    ax2.grid(True, alpha=0.3)
    
    # Mark complete polarization at Brewster's angle
    ax2.plot(np.degrees(brewster_angle), 1.0, 'go', markersize=10)
    ax2.annotate('Complete\npolarization', 
                (np.degrees(brewster_angle), 1.0), 
                xytext=(10, -20), textcoords='offset points', 
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Brewster Angle Analysis:")
    print(f"   Interface: Air (n={n1:.1f}) → Glass (n={n2:.1f})")
    print(f"   Brewster angle: {np.degrees(brewster_angle):.1f}°")
    print(f"   At Brewster angle: Rp = 0 (complete polarization)")
    print(f"   Critical angle: {np.degrees(np.arcsin(n1/n2)):.1f}°")
    
    print(f"\n🌅 Applications:")
    print(f"   • Polarizing sunglasses (reduce glare)")
    print(f"   • Laser windows (minimum reflection)")
    print(f"   • Photography filters")
    print(f"   • Optical polarizers")


def demonstrate_wave_plates():
    """Demonstrate quarter-wave and half-wave plates."""
    print("\n🔄 Wave Plates and Circular Polarization")
    print("=" * 40)
    
    # Create linearly polarized light at 45°
    incident = PolarizedLight(1/np.sqrt(2), 1/np.sqrt(2), 0, 0)
    
    # Different wave plates
    plates = {
        'Quarter-wave': np.pi/2,
        'Half-wave': np.pi,
        'Full-wave': 2*np.pi
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (plate_name, retardation) in enumerate(plates.items()):
        wave_plate = WavePlate(retardation, fast_axis_angle=0)
        output_light = wave_plate.transmit(incident)
        
        # Calculate polarization ellipse
        a, b, psi = output_light.polarization_ellipse_params()
        
        # Time evolution of E-field
        t = np.linspace(0, 2*np.pi, 100)
        Ex_t, Ey_t = [], []
        
        for time in t:
            Ex, Ey = output_light.electric_field(0, time)
            Ex_t.append(Ex)
            Ey_t.append(Ey)
        
        Ex_t, Ey_t = np.array(Ex_t), np.array(Ey_t)
        
        # Plot E-field trajectory
        axes[0, i].plot(Ex_t, Ey_t, 'b-', linewidth=2)
        axes[0, i].arrow(0, 0, Ex_t[0], Ey_t[0], head_width=0.05, 
                        head_length=0.05, fc='red', ec='red')
        axes[0, i].set_xlim(-1.2, 1.2)
        axes[0, i].set_ylim(-1.2, 1.2)
        axes[0, i].set_xlabel('Ex')
        axes[0, i].set_ylabel('Ey')
        axes[0, i].set_title(f'{plate_name} Plate')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_aspect('equal')
        
        # Determine polarization type
        if abs(a - b) < 0.01:
            if a > 0.01:
                pol_type = "Circular"
            else:
                pol_type = "Unpolarized"
        elif b < 0.01:
            pol_type = "Linear"
        else:
            pol_type = "Elliptical"
        
        axes[0, i].text(0.02, 0.98, pol_type, transform=axes[0, i].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow'))
        
        # Plot time evolution of components
        omega = 1  # Normalized frequency
        time_points = np.linspace(0, 4*np.pi, 1000)
        Ex_vs_t = output_light.Ex * np.cos(omega * time_points + output_light.phi_x)
        Ey_vs_t = output_light.Ey * np.cos(omega * time_points + output_light.phi_y)
        
        axes[1, i].plot(time_points, Ex_vs_t, 'r-', linewidth=2, label='Ex')
        axes[1, i].plot(time_points, Ey_vs_t, 'b-', linewidth=2, label='Ey')
        axes[1, i].set_xlabel('ωt (radians)')
        axes[1, i].set_ylabel('Electric Field')
        axes[1, i].set_title(f'Time Evolution - {plate_name}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔄 Wave Plate Effects:")
    print(f"   Quarter-wave (λ/4): Linear → Circular polarization")
    print(f"   Half-wave (λ/2): Rotates linear polarization")
    print(f"   Full-wave (λ): No net change")
    
    # Analyze specific case: 45° linear → quarter-wave → circular
    qwp = WavePlate(np.pi/2, 0)
    circular_output = qwp.transmit(incident)
    
    print(f"\n📊 45° Linear → Quarter-wave Analysis:")
    print(f"   Input: Ex = Ey = 1/√2, φx = φy = 0")
    print(f"   Output: Ex = {circular_output.Ex:.3f}, Ey = {circular_output.Ey:.3f}")
    print(f"   Phase difference: {circular_output.phi_y - circular_output.phi_x:.3f} rad")
    print(f"   Polarization: {'Circular' if abs(abs(circular_output.phi_y - circular_output.phi_x) - np.pi/2) < 0.1 else 'Other'}")


def demonstrate_optical_activity():
    """Demonstrate optical activity and polarization rotation."""
    print("\n🍯 Optical Activity and Polarization Rotation")
    print("=" * 50)
    
    # Simulate optical rotation through chiral medium
    thickness_range = np.linspace(0, 10e-3, 100)  # 0 to 10 mm
    specific_rotation = 66.5  # degrees per mm for sucrose solution
    
    # Initial linear polarization
    initial_angle = 0  # Vertical polarization
    
    # Calculate rotation angle vs thickness
    rotation_angles = specific_rotation * thickness_range * 1e3  # Convert to mm
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Rotation angle vs thickness
    ax1.plot(thickness_range * 1e3, rotation_angles, 'b-', linewidth=2)
    ax1.set_xlabel('Sample Thickness (mm)')
    ax1.set_ylabel('Rotation Angle (degrees)')
    ax1.set_title('Optical Rotation vs Thickness')
    ax1.grid(True, alpha=0.3)
    
    # Add concentration dependence annotation
    concentrations = [0.5, 1.0, 2.0]  # Different concentrations
    colors = ['green', 'blue', 'red']
    
    for i, conc in enumerate(concentrations):
        rotation_conc = specific_rotation * conc * thickness_range * 1e3
        ax1.plot(thickness_range * 1e3, rotation_conc, 
                color=colors[i], linestyle='--', linewidth=2,
                label=f'{conc:.1f}× concentration')
    
    ax1.legend()
    
    # Polarization direction visualization
    angles_to_show = np.linspace(0, 180, 13)  # Every 15 degrees
    thicknesses_to_show = angles_to_show / specific_rotation / 1e3
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    
    for i, (angle, thickness) in enumerate(zip(angles_to_show, thicknesses_to_show)):
        if thickness <= thickness_range.max():
            # Draw polarization direction
            x_end = np.cos(np.radians(angle))
            y_end = np.sin(np.radians(angle))
            
            # Color code by thickness
            color_intensity = i / len(angles_to_show)
            color = plt.cm.viridis(color_intensity)
            
            ax2.arrow(0, 0, x_end, y_end, head_width=0.05, head_length=0.05,
                     fc=color, ec=color, alpha=0.7, width=0.02)
            
            # Label every 3rd arrow
            if i % 3 == 0:
                ax2.text(x_end * 1.2, y_end * 1.2, f'{angle:.0f}°',
                        ha='center', va='center', fontsize=10)
    
    ax2.set_xlabel('Ex direction')
    ax2.set_ylabel('Ey direction')
    ax2.set_title('Polarization Direction vs Sample Thickness')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Add circular reference
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle=':', alpha=0.5)
    ax2.add_patch(circle)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🧪 Optical Activity Analysis:")
    print(f"   Specific rotation: {specific_rotation:.1f}°/mm")
    print(f"   For 1 mm thickness: {specific_rotation:.1f}° rotation")
    print(f"   For 5 mm thickness: {5 * specific_rotation:.1f}° rotation")
    
    print(f"\n🔬 Applications:")
    print(f"   • Polarimetry (concentration measurement)")
    print(f"   • Sugar content analysis")
    print(f"   • Pharmaceutical purity testing")
    print(f"   • Chiral molecule identification")
    
    # Calculate concentration from rotation measurement
    measured_rotation = 132.5  # degrees
    sample_thickness = 2e-3    # 2 mm
    calculated_conc = measured_rotation / (specific_rotation * sample_thickness * 1e3)
    
    print(f"\n📊 Example Measurement:")
    print(f"   Measured rotation: {measured_rotation:.1f}°")
    print(f"   Sample thickness: {sample_thickness*1e3:.1f} mm")
    print(f"   Calculated concentration: {calculated_conc:.2f} relative units")


def demonstrate_lcd_polarization():
    """Demonstrate LCD display polarization principles."""
    print("\n📱 LCD Display Polarization")
    print("=" * 30)
    
    # LCD operation simulation
    voltages = np.linspace(0, 5, 100)  # Applied voltage
    twist_angles = 90 * (1 - voltages / 5)  # Liquid crystal twist (degrees)
    
    # Transmission through LCD cell
    transmission = []
    
    for twist in twist_angles:
        # Simplified model: transmission depends on twist angle
        # Maximum transmission when twist = 90°, minimum when twist = 0°
        if twist == 0:
            trans = 0  # Crossed polarizers, no rotation
        else:
            # Approximate transmission for twisted nematic cell
            trans = np.sin(np.radians(twist))**2
        transmission.append(trans)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Twist angle vs voltage
    ax1.plot(voltages, twist_angles, 'b-', linewidth=3)
    ax1.set_xlabel('Applied Voltage (V)')
    ax1.set_ylabel('LC Twist Angle (degrees)')
    ax1.set_title('Liquid Crystal Orientation vs Voltage')
    ax1.grid(True, alpha=0.3)
    
    # Mark key operating points
    ax1.plot(0, 90, 'go', markersize=10, label='OFF state (90° twist)')
    ax1.plot(5, 0, 'ro', markersize=10, label='ON state (0° twist)')
    ax1.legend()
    
    # Transmission vs voltage
    ax2.plot(voltages, transmission, 'purple', linewidth=3)
    ax2.set_xlabel('Applied Voltage (V)')
    ax2.set_ylabel('Light Transmission')
    ax2.set_title('LCD Transmission Characteristics')
    ax2.grid(True, alpha=0.3)
    
    # Mark operating regions
    ax2.axvspan(0, 1, alpha=0.2, color='green', label='Bright (transmitting)')
    ax2.axvspan(4, 5, alpha=0.2, color='red', label='Dark (blocking)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📺 LCD Operating Principle:")
    print(f"   1. Backlight provides unpolarized light")
    print(f"   2. Rear polarizer creates linear polarization")
    print(f"   3. LC layer rotates polarization (voltage-dependent)")
    print(f"   4. Front polarizer analyzes rotated light")
    
    print(f"\n⚡ Voltage Control:")
    print(f"   0V: 90° twist → Maximum transmission (bright)")
    print(f"   5V: 0° twist → Minimum transmission (dark)")
    print(f"   Intermediate: Gray levels")
    
    # Calculate contrast ratio
    max_transmission = max(transmission)
    min_transmission = min(transmission) if min(transmission) > 0 else 0.001
    contrast_ratio = max_transmission / min_transmission
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Maximum transmission: {max_transmission:.3f}")
    print(f"   Minimum transmission: {min_transmission:.3f}")
    print(f"   Contrast ratio: {contrast_ratio:.0f}:1")


if __name__ == "__main__":
    print("🌈 Welcome to Polarization Explorer! 🌈")
    print("This script demonstrates the polarization properties of light.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_malus_law()
        demonstrate_brewster_angle()
        demonstrate_wave_plates()
        demonstrate_optical_activity()
        demonstrate_lcd_polarization()
        
        print(f"\n" + "="*70)
        print("🎓 Polarization Summary:")
        print("="*70)
        print("Key Concepts:")
        print("• Malus's Law: I = I₀ cos²(θ) for linear polarizers")
        print("• Brewster's Angle: θB = arctan(n₂/n₁) for complete polarization")
        print("• Wave plates: Control phase difference between orthogonal components")
        print("• Optical activity: Rotation of polarization plane in chiral media")
        
        print(f"\nPolarization States:")
        print("• Linear: Ex and Ey in phase (δ = 0, π)")
        print("• Circular: |Ex| = |Ey| and δ = ±π/2")
        print("• Elliptical: General case with arbitrary amplitudes and phase")
        
        print(f"\nApplications:")
        print("• Polarizing sunglasses and photography filters")
        print("• LCD displays and optical modulators")
        print("• Stress analysis (photoelasticity)")
        print("• Chemical analysis (polarimetry)")
        print("• 3D movie systems and optical isolation")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring polarization phenomena! ✨")