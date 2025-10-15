#!/usr/bin/env python3
"""
Wave-Particle Duality of Light
==============================

This module demonstrates the fundamental quantum mechanical principle that light exhibits
both wave and particle properties depending on the experimental setup and observation method.

WAVE NATURE demonstrates:
1. Interference - Constructive and destructive wave patterns
2. Diffraction - Wave bending around obstacles
3. Reflection - Wave behavior at interfaces  
4. Refraction - Wave bending through media

PARTICLE NATURE demonstrates:
5. Photoelectric Effect - Photon energy threshold behavior
6. Photon Energy Distribution - Discrete energy packets
7. Compton Scattering - Photon-electron collisions
8. Quantum Energy Levels - Discrete absorption/emission

Author: Physics Education Project
Date: October 2025
"""

import numpy as np
import matplotlib
# Set backend for GUI display - use TkAgg if available, fallback to Agg
try:
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')
    print("⚠️  Using Agg backend - plots will be saved but not displayed")
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import math


class WaveParticleDuality:
    """Main class for wave-particle duality demonstrations."""
    
    def __init__(self):
        self.h = 6.626e-34  # Planck's constant (J⋅s)
        self.c = 2.998e8    # Speed of light (m/s)
        self.e = 1.602e-19  # Elementary charge (C)
        self.me = 9.109e-31 # Electron mass (kg)
        
    def wave_interference(self, x: np.ndarray, wavelength: float, 
                         phase_diff: float = 0) -> np.ndarray:
        """
        Calculate wave interference pattern.
        
        Args:
            x: Position array
            wavelength: Wavelength of light (m)
            phase_diff: Phase difference between waves (radians)
            
        Returns:
            Interference amplitude
        """
        k = 2 * np.pi / wavelength  # Wave number
        wave1 = np.sin(k * x)
        wave2 = np.sin(k * x + phase_diff)
        return wave1 + wave2
    
    def diffraction_pattern(self, theta: np.ndarray, wavelength: float, 
                           slit_width: float) -> np.ndarray:
        """
        Calculate single-slit diffraction pattern.
        
        Args:
            theta: Angle array (radians)
            wavelength: Wavelength of light (m)
            slit_width: Width of the slit (m)
            
        Returns:
            Diffraction intensity pattern
        """
        beta = (np.pi * slit_width * np.sin(theta)) / wavelength
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            intensity = np.where(beta == 0, 1.0, (np.sin(beta) / beta)**2)
        return intensity
    
    def photoelectric_effect(self, frequency: np.ndarray, work_function: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate photoelectric effect parameters.
        
        Args:
            frequency: Light frequency array (Hz)
            work_function: Work function of material (eV)
            
        Returns:
            Tuple of (kinetic_energy, photocurrent)
        """
        photon_energy = self.h * frequency / self.e  # Convert to eV
        kinetic_energy = np.maximum(0, photon_energy - work_function)
        # Photocurrent proportional to sqrt of excess energy, only when E > work function
        excess_energy = np.maximum(0, photon_energy - work_function)
        photocurrent = np.where(photon_energy >= work_function, 
                               np.sqrt(excess_energy), 0)
        return kinetic_energy, photocurrent
    
    def photon_energy(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate photon energy for given wavelengths.
        
        Args:
            wavelength: Wavelength array (m)
            
        Returns:
            Photon energy in eV
        """
        frequency = self.c / wavelength
        return self.h * frequency / self.e
    
    def compton_scattering(self, theta: np.ndarray, initial_wavelength: float) -> np.ndarray:
        """
        Calculate wavelength shift in Compton scattering.
        
        Args:
            theta: Scattering angle array (radians)
            initial_wavelength: Initial photon wavelength (m)
            
        Returns:
            Final wavelength after scattering
        """
        compton_wavelength = self.h / (self.me * self.c)  # Compton wavelength
        delta_lambda = compton_wavelength * (1 - np.cos(theta))
        return initial_wavelength + delta_lambda


def plot_wave_particle_duality():
    """
    Create comprehensive plots showing wave-particle duality of light
    """
    duality = WaveParticleDuality()
    
    # Set up the figure with 2 rows and 4 columns 
    fig = plt.figure(figsize=(20, 12))
    
    # WAVE NATURE DEMONSTRATIONS
    
    # Plot 1: Wave Interference
    ax1 = plt.subplot(2, 4, 1)
    x = np.linspace(0, 4*np.pi, 1000)
    wavelength = 1.0
    
    # Individual waves
    wave1 = np.sin(2*np.pi*x/wavelength)
    wave2 = np.sin(2*np.pi*x/wavelength + np.pi/4)  # Phase shifted
    
    # Interference patterns
    constructive = duality.wave_interference(x, wavelength, 0)  # In phase
    destructive = duality.wave_interference(x, wavelength, np.pi)  # Out of phase
    
    ax1.plot(x, wave1, 'b--', alpha=0.6, label='Wave 1')
    ax1.plot(x, wave2, 'r--', alpha=0.6, label='Wave 2')
    ax1.plot(x, constructive/2, 'g-', linewidth=3, label='Constructive')
    ax1.plot(x, destructive/2, 'orange', linewidth=3, label='Destructive')
    
    ax1.set_xlabel('Position (wavelengths)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('1. Wave Interference\nConstructive & Destructive', fontsize=11, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-2.5, 2.5)
    
    # Plot 2: Single-Slit Diffraction
    ax2 = plt.subplot(2, 4, 2)
    theta = np.linspace(-0.05, 0.05, 1000)  # Small angles
    wavelength_vis = 550e-9  # Green light
    slit_widths = [1e-6, 2e-6, 5e-6]  # Different slit widths
    colors = ['red', 'blue', 'green']
    
    for slit_width, color in zip(slit_widths, colors):
        intensity = duality.diffraction_pattern(theta, wavelength_vis, slit_width)
        ax2.plot(np.degrees(theta), intensity, color=color, linewidth=2.5,
                label=f'a = {slit_width*1e6:.0f} μm')
    
    ax2.set_xlabel('Diffraction Angle (degrees)', fontsize=11)
    ax2.set_ylabel('Relative Intensity', fontsize=11)
    ax2.set_title('2. Single-Slit Diffraction\nWave Bending Pattern', fontsize=11, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Wave Reflection & Transmission
    ax3 = plt.subplot(2, 4, 3)
    angles = np.linspace(0, 90, 91)
    n1, n2 = 1.0, 1.5  # Air to glass
    
    # Fresnel equations for s-polarized light
    angles_rad = np.radians(angles)
    theta2_rad = np.arcsin((n1/n2) * np.sin(angles_rad))
    
    # Reflection coefficient
    Rs = ((n1*np.cos(angles_rad) - n2*np.cos(theta2_rad)) / 
          (n1*np.cos(angles_rad) + n2*np.cos(theta2_rad)))**2
    
    # Transmission coefficient  
    Ts = 1 - Rs
    
    ax3.plot(angles, Rs, 'r-', linewidth=3, label='Reflection (R)')
    ax3.plot(angles, Ts, 'b-', linewidth=3, label='Transmission (T)')
    ax3.plot(angles, Rs + Ts, 'k--', alpha=0.7, label='R + T = 1')
    
    ax3.set_xlabel('Incident Angle (degrees)', fontsize=11)
    ax3.set_ylabel('Fraction', fontsize=11)
    ax3.set_title('3. Wave Reflection/Transmission\nFresnel Equations', fontsize=11, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    # Plot 4: Wave Dispersion
    ax4 = plt.subplot(2, 4, 4)
    wavelengths = np.linspace(400, 700, 100)  # Visible spectrum (nm)
    
    # Sellmeier equation for glass dispersion (simplified)
    n_glass = 1.5 + 0.01 / (wavelengths/1000)**2  # Approximate dispersion
    
    # Color mapping for visible spectrum
    colors_spectrum = plt.cm.plasma((wavelengths - 400) / 300)
    
    for i in range(len(wavelengths)-1):
        ax4.plot(wavelengths[i:i+2], n_glass[i:i+2], color=colors_spectrum[i], linewidth=3)
    
    ax4.set_xlabel('Wavelength (nm)', fontsize=11)
    ax4.set_ylabel('Refractive Index', fontsize=11)
    ax4.set_title('4. Wave Dispersion\nWavelength-Dependent Refraction', fontsize=11, weight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add spectrum labels
    spectrum_regions = {'Violet': 420, 'Blue': 470, 'Green': 530, 'Yellow': 580, 'Red': 650}
    for color_name, wl in spectrum_regions.items():
        idx = np.argmin(np.abs(wavelengths - wl))
        ax4.annotate(color_name, (wl, n_glass[idx]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # PARTICLE NATURE DEMONSTRATIONS
    
    # Plot 5: Photoelectric Effect
    ax5 = plt.subplot(2, 4, 5)
    frequencies = np.linspace(5e14, 15e14, 100)  # Hz
    work_functions = [2.1, 4.3, 5.1]  # eV for different metals
    metals = ['Cesium', 'Zinc', 'Platinum']
    colors_metals = ['gold', 'silver', 'gray']
    
    for wf, metal, color in zip(work_functions, metals, colors_metals):
        ke, current = duality.photoelectric_effect(frequencies, wf)
        ax5.plot(frequencies/1e14, ke, color=color, linewidth=3, label=f'{metal} (φ={wf}eV)')
    
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Frequency (×10¹⁴ Hz)', fontsize=11)
    ax5.set_ylabel('Kinetic Energy (eV)', fontsize=11)
    ax5.set_title('5. Photoelectric Effect\nPhoton Energy Threshold', fontsize=11, weight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Photon Energy Distribution
    ax6 = plt.subplot(2, 4, 6)
    wavelengths_nm = np.linspace(200, 800, 100)
    wavelengths_m = wavelengths_nm * 1e-9
    photon_energies = duality.photon_energy(wavelengths_m)
    
    # Create color map for energy levels
    ax6.plot(wavelengths_nm, photon_energies, 'purple', linewidth=4, label='E = hf = hc/λ')
    ax6.fill_between(wavelengths_nm, 0, photon_energies, alpha=0.3, color='purple')
    
    # Mark important energy thresholds
    energy_thresholds = {'UV-C': 6.2, 'UV-B': 4.4, 'UV-A': 3.1, 'Visible': 1.8, 'IR': 1.2}
    for region, energy in energy_thresholds.items():
        wl_threshold = 1240 / energy  # Wavelength in nm for given energy in eV
        if 200 <= wl_threshold <= 800:
            ax6.axvline(x=wl_threshold, color='red', linestyle=':', alpha=0.8)
            ax6.text(wl_threshold, energy + 0.2, region, rotation=90, fontsize=9)
    
    ax6.set_xlabel('Wavelength (nm)', fontsize=11)
    ax6.set_ylabel('Photon Energy (eV)', fontsize=11)
    ax6.set_title('6. Photon Energy Distribution\nDiscrete Energy Packets', fontsize=11, weight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Compton Scattering
    ax7 = plt.subplot(2, 4, 7)
    scattering_angles = np.linspace(0, np.pi, 100)
    initial_wavelength = 0.71e-12  # X-ray wavelength (m)
    
    final_wavelengths = duality.compton_scattering(scattering_angles, initial_wavelength)
    wavelength_shift = (final_wavelengths - initial_wavelength) * 1e12  # in pm
    
    ax7.plot(np.degrees(scattering_angles), wavelength_shift, 'red', linewidth=3, 
            label='Δλ = λc(1 - cos θ)')
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Mark key angles
    key_angles = [0, 90, 180]
    for angle in key_angles:
        angle_rad = np.radians(angle)
        shift = (duality.compton_scattering(np.array([angle_rad]), initial_wavelength)[0] - 
                initial_wavelength) * 1e12
        ax7.plot(angle, shift, 'o', markersize=8, color='blue')
        ax7.annotate(f'{angle}°\n{shift:.2f}pm', (angle, shift), 
                    xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    ax7.set_xlabel('Scattering Angle (degrees)', fontsize=11)
    ax7.set_ylabel('Wavelength Shift (pm)', fontsize=11)
    ax7.set_title('7. Compton Scattering\nPhoton-Electron Collision', fontsize=11, weight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Wave-Particle Duality Summary
    ax8 = plt.subplot(2, 4, 8)
    
    # Create comparison of wave vs particle characteristics
    phenomena = ['Interference', 'Diffraction', 'Photoelectric', 'Compton']
    wave_strength = [100, 100, 0, 20]  # Percentage wave character
    particle_strength = [0, 0, 100, 80]  # Percentage particle character
    
    x_pos = np.arange(len(phenomena))
    width = 0.35
    
    bars1 = ax8.bar(x_pos - width/2, wave_strength, width, label='Wave Nature', 
                   color='blue', alpha=0.7)
    bars2 = ax8.bar(x_pos + width/2, particle_strength, width, label='Particle Nature', 
                   color='red', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax8.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    ax8.set_xlabel('Optical Phenomena', fontsize=11)
    ax8.set_ylabel('Character Strength (%)', fontsize=11)
    ax8.set_title('8. Wave-Particle Duality\nComplementarity Principle', fontsize=11, weight='bold')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(phenomena, rotation=45, ha='right')
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.legend()
    ax8.set_ylim(0, 110)
    
    plt.tight_layout(pad=2.5, h_pad=4.0, w_pad=2.0)
    plt.show()


def demonstrate_wave_particle_duality():
    """
    Main demonstration function with detailed explanations
    """
    print("🌊⚛️  WAVE-PARTICLE DUALITY OF LIGHT ⚛️🌊")
    print("="*60)
    print("Light exhibits both wave and particle properties - a fundamental")
    print("principle of quantum mechanics discovered in the early 20th century.")
    print()
    
    print("📊 WAVE NATURE EVIDENCE:")
    print("-" * 30)
    print("1. 🌊 Interference: Waves add constructively/destructively")
    print("2. 🔄 Diffraction: Waves bend around obstacles")  
    print("3. 🪞 Reflection: Wave amplitude reflection at interfaces")
    print("4. 🌈 Dispersion: Wavelength-dependent refraction")
    print()
    
    print("⚛️  PARTICLE NATURE EVIDENCE:")
    print("-" * 30)
    print("5. ⚡ Photoelectric Effect: Energy threshold behavior")
    print("6. 📦 Photon Packets: Discrete energy quanta E = hf")
    print("7. 💥 Compton Scattering: Photon-electron collisions")
    print("8. 🎯 Complementarity: Wave OR particle, not both simultaneously")
    print()
    
    duality = WaveParticleDuality()
    
    print("🔬 EXAMPLE CALCULATIONS:")
    print("=" * 40)
    
    # Wave calculations
    wavelength = 550e-9  # Green light
    frequency = duality.c / wavelength
    photon_energy_ev = duality.photon_energy(np.array([wavelength]))[0]
    
    print(f"Green Light (λ = {wavelength*1e9:.0f} nm):")
    print(f"  • Frequency: {frequency:.2e} Hz")
    print(f"  • Photon Energy: {photon_energy_ev:.2f} eV")
    print(f"  • Wave Number: {2*np.pi/wavelength:.2e} m⁻¹")
    print()
    
    # Photoelectric effect example
    work_function = 4.3  # Zinc work function (eV)
    if photon_energy_ev > work_function:
        ke = photon_energy_ev - work_function
        print(f"Photoelectric Effect (Zinc surface, φ = {work_function} eV):")
        print(f"  • Kinetic Energy: {ke:.2f} eV")
        print(f"  • Electron Speed: {np.sqrt(2*ke*duality.e/duality.me):.2e} m/s")
    else:
        print(f"Photoelectric Effect: No emission (E < φ)")
    print()
    
    # Compton scattering example
    initial_wl = 0.71e-12  # X-ray wavelength
    scattering_angle = np.pi/2  # 90 degrees
    final_wl = duality.compton_scattering(np.array([scattering_angle]), initial_wl)[0]
    shift = (final_wl - initial_wl) * 1e12
    
    print(f"Compton Scattering (90° scattering):")
    print(f"  • Initial λ: {initial_wl*1e12:.2f} pm")
    print(f"  • Final λ: {final_wl*1e12:.2f} pm") 
    print(f"  • Wavelength Shift: {shift:.2f} pm")
    print()
    
    print("📈 Generating wave-particle duality plots...")
    plot_wave_particle_duality()
    
    print("\n🎓 KEY INSIGHTS:")
    print("=" * 40)
    print("• Light is neither purely wave nor purely particle")
    print("• The observed nature depends on the experimental setup")
    print("• Wave-particle duality is fundamental to quantum mechanics")
    print("• Complementarity principle: Complete description requires both aspects")
    print("• Modern physics: Light is a quantum field excitation (photons)")


if __name__ == "__main__":
    print("🌟 Welcome to Wave-Particle Duality Explorer! 🌟")
    print("This script demonstrates the fundamental quantum nature of light:")
    print("• Wave properties: Interference, diffraction, reflection, dispersion")
    print("• Particle properties: Photoelectric effect, photons, Compton scattering")
    
    print("\nSelect demonstration:")
    print("1. Complete Wave-Particle Duality Analysis") 
    print("2. Wave Nature Only")
    print("3. Particle Nature Only")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    try:
        if choice == '1' or choice == '':
            demonstrate_wave_particle_duality()
        elif choice == '2':
            print("\n🌊 Wave Nature Demonstration...")
            # Could add wave-only demo here
            demonstrate_wave_particle_duality()
        elif choice == '3':
            print("\n⚛️  Particle Nature Demonstration...")
            # Could add particle-only demo here  
            demonstrate_wave_particle_duality()
        else:
            print("Running complete demonstration by default...")
            demonstrate_wave_particle_duality()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring wave-particle duality! ✨")