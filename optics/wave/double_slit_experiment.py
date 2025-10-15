#!/usr/bin/env python3
"""
Wave Optics: Young's Double-Slit Experiment
==========================================

This module demonstrates Young's famous double-slit experiment and its implications:
- Wave-particle duality of light
- Interference pattern formation
- Single photon interference
- Effect of slit separation and width
- Near-field and far-field patterns
- Coherence requirements
- Modern variations and applications

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
from typing import Tuple, List, Optional
import math


class DoubleSlit:
    """Represents a double-slit configuration."""
    
    def __init__(self, slit_separation: float, slit_width: float, 
                 wavelength: float, screen_distance: float):
        self.d = slit_separation      # Distance between slits
        self.a = slit_width          # Width of each slit
        self.wavelength = wavelength  # Wavelength of light
        self.L = screen_distance     # Distance to screen
        self.k = 2 * np.pi / wavelength  # Wavenumber
    
    def calculate_intensity_pattern(self, y_screen: np.ndarray) -> np.ndarray:
        """Calculate intensity pattern on screen using Fraunhofer diffraction."""
        
        # Angular position from center
        theta = np.arctan(y_screen / self.L)
        
        # Phase difference between slits
        delta = self.k * self.d * np.sin(theta)
        
        # Single-slit diffraction envelope
        beta = self.k * self.a * np.sin(theta) / 2
        
        # Avoid division by zero
        envelope = np.ones_like(beta)
        mask = np.abs(beta) > 1e-10
        envelope[mask] = (np.sin(beta[mask]) / beta[mask])**2
        
        # Double-slit interference
        interference = np.cos(delta / 2)**2
        
        # Total intensity (normalized)
        intensity = envelope * interference
        
        return intensity
    
    def calculate_fringe_positions(self, order_max: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate positions of bright and dark fringes."""
        
        # Bright fringes (maxima): d sin(θ) = mλ
        m_bright = np.arange(-order_max, order_max + 1)
        theta_bright = np.arcsin(m_bright * self.wavelength / self.d)
        y_bright = self.L * np.tan(theta_bright)
        
        # Dark fringes (minima): d sin(θ) = (m + 1/2)λ
        m_dark = np.arange(-order_max, order_max) + 0.5
        # Filter out angles > 90 degrees
        valid_dark = np.abs(m_dark * self.wavelength / self.d) <= 1
        m_dark = m_dark[valid_dark]
        theta_dark = np.arcsin(m_dark * self.wavelength / self.d)
        y_dark = self.L * np.tan(theta_dark)
        
        return y_bright, y_dark
    
    def plot_intensity_pattern(self, y_range: float = 0.02):
        """Plot the intensity pattern on the screen."""
        y_screen = np.linspace(-y_range, y_range, 1000)
        intensity = self.calculate_intensity_pattern(y_screen)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot intensity vs position
        ax1.plot(y_screen * 1000, intensity, 'b-', linewidth=2, label='Total Intensity')
        ax1.set_xlabel('Position on screen (mm)')
        ax1.set_ylabel('Normalized Intensity')
        ax1.set_title(f'Double-Slit Pattern (λ={self.wavelength*1e9:.0f}nm, d={self.d*1e6:.1f}μm)')
        ax1.grid(True, alpha=0.3)
        
        # Mark fringe positions
        y_bright, y_dark = self.calculate_fringe_positions(5)
        
        # Filter to display range
        y_bright_vis = y_bright[np.abs(y_bright) <= y_range]
        y_dark_vis = y_dark[np.abs(y_dark) <= y_range]
        
        for y in y_bright_vis:
            ax1.axvline(y * 1000, color='green', linestyle='--', alpha=0.7, label='Bright fringe' if y == y_bright_vis[0] else '')
        
        for y in y_dark_vis:
            ax1.axvline(y * 1000, color='red', linestyle='--', alpha=0.7, label='Dark fringe' if y == y_dark_vis[0] else '')
        
        ax1.legend()
        
        # Create 2D intensity map
        y_2d = np.linspace(-y_range, y_range, 200)
        x_2d = np.linspace(0, 0.01, 50)  # Small width for visualization
        Y_2d, X_2d = np.meshgrid(y_2d, x_2d)
        
        # Intensity pattern (same for all x)
        intensity_2d = np.tile(self.calculate_intensity_pattern(y_2d), (len(x_2d), 1))
        
        im = ax2.imshow(intensity_2d, extent=[-y_range*1000, y_range*1000, 0, 10], 
                       cmap='hot', aspect='auto', origin='lower')
        ax2.set_xlabel('Position on screen (mm)')
        ax2.set_ylabel('Width (mm)')
        ax2.set_title('2D Intensity Pattern')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Intensity')
        
        plt.tight_layout()
        return fig, (ax1, ax2)


def demonstrate_classic_double_slit():
    """Demonstrate the classic Young's double-slit experiment."""
    print("🌊 Classic Double-Slit Experiment")
    print("=" * 35)
    
    # Standard parameters
    wavelength = 632.8e-9  # HeNe laser (red)
    slit_separation = 100e-6  # 100 micrometers
    slit_width = 20e-6      # 20 micrometers
    screen_distance = 2.0    # 2 meters
    
    # Create double-slit system
    double_slit = DoubleSlit(slit_separation, slit_width, wavelength, screen_distance)
    
    # Plot intensity pattern
    fig, axes = double_slit.plot_intensity_pattern()
    plt.show()
    
    # Calculate and display key parameters
    fringe_spacing = wavelength * screen_distance / slit_separation
    
    print(f"\n📊 Experiment Parameters:")
    print(f"   Wavelength: {wavelength*1e9:.1f} nm")
    print(f"   Slit separation: {slit_separation*1e6:.1f} μm")
    print(f"   Slit width: {slit_width*1e6:.1f} μm")
    print(f"   Screen distance: {screen_distance:.1f} m")
    print(f"   Fringe spacing: {fringe_spacing*1e3:.2f} mm")
    
    # Angular resolution
    angular_resolution = wavelength / slit_separation
    print(f"   Angular fringe width: {np.degrees(angular_resolution)*3600:.1f} arcseconds")


def demonstrate_parameter_effects():
    """Demonstrate effects of changing experimental parameters."""
    print("\n🔧 Parameter Effects on Interference Pattern")
    print("=" * 50)
    
    # Base parameters
    wavelength = 550e-9  # Green light
    screen_distance = 1.0
    slit_width = 10e-6
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Effect of slit separation
    separations = [50e-6, 100e-6, 200e-6]  # Different separations
    y_range = 0.015
    y_screen = np.linspace(-y_range, y_range, 500)
    
    for i, d in enumerate(separations):
        double_slit = DoubleSlit(d, slit_width, wavelength, screen_distance)
        intensity = double_slit.calculate_intensity_pattern(y_screen)
        
        axes[0, 0].plot(y_screen*1000, intensity + i*0.5, 
                       label=f'd = {d*1e6:.0f} μm', linewidth=2)
    
    axes[0, 0].set_xlabel('Position (mm)')
    axes[0, 0].set_ylabel('Intensity (offset)')
    axes[0, 0].set_title('Effect of Slit Separation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Effect of slit width
    slit_separation = 100e-6
    widths = [5e-6, 15e-6, 30e-6]  # Different widths
    
    for i, a in enumerate(widths):
        double_slit = DoubleSlit(slit_separation, a, wavelength, screen_distance)
        intensity = double_slit.calculate_intensity_pattern(y_screen)
        
        axes[0, 1].plot(y_screen*1000, intensity + i*0.5, 
                       label=f'a = {a*1e6:.0f} μm', linewidth=2)
    
    axes[0, 1].set_xlabel('Position (mm)')
    axes[0, 1].set_ylabel('Intensity (offset)')
    axes[0, 1].set_title('Effect of Slit Width')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Effect of wavelength
    wavelengths = [450e-9, 550e-9, 650e-9]  # Blue, Green, Red
    colors = ['blue', 'green', 'red']
    
    for i, (wl, color) in enumerate(zip(wavelengths, colors)):
        double_slit = DoubleSlit(slit_separation, slit_width, wl, screen_distance)
        intensity = double_slit.calculate_intensity_pattern(y_screen)
        
        axes[1, 0].plot(y_screen*1000, intensity + i*0.5, 
                       color=color, label=f'λ = {wl*1e9:.0f} nm', linewidth=2)
    
    axes[1, 0].set_xlabel('Position (mm)')
    axes[1, 0].set_ylabel('Intensity (offset)')
    axes[1, 0].set_title('Effect of Wavelength')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Effect of screen distance
    distances = [0.5, 1.0, 2.0]  # Different distances
    
    for i, L in enumerate(distances):
        double_slit = DoubleSlit(slit_separation, slit_width, wavelength, L)
        # Adjust y_range for different distances
        y_adj = y_screen * (1.0 / L)  # Scale to keep same angular range
        intensity = double_slit.calculate_intensity_pattern(y_adj)
        
        axes[1, 1].plot(y_adj*1000, intensity + i*0.5, 
                       label=f'L = {L:.1f} m', linewidth=2)
    
    axes[1, 1].set_xlabel('Position (mm)')
    axes[1, 1].set_ylabel('Intensity (offset)')
    axes[1, 1].set_title('Effect of Screen Distance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📈 Parameter Relationships:")
    print(f"   Fringe spacing ∝ λL/d (wavelength × distance / separation)")
    print(f"   Smaller slit separation → wider fringes")
    print(f"   Larger wavelength → wider fringes") 
    print(f"   Farther screen → wider fringes")
    print(f"   Slit width affects envelope, not fringe spacing")


def demonstrate_near_vs_far_field():
    """Demonstrate near-field vs far-field diffraction patterns."""
    print("\n🔍 Near-Field vs Far-Field Patterns")
    print("=" * 40)
    
    # Parameters
    wavelength = 633e-9
    slit_separation = 50e-6
    slit_width = 10e-6
    
    # Different screen distances
    distances = [0.1, 0.5, 2.0, 10.0]  # From near-field to far-field
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, distance in enumerate(distances):
        double_slit = DoubleSlit(slit_separation, slit_width, wavelength, distance)
        
        # Calculate Fresnel number
        fresnel_number = slit_separation**2 / (wavelength * distance)
        
        # Adjust y-range based on distance
        if fresnel_number > 1:  # Near-field
            y_range = 0.001
        else:  # Far-field
            y_range = 0.02
        
        y_screen = np.linspace(-y_range, y_range, 500)
        intensity = double_slit.calculate_intensity_pattern(y_screen)
        
        axes[i].plot(y_screen*1000, intensity, 'b-', linewidth=2)
        axes[i].set_xlabel('Position (mm)')
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'L = {distance:.1f}m, F = {fresnel_number:.2f}')
        axes[i].grid(True, alpha=0.3)
        
        # Add regime label
        regime = "Near-field" if fresnel_number > 1 else "Far-field"
        axes[i].text(0.02, 0.95, regime, transform=axes[i].transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📏 Fresnel Number Analysis:")
    print(f"   Fresnel number: F = d²/(λL)")
    print(f"   F >> 1: Near-field (Fresnel diffraction)")
    print(f"   F << 1: Far-field (Fraunhofer diffraction)")
    print(f"   Transition typically occurs around F ≈ 1")


def demonstrate_coherence_requirements():
    """Demonstrate coherence requirements for interference."""
    print("\n🎯 Coherence Requirements")
    print("=" * 30)
    
    # Simulate partially coherent light
    wavelength = 550e-9
    slit_separation = 100e-6
    slit_width = 20e-6
    screen_distance = 1.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Temporal coherence effect
    coherence_lengths = [1e-3, 1e-4, 1e-5, 1e-6]  # Different coherence lengths
    y_screen = np.linspace(-0.01, 0.01, 500)
    
    double_slit = DoubleSlit(slit_separation, slit_width, wavelength, screen_distance)
    base_intensity = double_slit.calculate_intensity_pattern(y_screen)
    
    for i, Lc in enumerate(coherence_lengths):
        # Path difference between slits
        theta = np.arctan(y_screen / screen_distance)
        path_diff = slit_separation * np.sin(theta)
        
        # Visibility reduction due to finite coherence
        visibility = np.exp(-path_diff / Lc)
        
        # Modify intensity pattern
        intensity_coherent = visibility * base_intensity + (1 - visibility) * 0.5
        
        ax1.plot(y_screen*1000, intensity_coherent + i*0.5, 
                label=f'Lc = {Lc*1e3:.1f} mm', linewidth=2)
    
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Intensity (offset)')
    ax1.set_title('Effect of Temporal Coherence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Spatial coherence effect (finite source size)
    source_sizes = np.array([1e-6, 10e-6, 100e-6, 1e-3])  # Different source sizes
    source_distance = 1.0  # Distance from source to slits
    
    for i, source_size in enumerate(source_sizes):
        # Angular size of source
        angular_source = source_size / source_distance
        
        # Coherence condition: θc ≈ λ/d (for first minimum)
        coherence_angle = wavelength / slit_separation
        
        # Visibility for extended source
        if angular_source < coherence_angle:
            visibility_spatial = 1.0  # Fully coherent
        else:
            visibility_spatial = coherence_angle / angular_source
        
        # Apply spatial coherence effect
        intensity_spatial = visibility_spatial * base_intensity + (1 - visibility_spatial) * 0.5
        
        ax2.plot(y_screen*1000, intensity_spatial + i*0.5, 
                label=f'Source: {source_size*1e6:.1f} μm', linewidth=2)
    
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Intensity (offset)')
    ax2.set_title('Effect of Spatial Coherence (Source Size)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔬 Coherence Criteria:")
    print(f"   Temporal: Path difference < coherence length")
    print(f"   Spatial: Source angular size < λ/d")
    print(f"   Both required for high-contrast fringes")


def demonstrate_modern_applications():
    """Demonstrate modern applications of double-slit principles."""
    print("\n🚀 Modern Applications")
    print("=" * 25)
    
    print(f"\n💡 Applications of Double-Slit Principles:")
    print(f"   • Interferometric sensors (displacement, pressure)")
    print(f"   • Wavelength meters and spectrometers")
    print(f"   • Coherence measurement instruments")
    print(f"   • Quantum mechanics demonstrations")
    print(f"   • Optical computing and signal processing")
    
    # Simulate a wavelength measurement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Known double-slit setup
    slit_separation = 75e-6
    slit_width = 15e-6
    screen_distance = 1.5
    
    # Measure fringe spacing to determine wavelength
    known_wavelengths = np.array([532e-9, 633e-9, 780e-9])  # Green, Red, IR
    colors = ['green', 'red', 'brown']
    
    y_screen = np.linspace(-0.015, 0.015, 1000)
    
    for i, (wl, color) in enumerate(zip(known_wavelengths, colors)):
        double_slit = DoubleSlit(slit_separation, slit_width, wl, screen_distance)
        intensity = double_slit.calculate_intensity_pattern(y_screen)
        
        ax1.plot(y_screen*1000, intensity + i*1.2, 
                color=color, label=f'{wl*1e9:.0f} nm', linewidth=2)
        
        # Mark first maximum positions
        fringe_spacing = wl * screen_distance / slit_separation
        ax1.axvline(fringe_spacing*1000, color=color, linestyle='--', alpha=0.7)
        ax1.axvline(-fringe_spacing*1000, color=color, linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Intensity (offset)')
    ax1.set_title('Wavelength Measurement via Fringe Spacing')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision vs slit separation
    slit_seps = np.linspace(20e-6, 200e-6, 100)
    wavelength_test = 633e-9
    
    # Calculate measurement precision (assuming ±0.1 mm position accuracy)
    position_error = 0.1e-3  # 0.1 mm
    fringe_spacings = wavelength_test * screen_distance / slit_seps
    wavelength_errors = position_error * wavelength_test / fringe_spacings
    relative_errors = wavelength_errors / wavelength_test * 100
    
    ax2.loglog(slit_seps*1e6, relative_errors, 'b-', linewidth=2)
    ax2.set_xlabel('Slit Separation (μm)')
    ax2.set_ylabel('Wavelength Error (%)')
    ax2.set_title('Measurement Precision vs Slit Separation')
    ax2.grid(True, alpha=0.3)
    
    # Mark practical range
    ax2.axvspan(50, 150, alpha=0.2, color='green', label='Practical range')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🎯 Design Considerations:")
    print(f"   • Smaller slits → better wavelength resolution")
    print(f"   • Larger screen distance → better spatial resolution")
    print(f"   • Coherent illumination essential")
    print(f"   • Environmental stability critical for precision")


if __name__ == "__main__":
    print("🔬 Welcome to Young's Double-Slit Explorer! 🔬")
    print("This script demonstrates the famous Young's double-slit experiment and its applications.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_classic_double_slit()
        demonstrate_parameter_effects()
        demonstrate_near_vs_far_field()
        demonstrate_coherence_requirements()
        demonstrate_modern_applications()
        
        print(f"\n" + "="*70)
        print("🎓 Young's Double-Slit Summary:")
        print("="*70)
        print("Historical Significance:")
        print("• First convincing demonstration of wave nature of light (1801)")
        print("• Established wave-particle duality concept")
        print("• Foundation for quantum mechanical interpretation")
        
        print(f"\nKey Physics Principles:")
        print("• Superposition of coherent waves")
        print("• Interference pattern: I ∝ cos²(πd sin θ/λ)")
        print("• Fringe spacing: Δy = λL/d")
        print("• Coherence requirements for visibility")
        
        print(f"\nModern Impact:")
        print("• Quantum mechanics education and research")
        print("• Precision measurement instruments")
        print("• Optical system design and testing")
        print("• Fundamental physics experiments")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring Young's double-slit experiment! ✨")