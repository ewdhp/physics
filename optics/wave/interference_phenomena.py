#!/usr/bin/env python3
"""
Wave Optics: Interference Phenomena
==================================

This module demonstrates the fundamental principles of optical interference:
- Superposition principle
- Constructive and destructive interference
- Path difference and phase relationships
- Interference in thin films (soap bubbles, oil films)
- Newton's rings
- Michelson interferometer principles
- Temporal and spatial coherence effects

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


class WaveSource:
    """Represents a coherent light source."""
    
    def __init__(self, x: float, y: float, amplitude: float = 1.0, 
                 wavelength: float = 500e-9, phase: float = 0.0):
        self.x = x
        self.y = y
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.frequency = 3e8 / wavelength  # c = λf
        self.phase = phase
        self.wavenumber = 2 * np.pi / wavelength
    
    def wave_at_point(self, x: float, y: float, t: float = 0) -> complex:
        """Calculate wave amplitude at a given point and time."""
        distance = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        phase = self.wavenumber * distance - 2 * np.pi * self.frequency * t + self.phase
        return self.amplitude * np.exp(1j * phase) / np.sqrt(distance + 1e-10)


class InterferenceSimulator:
    """Simulates interference patterns from multiple wave sources."""
    
    def __init__(self):
        self.sources: List[WaveSource] = []
        self.grid_size = (200, 200)
        self.physical_size = (10e-3, 10e-3)  # 10mm x 10mm
    
    def add_source(self, source: WaveSource):
        """Add a wave source to the simulation."""
        self.sources.append(source)
    
    def calculate_interference_pattern(self, t: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the interference pattern at time t."""
        # Create coordinate grid
        x = np.linspace(-self.physical_size[0]/2, self.physical_size[0]/2, self.grid_size[0])
        y = np.linspace(-self.physical_size[1]/2, self.physical_size[1]/2, self.grid_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Initialize total field
        total_field = np.zeros_like(X, dtype=complex)
        
        # Sum contributions from all sources
        for source in self.sources:
            # Avoid singularity at source location
            mask = np.sqrt((X - source.x)**2 + (Y - source.y)**2) > 1e-6
            field = np.zeros_like(X, dtype=complex)
            
            distances = np.sqrt((X - source.x)**2 + (Y - source.y)**2)
            phases = source.wavenumber * distances - 2 * np.pi * source.frequency * t + source.phase
            
            field[mask] = source.amplitude * np.exp(1j * phases[mask]) / np.sqrt(distances[mask])
            total_field += field
        
        # Calculate intensity (|E|²)
        intensity = np.abs(total_field)**2
        
        return X, Y, intensity
    
    def plot_interference_pattern(self, t: float = 0, title: str = "Interference Pattern"):
        """Plot the interference pattern."""
        X, Y, intensity = self.calculate_interference_pattern(t)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot intensity pattern
        im = ax.imshow(intensity, extent=[X.min()*1e3, X.max()*1e3, Y.min()*1e3, Y.max()*1e3],
                      cmap='hot', origin='lower', aspect='equal')
        
        # Mark source locations
        for i, source in enumerate(self.sources):
            ax.plot(source.x*1e3, source.y*1e3, 'wo', markersize=8, 
                   markeredgecolor='blue', markeredgewidth=2, 
                   label=f'Source {i+1}')
        
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(title)
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity (arbitrary units)')
        
        plt.tight_layout()
        return fig, ax


def demonstrate_two_source_interference():
    """Demonstrate interference from two coherent point sources."""
    print("🌊 Two-Source Interference")
    print("=" * 30)
    
    # Create interferometer with two sources
    sim = InterferenceSimulator()
    
    # Two coherent sources separated by 2mm
    source1 = WaveSource(-1e-3, 0, amplitude=1.0, wavelength=633e-9)  # Red laser
    source2 = WaveSource(1e-3, 0, amplitude=1.0, wavelength=633e-9)
    
    sim.add_source(source1)
    sim.add_source(source2)
    
    # Plot interference pattern
    fig, ax = sim.plot_interference_pattern(title="Two-Source Interference (λ = 633 nm)")
    plt.show()
    
    # Analyze interference conditions
    wavelength = 633e-9
    source_separation = 2e-3
    
    print(f"\n📊 Interference Analysis:")
    print(f"   Wavelength: {wavelength*1e9:.0f} nm")
    print(f"   Source separation: {source_separation*1e3:.1f} mm")
    print(f"   Fringe spacing at 1m: {wavelength * 1.0 / source_separation * 1e3:.2f} mm")
    
    # Calculate fringe visibility
    max_intensity = 4.0  # I1 + I2 + 2√(I1*I2) for equal amplitudes
    min_intensity = 0.0  # Complete destructive interference possible
    visibility = (max_intensity - min_intensity) / (max_intensity + min_intensity)
    print(f"   Fringe visibility: {visibility:.2f}")


def demonstrate_thin_film_interference():
    """Demonstrate thin film interference (soap bubbles, oil films)."""
    print("\n🫧 Thin Film Interference")
    print("=" * 30)
    
    # Thin film parameters
    film_thickness = np.linspace(100e-9, 1000e-9, 100)  # 100-1000 nm
    n_film = 1.33  # Water (soap bubble)
    n_air = 1.0
    wavelengths = np.array([450e-9, 550e-9, 650e-9])  # Blue, Green, Red
    colors = ['blue', 'green', 'red']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate reflectance for different wavelengths
    for i, (wavelength, color) in enumerate(zip(wavelengths, colors)):
        # Optical path difference for normal incidence
        # δ = 2nt (with phase change at air-film interface)
        optical_path_diff = 2 * n_film * film_thickness
        phase_diff = 2 * np.pi * optical_path_diff / wavelength + np.pi  # +π for phase change
        
        # Reflectance from thin film (simplified)
        reflectance = np.sin(phase_diff / 2)**2
        
        ax1.plot(film_thickness * 1e9, reflectance, color=color, linewidth=2, 
                label=f'{wavelength*1e9:.0f} nm')
    
    ax1.set_xlabel('Film Thickness (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Thin Film Interference - Spectral Response')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Show color appearance for white light
    thickness_values = np.array([200e-9, 300e-9, 500e-9, 800e-9])
    
    for j, thickness in enumerate(thickness_values):
        # Calculate color components
        rgb_intensity = []
        for wavelength in wavelengths:
            optical_path_diff = 2 * n_film * thickness
            phase_diff = 2 * np.pi * optical_path_diff / wavelength + np.pi
            intensity = np.sin(phase_diff / 2)**2
            rgb_intensity.append(intensity)
        
        # Normalize to create RGB color
        rgb_normalized = np.array(rgb_intensity) / max(rgb_intensity)
        
        # Create color patch
        rect = plt.Rectangle((j, 0), 0.8, 1, facecolor=rgb_normalized, 
                           edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        
        # Add thickness label
        ax2.text(j + 0.4, -0.2, f'{thickness*1e9:.0f} nm', 
                ha='center', va='top', fontsize=10)
    
    ax2.set_xlim(-0.2, len(thickness_values))
    ax2.set_ylim(-0.3, 1.2)
    ax2.set_xlabel('Film Thickness')
    ax2.set_ylabel('Relative Color')
    ax2.set_title('Soap Bubble Colors')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 Thin Film Physics:")
    print(f"   • Constructive interference: 2nt = (m + ½)λ (with phase change)")
    print(f"   • Destructive interference: 2nt = mλ")
    print(f"   • Phase change occurs at higher refractive index interface")
    print(f"   • Colors depend on film thickness and viewing angle")


def demonstrate_newtons_rings():
    """Demonstrate Newton's rings interference pattern."""
    print("\n🔵 Newton's Rings")
    print("=" * 20)
    
    # Create radial coordinate system
    r_max = 10e-3  # 10 mm radius
    r = np.linspace(0, r_max, 500)
    
    # Newton's rings parameters
    wavelength = 589e-9  # Sodium D-line
    radius_of_curvature = 1.0  # 1 meter radius lens
    
    # Air gap thickness as function of radius
    # t(r) = r²/(2R) for small angles
    air_gap = r**2 / (2 * radius_of_curvature)
    
    # Path difference (with phase change at glass-air interface)
    path_difference = 2 * air_gap
    phase_difference = 2 * np.pi * path_difference / wavelength + np.pi
    
    # Intensity pattern
    intensity = np.cos(phase_difference / 2)**2
    
    # Create 2D pattern
    theta = np.linspace(0, 2*np.pi, 360)
    R, Theta = np.meshgrid(r, theta)
    
    # Extend intensity to 2D
    Intensity_2D = np.tile(intensity, (len(theta), 1))
    
    # Convert to Cartesian coordinates for plotting
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 2D Newton's rings
    im = ax1.imshow(Intensity_2D, extent=[-r_max*1e3, r_max*1e3, -r_max*1e3, r_max*1e3],
                   cmap='gray', origin='lower', aspect='equal')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title("Newton's Rings Pattern")
    
    # Plot radial intensity profile
    ax2.plot(r*1e3, intensity, 'b-', linewidth=2)
    ax2.set_xlabel('Radius (mm)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Radial Intensity Profile')
    ax2.grid(True, alpha=0.3)
    
    # Mark dark rings
    dark_ring_condition = np.pi * (2*np.arange(1, 6) - 1)  # (2m-1)π
    dark_radii = np.sqrt(dark_ring_condition * wavelength * radius_of_curvature / (2*np.pi))
    
    for i, radius in enumerate(dark_radii):
        if radius < r_max:
            ax2.axvline(radius*1e3, color='red', linestyle='--', alpha=0.7)
            ax2.text(radius*1e3, 0.8-i*0.1, f'Dark {i+1}', rotation=90, 
                    ha='right', va='bottom', color='red')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔍 Newton's Rings Analysis:")
    print(f"   Wavelength: {wavelength*1e9:.0f} nm")
    print(f"   Lens curvature radius: {radius_of_curvature:.1f} m")
    print(f"   Dark ring radii (first 3):")
    for i, radius in enumerate(dark_radii[:3]):
        print(f"     Ring {i+1}: {radius*1e3:.2f} mm")


def demonstrate_michelson_interferometer():
    """Demonstrate Michelson interferometer principles."""
    print("\n🔬 Michelson Interferometer")
    print("=" * 30)
    
    # Michelson interferometer simulation
    wavelength = 632.8e-9  # HeNe laser
    
    # Path difference range (mirror movement)
    path_difference = np.linspace(0, 5*wavelength, 1000)
    
    # Phase difference
    phase_difference = 2 * np.pi * path_difference / wavelength
    
    # Intensity variation (assuming equal beam intensities)
    intensity = np.cos(phase_difference / 2)**2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot intensity vs path difference
    ax1.plot(path_difference*1e9, intensity, 'r-', linewidth=2)
    ax1.set_xlabel('Path Difference (nm)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Michelson Interferometer - Intensity vs Path Difference')
    ax1.grid(True, alpha=0.3)
    
    # Mark constructive and destructive interference
    constructive_points = np.arange(0, 6) * wavelength
    destructive_points = (np.arange(0, 5) + 0.5) * wavelength
    
    for point in constructive_points:
        if point <= path_difference.max():
            ax1.axvline(point*1e9, color='green', linestyle='--', alpha=0.7)
    
    for point in destructive_points:
        if point <= path_difference.max():
            ax1.axvline(point*1e9, color='red', linestyle='--', alpha=0.7)
    
    # Schematic diagram
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    
    # Draw interferometer components
    # Laser source
    ax2.plot(1, 4, 'ro', markersize=10, label='Laser')
    ax2.text(0.5, 4.5, 'Laser', ha='center')
    
    # Beam splitter
    ax2.plot([3, 5], [4, 6], 'k-', linewidth=3, label='Beam Splitter')
    ax2.plot([3, 5], [6, 4], 'k-', linewidth=3)
    
    # Mirrors
    ax2.plot([7, 9], [6, 6], 'k-', linewidth=5, label='Mirror 1')
    ax2.plot([4, 4], [7.5, 7.5], 'k-', linewidth=5, label='Mirror 2')
    
    # Detector
    ax2.plot(1, 1, 'bs', markersize=12, label='Detector')
    ax2.text(0.5, 0.5, 'Detector', ha='center')
    
    # Light paths
    ax2.arrow(1.2, 4, 1.5, 1.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax2.arrow(4.2, 5.8, 2.5, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax2.arrow(4, 6.2, 0, 1, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax2.arrow(3.8, 5.8, -2.5, -4.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax2.set_aspect('equal')
    ax2.set_title('Michelson Interferometer Schematic')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n⚡ Interferometer Applications:")
    print(f"   • Precision length measurement (nm resolution)")
    print(f"   • Gravitational wave detection (LIGO)")
    print(f"   • Refractive index measurement")
    print(f"   • Surface profiling and topography")
    print(f"   • Wavelength calibration")


def demonstrate_coherence_effects():
    """Demonstrate temporal and spatial coherence effects."""
    print("\n🎯 Coherence in Interference")
    print("=" * 30)
    
    # Temporal coherence - finite coherence length
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Different light sources and their coherence properties
    sources = {
        'HeNe Laser': {'wavelength': 632.8e-9, 'linewidth': 1e-12, 'color': 'red'},
        'LED': {'wavelength': 650e-9, 'linewidth': 20e-9, 'color': 'orange'},
        'Tungsten Lamp': {'wavelength': 550e-9, 'linewidth': 100e-9, 'color': 'yellow'}
    }
    
    path_difference = np.linspace(0, 1e-3, 1000)  # 0 to 1 mm
    
    for source_name, props in sources.items():
        wavelength = props['wavelength']
        linewidth = props['linewidth']
        
        # Coherence length
        coherence_length = wavelength**2 / linewidth
        
        # Visibility as function of path difference
        visibility = np.exp(-path_difference / coherence_length)
        
        ax1.plot(path_difference*1e3, visibility, color=props['color'], 
                linewidth=2, label=f"{source_name} (Lc={coherence_length*1e3:.2f}mm)")
    
    ax1.set_xlabel('Path Difference (mm)')
    ax1.set_ylabel('Fringe Visibility')
    ax1.set_title('Temporal Coherence - Visibility vs Path Difference')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Spatial coherence - finite source size
    source_sizes = np.logspace(-6, -3, 100)  # 1 μm to 1 mm
    wavelength = 550e-9
    distance = 1.0  # 1 meter from source
    
    # Angular size of source
    angular_size = source_sizes / distance
    
    # Coherence area (van Cittert-Zernike theorem)
    coherence_angle = wavelength / source_sizes
    coherence_area = (coherence_angle * distance)**2
    
    ax2.loglog(source_sizes*1e6, coherence_area*1e6, 'b-', linewidth=2)
    ax2.set_xlabel('Source Size (μm)')
    ax2.set_ylabel('Coherence Area (μm²)')
    ax2.set_title('Spatial Coherence - Coherence Area vs Source Size')
    ax2.grid(True, alpha=0.3)
    
    # Mark typical sources
    typical_sources = {
        'Star': 1e-6,
        'Pinhole': 10e-6,
        'LED': 100e-6,
        'Sun': 1e-3
    }
    
    for name, size in typical_sources.items():
        if size*1e6 >= source_sizes[0]*1e6 and size*1e6 <= source_sizes[-1]*1e6:
            coherence = (wavelength / size * distance)**2
            ax2.plot(size*1e6, coherence*1e6, 'ro', markersize=8)
            ax2.annotate(name, (size*1e6, coherence*1e6), 
                        xytext=(10, 10), textcoords='offset points')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📏 Coherence Parameters:")
    print(f"   Temporal coherence length: Lc = λ²/Δλ")
    print(f"   Spatial coherence length: lc = λD/d")
    print(f"   Where D = distance to source, d = source size")
    print(f"\n🔬 Practical Implications:")
    print(f"   • Laser: Long coherence length → stable interference")
    print(f"   • LED: Short coherence → requires matched path lengths")
    print(f"   • Sunlight: Very short coherence → difficult to observe interference")


if __name__ == "__main__":
    print("🌈 Welcome to Interference Phenomena Explorer! 🌈")
    print("This script demonstrates the fundamental principles of optical interference.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_two_source_interference()
        demonstrate_thin_film_interference()
        demonstrate_newtons_rings()
        demonstrate_michelson_interferometer()
        demonstrate_coherence_effects()
        
        print(f"\n" + "="*70)
        print("🎓 Interference Phenomena Summary:")
        print("="*70)
        print("Key Principles Demonstrated:")
        print("• Superposition of coherent waves")
        print("• Path difference and phase relationships")
        print("• Constructive and destructive interference conditions")
        print("• Thin film interference and structural colors")
        print("• Newton's rings and curved surface interference")
        print("• Michelson interferometer operation")
        print("• Temporal and spatial coherence effects")
        
        print(f"\n💡 Applications:")
        print("• Precision metrology and length standards")
        print("• Anti-reflection coatings on lenses")
        print("• Interferometric imaging and microscopy")
        print("• Gravitational wave detection (LIGO)")
        print("• Optical coherence tomography (medical imaging)")
        print("• Quality control in manufacturing")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring interference phenomena! ✨")