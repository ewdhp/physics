#!/usr/bin/env python3
"""
Wave Optics: Coherence
======================

This module demonstrates coherence properties of light:
- Temporal coherence and coherence time
- Spatial coherence and coherence area
- Coherence length and fringe visibility
- Partially coherent light sources
- Van Cittert-Zernike theorem
- Stellar interferometry applications
- Laser coherence properties
- Measurement techniques (Michelson stellar interferometer)

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
from matplotlib.patches import Circle
from scipy import signal
from typing import Tuple, List, Optional
import math


class LightSource:
    """Represents a light source with coherence properties."""
    
    def __init__(self, central_wavelength: float, spectral_width: float, 
                 source_size: float, source_type: str = "thermal"):
        self.lambda_0 = central_wavelength   # Central wavelength
        self.delta_lambda = spectral_width   # Spectral linewidth
        self.source_size = source_size       # Physical source size
        self.source_type = source_type       # "thermal", "laser", "LED"
        
        # Calculate coherence properties
        self.coherence_length = self.calculate_coherence_length()
        self.coherence_time = self.calculate_coherence_time()
    
    def calculate_coherence_length(self) -> float:
        """Calculate temporal coherence length."""
        if self.delta_lambda > 0:
            return self.lambda_0**2 / self.delta_lambda
        else:
            return float('inf')  # Perfectly monochromatic
    
    def calculate_coherence_time(self) -> float:
        """Calculate coherence time."""
        c = 3e8  # Speed of light
        return self.coherence_length / c
    
    def spectral_profile(self, wavelengths: np.ndarray) -> np.ndarray:
        """Calculate normalized spectral profile."""
        if self.source_type == "laser":
            # Lorentzian profile for laser
            gamma = self.delta_lambda / 2
            profile = gamma**2 / ((wavelengths - self.lambda_0)**2 + gamma**2)
        elif self.source_type == "LED":
            # Gaussian profile for LED
            sigma = self.delta_lambda / (2 * np.sqrt(2 * np.log(2)))
            profile = np.exp(-0.5 * ((wavelengths - self.lambda_0) / sigma)**2)
        else:  # thermal
            # Blackbody-like profile (simplified Gaussian)
            sigma = self.delta_lambda / 3
            profile = np.exp(-0.5 * ((wavelengths - self.lambda_0) / sigma)**2)
        
        return profile / np.max(profile)
    
    def visibility_vs_path_difference(self, path_differences: np.ndarray) -> np.ndarray:
        """Calculate fringe visibility vs optical path difference."""
        # Visibility decreases exponentially with path difference
        return np.exp(-np.abs(path_differences) / self.coherence_length)
    
    def spatial_coherence_function(self, separation: float, distance: float) -> float:
        """Calculate spatial coherence using van Cittert-Zernike theorem."""
        if self.source_size == 0:
            return 1.0  # Point source - perfectly coherent
        
        # Angular size of source
        angular_size = self.source_size / distance
        
        # Spatial coherence (first-order approximation)
        x = np.pi * separation * angular_size / self.lambda_0
        
        if abs(x) < 1e-10:
            return 1.0
        else:
            return abs(np.sin(x) / x)


class CoherenceMeasurement:
    """Simulates coherence measurement techniques."""
    
    def __init__(self, source: LightSource):
        self.source = source
    
    def michelson_interferometer(self, path_difference_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Michelson interferometer measurement."""
        visibility = self.source.visibility_vs_path_difference(path_difference_range)
        
        # Intensity pattern with envelope
        phase = 2 * np.pi * path_difference_range / self.source.lambda_0
        intensity = 0.5 * (1 + visibility * np.cos(phase))
        
        return intensity, visibility
    
    def stellar_interferometer(self, baseline_range: np.ndarray, 
                             star_distance: float) -> np.ndarray:
        """Simulate stellar interferometer visibility."""
        visibility = []
        
        for baseline in baseline_range:
            vis = self.source.spatial_coherence_function(baseline, star_distance)
            visibility.append(vis)
        
        return np.array(visibility)


def demonstrate_temporal_coherence():
    """Demonstrate temporal coherence properties."""
    print("⏰ Temporal Coherence")
    print("=" * 20)
    
    # Different light sources
    sources = {
        "HeNe Laser": LightSource(632.8e-9, 1e-12, 0, "laser"),
        "LED": LightSource(650e-9, 20e-9, 1e-3, "LED"),
        "Tungsten Lamp": LightSource(550e-9, 100e-9, 5e-3, "thermal"),
        "Sunlight": LightSource(550e-9, 300e-9, 1.4e9, "thermal")  # Sun diameter ≈ 1.4e9 m
    }
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Spectral profiles
    wavelengths = np.linspace(400e-9, 800e-9, 1000)
    
    for name, source in sources.items():
        if source.lambda_0 >= 400e-9 and source.lambda_0 <= 800e-9:
            spectrum = source.spectral_profile(wavelengths)
            ax1.plot(wavelengths * 1e9, spectrum, linewidth=2, label=name)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Spectral Profiles of Different Light Sources')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(400, 800)
    
    # Coherence lengths
    source_names = list(sources.keys())
    coherence_lengths = [source.coherence_length for source in sources.values()]
    
    ax2.bar(source_names, np.log10(np.array(coherence_lengths) * 1e3), 
           color=['red', 'orange', 'yellow', 'blue'])
    ax2.set_ylabel('log₁₀(Coherence Length) [mm]')
    ax2.set_title('Temporal Coherence Length Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add actual values as text
    for i, (name, length) in enumerate(zip(source_names, coherence_lengths)):
        if length < 1e-3:
            label = f'{length*1e6:.1f} μm'
        elif length < 1:
            label = f'{length*1e3:.1f} mm'
        else:
            label = f'{length:.2f} m'
        
        ax2.text(i, np.log10(length * 1e3) + 0.1, label, 
                ha='center', va='bottom', fontsize=10, rotation=45)
    
    # Visibility vs path difference
    path_diff = np.linspace(0, 1e-3, 1000)  # 0 to 1 mm
    
    for name, source in list(sources.items())[:3]:  # Skip sunlight for clarity
        measurement = CoherenceMeasurement(source)
        _, visibility = measurement.michelson_interferometer(path_diff)
        ax3.plot(path_diff * 1e3, visibility, linewidth=2, label=name)
    
    ax3.set_xlabel('Path Difference (mm)')
    ax3.set_ylabel('Fringe Visibility')
    ax3.set_title('Temporal Coherence - Visibility vs Path Difference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Temporal Coherence Comparison:")
    for name, source in sources.items():
        print(f"   {name}:")
        print(f"     Spectral width: {source.delta_lambda*1e9:.1f} nm")
        print(f"     Coherence length: {source.coherence_length*1e3:.2f} mm")
        print(f"     Coherence time: {source.coherence_time*1e15:.1f} fs")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Narrower spectrum → longer coherence")
    print(f"   • Lasers have excellent temporal coherence")
    print(f"   • Thermal sources have poor temporal coherence")


def demonstrate_spatial_coherence():
    """Demonstrate spatial coherence and van Cittert-Zernike theorem."""
    print("\n🌌 Spatial Coherence")
    print("=" * 20)
    
    # Create different source configurations
    wavelength = 550e-9
    distance = 1000.0  # 1 km distance to source
    
    source_sizes = [1e-6, 10e-6, 100e-6, 1e-3, 10e-3]  # Various source sizes
    source_labels = ['1 μm', '10 μm', '100 μm', '1 mm', '10 mm']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Spatial coherence vs source size
    separation_range = np.logspace(-6, -2, 100)  # 1 μm to 10 mm
    
    for i, (size, label) in enumerate(zip(source_sizes, source_labels)):
        source = LightSource(wavelength, 10e-9, size)  # Small spectral width
        coherence_values = []
        
        for separation in separation_range:
            coherence = source.spatial_coherence_function(separation, distance)
            coherence_values.append(coherence)
        
        ax1.semilogx(separation_range * 1e6, coherence_values, 
                    linewidth=2, label=f'Source: {label}')
    
    ax1.set_xlabel('Separation (μm)')
    ax1.set_ylabel('Spatial Coherence |γ₁₂|')
    ax1.set_title('Spatial Coherence vs Detector Separation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Coherence area vs source size
    coherence_areas = []
    
    for size in source_sizes:
        # Coherence length ≈ λD/d where D is distance, d is source size
        coherence_length_spatial = wavelength * distance / size if size > 0 else float('inf')
        coherence_area = coherence_length_spatial**2
        coherence_areas.append(coherence_area)
    
    ax2.loglog(np.array(source_sizes) * 1e6, np.array(coherence_areas) * 1e12, 
              'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Source Size (μm)')
    ax2.set_ylabel('Coherence Area (μm²)')
    ax2.set_title('Coherence Area vs Source Size')
    ax2.grid(True, alpha=0.3)
    
    # Add theoretical line (inverse relationship)
    theory_line = (wavelength * distance / (source_sizes[0]))**2 * (source_sizes[0] / np.array(source_sizes))**2
    ax2.loglog(np.array(source_sizes) * 1e6, theory_line * 1e12, 
              'r--', linewidth=2, label='∝ 1/d²')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔍 Van Cittert-Zernike Theorem:")
    print(f"   Spatial coherence length: lc ≈ λD/d")
    print(f"   Where D = source distance, d = source size")
    
    print(f"\n📐 Coherence Areas (at 1 km distance):")
    for size, area in zip(source_sizes, coherence_areas):
        if area < 1e-6:
            area_str = f"{area*1e12:.1f} μm²"
        elif area < 1e-3:
            area_str = f"{area*1e9:.1f} mm²"
        else:
            area_str = f"{area:.2f} m²"
        
        print(f"   {size*1e6:6.0f} μm source → {area_str}")


def demonstrate_stellar_interferometry():
    """Demonstrate stellar interferometry principles."""
    print("\n⭐ Stellar Interferometry")
    print("=" * 25)
    
    # Stellar parameters
    wavelength = 550e-9  # Green light
    stellar_distances = [10, 100, 1000]  # parsecs (1 pc ≈ 3.09e16 m)
    pc_to_m = 3.09e16
    
    # Different types of stars
    stars = {
        "Red Giant": 100,      # Solar radii
        "Main Sequence": 1,    # Solar radii  
        "White Dwarf": 0.01    # Solar radii
    }
    
    solar_radius = 6.96e8  # meters
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Visibility vs baseline for different stars
    baseline_range = np.logspace(-1, 3, 100)  # 0.1 m to 1 km
    distance_m = 100 * pc_to_m  # 100 parsecs
    
    for star_name, radius_solar in stars.items():
        star_radius = radius_solar * solar_radius
        angular_diameter = 2 * star_radius / distance_m  # radians
        
        # Spatial coherence for circular disk source
        # |γ(r)| = |2J₁(πrθ/λ)/(πrθ/λ)| where θ is angular diameter
        visibility = []
        
        for baseline in baseline_range:
            x = np.pi * baseline * angular_diameter / wavelength
            if abs(x) < 1e-10:
                vis = 1.0
            else:
                # Bessel function approximation for small arguments
                if x < 0.1:
                    vis = 1 - (x**2)/8
                else:
                    vis = abs(2 * np.sin(x) / x)  # Simplified approximation
                vis = max(0, vis)
            
            visibility.append(vis)
        
        ax1.semilogx(baseline_range, visibility, linewidth=2, label=star_name)
        
        # Mark first null
        first_null_baseline = 1.22 * wavelength / angular_diameter
        if first_null_baseline <= baseline_range.max():
            ax1.axvline(first_null_baseline, linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Baseline Length (m)')
    ax1.set_ylabel('Fringe Visibility')
    ax1.set_title('Stellar Interferometry - Visibility vs Baseline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Angular resolution vs baseline
    resolution_arcsec = []
    
    for baseline in baseline_range:
        # Angular resolution: θ = λ/B (radians)
        resolution_rad = wavelength / baseline
        resolution_as = resolution_rad * (180 * 3600 / np.pi)  # arcseconds
        resolution_arcsec.append(resolution_as)
    
    ax2.loglog(baseline_range, resolution_arcsec, 'b-', linewidth=2)
    ax2.set_xlabel('Baseline Length (m)')
    ax2.set_ylabel('Angular Resolution (arcseconds)')
    ax2.set_title('Angular Resolution vs Baseline Length')
    ax2.grid(True, alpha=0.3)
    
    # Mark existing interferometer baselines
    interferometers = {
        "VLT Interferometer": 200,
        "Keck Interferometer": 85,
        "CHARA Array": 330
    }
    
    for name, baseline in interferometers.items():
        resolution = wavelength / baseline * (180 * 3600 / np.pi)
        ax2.plot(baseline, resolution, 'ro', markersize=8)
        ax2.annotate(name, (baseline, resolution), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔭 Stellar Interferometry Results:")
    print(f"   Angular resolution: θ = λ/B (Rayleigh criterion)")
    print(f"   For λ = {wavelength*1e9:.0f} nm:")
    
    for name, baseline in interferometers.items():
        resolution = wavelength / baseline * (180 * 3600 / np.pi)
        print(f"     {name} (B={baseline}m): {resolution:.4f} arcsec")
    
    print(f"\n⭐ Stellar Diameter Measurements:")
    distance_m = 10 * pc_to_m  # 10 parsecs (nearby stars)
    
    for star_name, radius_solar in stars.items():
        star_radius = radius_solar * solar_radius
        angular_diameter = 2 * star_radius / distance_m
        angular_diameter_mas = angular_diameter * (180 * 1000 * 3600 / np.pi)  # milliarcseconds
        
        # Baseline needed for first null
        required_baseline = 1.22 * wavelength / angular_diameter
        
        print(f"   {star_name} (at 10 pc):")
        print(f"     Angular diameter: {angular_diameter_mas:.2f} milliarcsec")
        print(f"     Baseline needed: {required_baseline:.1f} m")


def demonstrate_coherence_measurement():
    """Demonstrate coherence measurement techniques."""
    print("\n📏 Coherence Measurement Techniques")
    print("=" * 40)
    
    # Simulate interferogram from partially coherent source
    source = LightSource(633e-9, 0.1e-9, 1e-6, "laser")  # Narrow linewidth laser
    measurement = CoherenceMeasurement(source)
    
    # Path difference scan
    path_diff = np.linspace(-2e-3, 2e-3, 1000)  # ±2 mm scan
    intensity, visibility = measurement.michelson_interferometer(path_diff)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Interferogram
    ax1.plot(path_diff * 1e3, intensity, 'b-', linewidth=1, label='Interference pattern')
    ax1.plot(path_diff * 1e3, 0.5 * (1 + visibility), 'r--', linewidth=2, label='Envelope (visibility)')
    ax1.plot(path_diff * 1e3, 0.5 * (1 - visibility), 'r--', linewidth=2)
    ax1.set_xlabel('Path Difference (mm)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Michelson Interferogram for Coherence Measurement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Visibility extraction
    ax2.plot(path_diff * 1e3, visibility, 'g-', linewidth=2)
    ax2.axhline(y=1/np.e, color='red', linestyle='--', alpha=0.7, 
               label=f'1/e level = {1/np.e:.3f}')
    ax2.set_xlabel('Path Difference (mm)')
    ax2.set_ylabel('Fringe Visibility')
    ax2.set_title('Extracted Visibility Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark coherence length
    coherence_length_mm = source.coherence_length * 1e3
    ax2.axvline(coherence_length_mm, color='red', linestyle=':', linewidth=2,
               label=f'Coherence length = {coherence_length_mm:.2f} mm')
    ax2.axvline(-coherence_length_mm, color='red', linestyle=':', linewidth=2)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔬 Measurement Analysis:")
    print(f"   Source type: {source.source_type}")
    print(f"   Central wavelength: {source.lambda_0*1e9:.1f} nm")
    print(f"   Spectral width: {source.delta_lambda*1e12:.1f} pm")
    print(f"   Coherence length: {source.coherence_length*1e3:.2f} mm")
    print(f"   Coherence time: {source.coherence_time*1e15:.1f} fs")
    
    # Compare measurement with theory
    theoretical_width = source.coherence_length
    measured_width = 2 * coherence_length_mm * 1e-3  # FWHM approximation
    
    print(f"\n📊 Measurement vs Theory:")
    print(f"   Theoretical coherence length: {theoretical_width*1e3:.2f} mm")
    print(f"   Measured FWHM: {measured_width*1e3:.2f} mm")
    print(f"   Agreement: {abs(theoretical_width - measured_width)/theoretical_width*100:.1f}% difference")


def demonstrate_applications():
    """Demonstrate practical applications of coherence."""
    print("\n🚀 Coherence Applications")
    print("=" * 25)
    
    print(f"\n🔬 Scientific Applications:")
    print(f"   • Optical Coherence Tomography (OCT)")
    print(f"     - Medical imaging with μm resolution")
    print(f"     - Uses low-coherence interferometry")
    print(f"   • Stellar interferometry")
    print(f"     - Measure stellar diameters and binary separations")
    print(f"     - Requires long baseline coherence")
    print(f"   • Fourier Transform Spectroscopy")
    print(f"     - High-resolution spectroscopy")
    print(f"     - Based on temporal coherence measurement")
    
    print(f"\n🏭 Industrial Applications:")
    print(f"   • Laser processing")
    print(f"     - Coherence affects beam quality and focusing")
    print(f"   • Holography")
    print(f"     - Requires high temporal and spatial coherence")
    print(f"   • Optical communications")
    print(f"     - Coherent detection improves sensitivity")
    
    # OCT simulation example
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # OCT source characteristics
    oct_wavelength = 1310e-9  # Near-infrared
    oct_bandwidth = 100e-9    # Broad bandwidth for high resolution
    oct_source = LightSource(oct_wavelength, oct_bandwidth, 50e-6, "LED")
    
    # Depth resolution
    depth_resolution = oct_source.coherence_length / (2 * 1.4)  # Factor of 2 for round trip, 1.4 for tissue refractive index
    
    # Simulate OCT A-scan
    depths = np.linspace(0, 2e-3, 1000)  # 0 to 2 mm depth
    path_differences = 2 * depths * 1.4  # Round trip in tissue
    
    # Multiple reflectors at different depths
    reflector_depths = [0.2e-3, 0.8e-3, 1.5e-3]  # Different tissue layers
    reflector_strengths = [0.8, 0.6, 0.4]
    
    oct_signal = np.zeros_like(depths)
    for depth, strength in zip(reflector_depths, reflector_strengths):
        # Gaussian envelope centered at reflector
        envelope = np.exp(-((depths - depth) / (depth_resolution/2))**2)
        oct_signal += strength * envelope
    
    ax1.plot(depths * 1e3, oct_signal, 'b-', linewidth=2)
    ax1.set_xlabel('Depth (mm)')
    ax1.set_ylabel('OCT Signal')
    ax1.set_title(f'OCT A-scan (Δλ = {oct_bandwidth*1e9:.0f} nm)')
    ax1.grid(True, alpha=0.3)
    
    # Mark theoretical resolution
    ax1.text(0.02, 0.95, f'Axial resolution:\n{depth_resolution*1e6:.0f} μm', 
            transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Bandwidth vs resolution trade-off
    bandwidths = np.linspace(10e-9, 200e-9, 100)
    resolutions = []
    
    for bw in bandwidths:
        source_temp = LightSource(oct_wavelength, bw, 50e-6, "LED")
        res = source_temp.coherence_length / (2 * 1.4) * 1e6  # Convert to μm
        resolutions.append(res)
    
    ax2.plot(bandwidths * 1e9, resolutions, 'r-', linewidth=2)
    ax2.set_xlabel('Spectral Bandwidth (nm)')
    ax2.set_ylabel('Axial Resolution (μm)')
    ax2.set_title('OCT Resolution vs Source Bandwidth')
    ax2.grid(True, alpha=0.3)
    
    # Mark current bandwidth
    ax2.axvline(oct_bandwidth * 1e9, color='blue', linestyle='--', 
               label=f'Current: {oct_bandwidth*1e9:.0f} nm')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🏥 OCT Performance Example:")
    print(f"   Wavelength: {oct_wavelength*1e9:.0f} nm")
    print(f"   Bandwidth: {oct_bandwidth*1e9:.0f} nm")
    print(f"   Coherence length: {oct_source.coherence_length*1e6:.0f} μm")
    print(f"   Axial resolution in tissue: {depth_resolution*1e6:.0f} μm")
    print(f"   Imaging depth: ~2-3 mm")


if __name__ == "__main__":
    print("🌟 Welcome to Coherence Explorer! 🌟")
    print("This script demonstrates the coherence properties of light and their applications.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_temporal_coherence()
        demonstrate_spatial_coherence()
        demonstrate_stellar_interferometry()
        demonstrate_coherence_measurement()
        demonstrate_applications()
        
        print(f"\n" + "="*70)
        print("🎓 Coherence Summary:")
        print("="*70)
        print("Key Concepts:")
        print("• Temporal coherence: Related to spectral purity (Δt ≈ 1/Δν)")
        print("• Spatial coherence: Related to source size (van Cittert-Zernike)")
        print("• Coherence length: Lc = λ²/Δλ (temporal)")
        print("• Coherence area: Ac ≈ (λD/d)² (spatial)")
        
        print(f"\nMeasurement Techniques:")
        print("• Michelson interferometer for temporal coherence")
        print("• Young's double-slit for spatial coherence")
        print("• Stellar interferometry for angular diameter")
        print("• OCT for biological tissue imaging")
        
        print(f"\nSource Comparison (typical values):")
        print("• Laser: Lc ~ mm to km, excellent coherence")
        print("• LED: Lc ~ μm to mm, moderate coherence")
        print("• Thermal: Lc ~ μm, poor coherence")
        print("• Sunlight: Lc ~ few μm, very poor coherence")
        
        print(f"\n🔬 Applications:")
        print("• Precision metrology and interferometry")
        print("• Medical imaging (OCT)")
        print("• Astronomy (stellar interferometry)")
        print("• Holography and 3D imaging")
        print("• Optical communications")
        print("• Spectroscopy and material analysis")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring coherence phenomena! ✨")