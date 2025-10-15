#!/usr/bin/env python3
"""
Wave Optics: Diffraction Patterns
=================================

This module demonstrates various types of diffraction patterns:
- Single-slit diffraction (Fraunhofer and Fresnel)
- Circular aperture diffraction (Airy disk)
- Diffraction gratings (transmission and reflection)
- Multiple-slit diffraction
- Fresnel zones and zone plates
- Edge diffraction and shadow patterns
- Applications in optics and spectroscopy

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
from scipy import special
from typing import Tuple, List, Optional
import math


class SingleSlit:
    """Single-slit diffraction calculator."""
    
    def __init__(self, width: float, wavelength: float, screen_distance: float):
        self.a = width                # Slit width
        self.wavelength = wavelength  # Wavelength
        self.L = screen_distance     # Distance to screen
        self.k = 2 * np.pi / wavelength
    
    def fraunhofer_pattern(self, y_screen: np.ndarray) -> np.ndarray:
        """Calculate Fraunhofer (far-field) diffraction pattern."""
        # Angular position
        theta = np.arctan(y_screen / self.L)
        
        # Diffraction parameter
        beta = self.k * self.a * np.sin(theta) / 2
        
        # Intensity pattern
        intensity = np.ones_like(beta)
        mask = np.abs(beta) > 1e-10
        intensity[mask] = (np.sin(beta[mask]) / beta[mask])**2
        
        return intensity
    
    def fresnel_pattern(self, y_screen: np.ndarray) -> np.ndarray:
        """Calculate Fresnel (near-field) diffraction pattern using Fresnel integrals."""
        # Fresnel number
        fresnel_param = np.sqrt(2 / (self.wavelength * self.L))
        
        # Fresnel parameters
        u1 = fresnel_param * (y_screen + self.a/2)
        u2 = fresnel_param * (y_screen - self.a/2)
        
        # Fresnel integrals
        s1, c1 = special.fresnel(u1)
        s2, c2 = special.fresnel(u2)
        
        # Complex amplitude
        amplitude_real = c2 - c1
        amplitude_imag = s2 - s1
        
        # Intensity
        intensity = amplitude_real**2 + amplitude_imag**2
        
        return intensity
    
    def plot_comparison(self, y_range: float = 0.01):
        """Compare Fraunhofer and Fresnel patterns."""
        y_screen = np.linspace(-y_range, y_range, 1000)
        
        intensity_fraunhofer = self.fraunhofer_pattern(y_screen)
        intensity_fresnel = self.fresnel_pattern(y_screen)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Fraunhofer pattern
        ax1.plot(y_screen*1000, intensity_fraunhofer, 'b-', linewidth=2, label='Fraunhofer')
        ax1.set_xlabel('Position (mm)')
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Single-Slit Diffraction (a={self.a*1e6:.1f}μm, λ={self.wavelength*1e9:.0f}nm)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Mark minima positions
        # Minima at: a sin(θ) = mλ, m = ±1, ±2, ...
        for m in range(1, 4):
            if m * self.wavelength < self.a:
                theta_min = np.arcsin(m * self.wavelength / self.a)
                y_min = self.L * np.tan(theta_min)
                if y_min < y_range:
                    ax1.axvline(y_min*1000, color='red', linestyle='--', alpha=0.7)
                    ax1.axvline(-y_min*1000, color='red', linestyle='--', alpha=0.7)
        
        # Fresnel pattern
        ax2.plot(y_screen*1000, intensity_fresnel, 'r-', linewidth=2, label='Fresnel')
        ax2.set_xlabel('Position (mm)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Near-Field (Fresnel) Diffraction')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig, (ax1, ax2)


class CircularAperture:
    """Circular aperture diffraction (Airy disk)."""
    
    def __init__(self, diameter: float, wavelength: float, focal_length: float):
        self.D = diameter            # Aperture diameter
        self.wavelength = wavelength # Wavelength
        self.f = focal_length       # Focal length
    
    def airy_pattern(self, r: np.ndarray) -> np.ndarray:
        """Calculate Airy diffraction pattern."""
        # Numerical aperture-like parameter
        k = 2 * np.pi / self.wavelength
        
        # Radial diffraction parameter
        x = k * self.D * r / (2 * self.f)
        
        # Airy pattern: I(x) = (2J₁(x)/x)²
        intensity = np.ones_like(x)
        mask = np.abs(x) > 1e-10
        
        # Use Bessel function of first kind, order 1
        j1_values = special.j1(x[mask])
        intensity[mask] = (2 * j1_values / x[mask])**2
        
        return intensity
    
    def airy_disk_radius(self) -> float:
        """Calculate radius of first Airy disk minimum."""
        # First zero of J₁: x ≈ 3.832
        return 1.22 * self.wavelength * self.f / self.D
    
    def plot_airy_pattern(self, r_max: Optional[float] = None):
        """Plot 2D Airy pattern."""
        if r_max is None:
            r_max = 3 * self.airy_disk_radius()
        
        # Create radial coordinate
        r = np.linspace(0, r_max, 500)
        intensity_radial = self.airy_pattern(r)
        
        # Create 2D pattern
        x = np.linspace(-r_max, r_max, 300)
        y = np.linspace(-r_max, r_max, 300)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        intensity_2d = self.airy_pattern(R)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 2D Airy pattern
        im = ax1.imshow(intensity_2d, extent=[-r_max*1e6, r_max*1e6, -r_max*1e6, r_max*1e6],
                       cmap='hot', origin='lower')
        ax1.set_xlabel('x (μm)')
        ax1.set_ylabel('y (μm)')
        ax1.set_title(f'Airy Disk Pattern (D={self.D*1e3:.1f}mm)')
        
        # Mark Airy disk boundary
        airy_radius = self.airy_disk_radius()
        circle = Circle((0, 0), airy_radius*1e6, fill=False, color='blue', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(0, airy_radius*1e6*1.2, f'Airy disk\nr = {airy_radius*1e6:.1f} μm', 
                ha='center', va='bottom', color='blue', fontweight='bold')
        
        plt.colorbar(im, ax=ax1, label='Intensity')
        
        # Radial profile
        ax2.plot(r*1e6, intensity_radial, 'b-', linewidth=2)
        ax2.axvline(airy_radius*1e6, color='red', linestyle='--', linewidth=2, 
                   label=f'Airy disk edge')
        ax2.set_xlabel('Radius (μm)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Radial Intensity Profile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig, (ax1, ax2)


class DiffractionGrating:
    """Diffraction grating analysis."""
    
    def __init__(self, line_density: float, wavelength: float, 
                 slit_width: Optional[float] = None):
        self.N_per_mm = line_density    # Lines per mm
        self.d = 1e-3 / line_density    # Grating spacing
        self.wavelength = wavelength    # Wavelength
        self.slit_width = slit_width    # Individual slit width (optional)
    
    def grating_equation_angles(self, max_order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate diffraction angles using grating equation."""
        orders = np.arange(-max_order, max_order + 1)
        
        # Grating equation: d sin(θ) = mλ
        sin_theta = orders * self.wavelength / self.d
        
        # Filter physically realizable angles (|sin θ| ≤ 1)
        valid = np.abs(sin_theta) <= 1
        orders_valid = orders[valid]
        sin_theta_valid = sin_theta[valid]
        angles = np.degrees(np.arcsin(sin_theta_valid))
        
        return orders_valid, angles
    
    def grating_intensity_pattern(self, theta: np.ndarray, num_slits: int = 100) -> np.ndarray:
        """Calculate intensity pattern for N-slit grating."""
        k = 2 * np.pi / self.wavelength
        
        # Phase difference between adjacent slits
        beta = k * self.d * np.sin(np.radians(theta)) / 2
        
        # N-slit interference pattern
        # I = sin²(Nβ)/sin²(β)
        intensity = np.ones_like(beta)
        mask = np.abs(beta) > 1e-10
        
        sin_beta = np.sin(beta[mask])
        sin_N_beta = np.sin(num_slits * beta[mask])
        intensity[mask] = (sin_N_beta / sin_beta)**2 / num_slits**2
        
        # Include single-slit envelope if slit width is specified
        if self.slit_width is not None:
            alpha = k * self.slit_width * np.sin(np.radians(theta)) / 2
            envelope = np.ones_like(alpha)
            mask_env = np.abs(alpha) > 1e-10
            envelope[mask_env] = (np.sin(alpha[mask_env]) / alpha[mask_env])**2
            intensity *= envelope
        
        return intensity
    
    def plot_grating_spectrum(self, theta_range: float = 60):
        """Plot diffraction grating spectrum."""
        theta = np.linspace(-theta_range, theta_range, 2000)
        intensity = self.grating_intensity_pattern(theta)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Full intensity pattern
        ax1.plot(theta, intensity, 'b-', linewidth=1)
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Diffraction Grating ({self.N_per_mm:.0f} lines/mm, λ={self.wavelength*1e9:.0f}nm)')
        ax1.grid(True, alpha=0.3)
        
        # Mark principal maxima
        orders, angles = self.grating_equation_angles()
        for order, angle in zip(orders, angles):
            ax1.axvline(angle, color='red', linestyle='--', alpha=0.7)
            ax1.text(angle, 0.9, f'm={order}', rotation=90, ha='right', va='bottom')
        
        # Zoom in on central region
        central_range = min(30, theta_range/2)
        mask_central = np.abs(theta) <= central_range
        
        ax2.plot(theta[mask_central], intensity[mask_central], 'b-', linewidth=2)
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Central Region Detail')
        ax2.grid(True, alpha=0.3)
        
        # Mark orders in central region
        central_orders = orders[np.abs(angles) <= central_range]
        central_angles = angles[np.abs(angles) <= central_range]
        
        for order, angle in zip(central_orders, central_angles):
            ax2.axvline(angle, color='red', linestyle='--', alpha=0.7)
            ax2.text(angle, 0.8, f'm={order}', rotation=90, ha='right', va='bottom')
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def spectral_resolution(self, order: int, num_lines: int) -> float:
        """Calculate spectral resolution R = λ/Δλ."""
        return order * num_lines
    
    def dispersion_analysis(self, wavelength_range: Tuple[float, float], order: int = 1):
        """Analyze angular dispersion of the grating."""
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], 100)
        
        # Calculate angles for each wavelength
        angles = []
        for wl in wavelengths:
            sin_theta = order * wl / self.d
            if abs(sin_theta) <= 1:
                angles.append(np.degrees(np.arcsin(sin_theta)))
            else:
                angles.append(np.nan)
        
        angles = np.array(angles)
        
        # Angular dispersion: dθ/dλ
        valid_mask = ~np.isnan(angles)
        if np.sum(valid_mask) > 1:
            dispersion = np.gradient(angles[valid_mask], wavelengths[valid_mask])
        else:
            dispersion = np.array([])
        
        return wavelengths[valid_mask], angles[valid_mask], dispersion


def demonstrate_single_slit_diffraction():
    """Demonstrate single-slit diffraction patterns."""
    print("📏 Single-Slit Diffraction")
    print("=" * 30)
    
    # Parameters
    wavelength = 632.8e-9  # HeNe laser
    slit_width = 50e-6     # 50 micrometers
    screen_distance = 2.0   # 2 meters
    
    slit = SingleSlit(slit_width, wavelength, screen_distance)
    
    # Plot patterns
    fig, axes = slit.plot_comparison()
    plt.show()
    
    # Calculate key parameters
    angular_width = 2 * wavelength / slit_width  # First minimum to first minimum
    linear_width = angular_width * screen_distance
    
    print(f"\n📊 Single-Slit Parameters:")
    print(f"   Slit width: {slit_width*1e6:.1f} μm")
    print(f"   Wavelength: {wavelength*1e9:.0f} nm")
    print(f"   Central maximum width: {linear_width*1e3:.2f} mm")
    print(f"   Angular width: {np.degrees(angular_width)*3600:.1f} arcseconds")
    
    # Fresnel number
    fresnel_number = slit_width**2 / (wavelength * screen_distance)
    print(f"   Fresnel number: {fresnel_number:.2f}")


def demonstrate_airy_disk():
    """Demonstrate Airy disk diffraction pattern."""
    print("\n🔵 Airy Disk (Circular Aperture)")
    print("=" * 35)
    
    # Telescope parameters
    diameter = 10e-3       # 10 mm aperture
    wavelength = 550e-9    # Green light
    focal_length = 100e-3  # 100 mm focal length
    
    aperture = CircularAperture(diameter, wavelength, focal_length)
    
    # Plot Airy pattern
    fig, axes = aperture.plot_airy_pattern()
    plt.show()
    
    # Calculate resolution parameters
    airy_radius = aperture.airy_disk_radius()
    angular_resolution = 1.22 * wavelength / diameter  # Rayleigh criterion
    
    print(f"\n🔍 Airy Disk Analysis:")
    print(f"   Aperture diameter: {diameter*1e3:.1f} mm")
    print(f"   Focal length: {focal_length*1e3:.1f} mm")
    print(f"   Airy disk radius: {airy_radius*1e6:.2f} μm")
    print(f"   Angular resolution: {np.degrees(angular_resolution)*3600:.2f} arcseconds")
    
    # Compare with telescope resolution
    print(f"\n🔭 Telescope Resolution Comparison:")
    telescopes = {
        "Human eye": 2e-3,      # 2 mm pupil
        "Binoculars": 50e-3,    # 50 mm
        "Amateur telescope": 200e-3,  # 8 inch
        "Hubble": 2.4           # 2.4 meters
    }
    
    for name, D in telescopes.items():
        res = 1.22 * wavelength / D
        print(f"   {name}: {np.degrees(res)*3600:.2f} arcsec")


def demonstrate_diffraction_grating():
    """Demonstrate diffraction grating spectroscopy."""
    print("\n🌈 Diffraction Grating Spectroscopy")
    print("=" * 40)
    
    # Grating parameters
    line_density = 600      # 600 lines/mm
    wavelength = 589e-9     # Sodium D-line
    
    grating = DiffractionGrating(line_density, wavelength, slit_width=1e-6)
    
    # Plot spectrum
    fig, axes = grating.plot_grating_spectrum()
    plt.show()
    
    # Calculate diffraction orders and angles
    orders, angles = grating.grating_equation_angles()
    
    print(f"\n📐 Grating Analysis:")
    print(f"   Line density: {line_density} lines/mm")
    print(f"   Grating spacing: {grating.d*1e6:.2f} μm")
    print(f"   Wavelength: {wavelength*1e9:.0f} nm")
    
    print(f"\n   Diffraction Orders:")
    for order, angle in zip(orders, angles):
        print(f"     m = {order:2d}: θ = {angle:6.2f}°")
    
    # Spectral resolution analysis
    num_lines_illuminated = 10e-3 / grating.d  # 10 mm beam width
    resolution_1st = grating.spectral_resolution(1, num_lines_illuminated)
    resolution_2nd = grating.spectral_resolution(2, num_lines_illuminated)
    
    print(f"\n🔍 Spectral Resolution (10 mm beam):")
    print(f"   1st order: R = {resolution_1st:.0f}")
    print(f"   2nd order: R = {resolution_2nd:.0f}")
    print(f"   Δλ (1st order): {wavelength/resolution_1st*1e12:.2f} pm")


def demonstrate_multiple_wavelengths():
    """Demonstrate grating dispersion with multiple wavelengths."""
    print("\n🎨 Grating Dispersion - Multiple Wavelengths")
    print("=" * 50)
    
    # Visible spectrum
    wavelengths = np.array([400e-9, 500e-9, 600e-9, 700e-9])  # Violet to Red
    colors = ['violet', 'green', 'orange', 'red']
    line_density = 1200  # High resolution grating
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for wl, color in zip(wavelengths, colors):
        grating = DiffractionGrating(line_density, wl)
        theta = np.linspace(-30, 30, 1000)
        intensity = grating.grating_intensity_pattern(theta, num_slits=50)
        
        # Plot with appropriate color and offset
        offset = (wl - 550e-9) / 50e-9  # Offset for visualization
        ax.plot(theta, intensity + offset, color=color, linewidth=2, 
               label=f'{wl*1e9:.0f} nm')
        
        # Mark first-order peaks
        orders, angles = grating.grating_equation_angles(1)
        first_order_angle = angles[orders == 1]
        if len(first_order_angle) > 0:
            ax.axvline(first_order_angle[0], color=color, linestyle='--', alpha=0.7)
            ax.axvline(-first_order_angle[0], color=color, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Intensity (offset)')
    ax.set_title(f'Spectral Dispersion ({line_density} lines/mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    # Calculate angular dispersion
    grating_ref = DiffractionGrating(line_density, 550e-9)  # Reference
    wl_range = (400e-9, 700e-9)
    wavelengths_disp, angles_disp, dispersion = grating_ref.dispersion_analysis(wl_range, order=1)
    
    print(f"\n📈 Angular Dispersion Analysis:")
    print(f"   Grating: {line_density} lines/mm")
    print(f"   Average dispersion (1st order): {np.mean(dispersion):.2f} deg/nm")
    print(f"   Spectral range (±30°): {wavelengths_disp[0]*1e9:.0f}-{wavelengths_disp[-1]*1e9:.0f} nm")


def demonstrate_fresnel_zones():
    """Demonstrate Fresnel zone analysis."""
    print("\n🎯 Fresnel Zone Analysis")
    print("=" * 30)
    
    # Parameters
    wavelength = 500e-9
    distance = 1.0      # 1 meter
    
    # Calculate Fresnel zone radii
    zones = np.arange(1, 11)
    radii = np.sqrt(zones * wavelength * distance / 2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Fresnel zones
    for i, radius in enumerate(radii[:5]):  # First 5 zones
        circle = Circle((0, 0), radius*1000, fill=False, linewidth=2, 
                       color=f'C{i}', label=f'Zone {i+1}')
        ax1.add_patch(circle)
    
    ax1.set_xlim(-radii[4]*1000*1.1, radii[4]*1000*1.1)
    ax1.set_ylim(-radii[4]*1000*1.1, radii[4]*1000*1.1)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Fresnel Zones (First 5)')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zone radii vs zone number
    ax2.plot(zones, radii*1000, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Zone Number')
    ax2.set_ylabel('Zone Radius (mm)')
    ax2.set_title(f'Fresnel Zone Radii (λ={wavelength*1e9:.0f}nm, z={distance}m)')
    ax2.grid(True, alpha=0.3)
    
    # Show square root dependence
    theory_line = np.sqrt(zones) * np.sqrt(wavelength * distance / 2) * 1000
    ax2.plot(zones, theory_line, 'r--', label='√n dependence')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Fresnel Zone Parameters:")
    print(f"   Wavelength: {wavelength*1e9:.0f} nm")
    print(f"   Distance: {distance:.1f} m")
    print(f"   First zone radius: {radii[0]*1e3:.2f} mm")
    print(f"   Zone area: {np.pi * radii[0]**2 * 1e6:.2f} mm²")
    
    print(f"\n💡 Applications:")
    print(f"   • Zone plates for X-ray focusing")
    print(f"   • Fresnel lenses for lighthouses")
    print(f"   • Radio telescope design")
    print(f"   • Diffraction analysis")


if __name__ == "__main__":
    print("🔍 Welcome to Diffraction Patterns Explorer! 🔍")
    print("This script demonstrates various types of optical diffraction phenomena.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_single_slit_diffraction()
        demonstrate_airy_disk()
        demonstrate_diffraction_grating()
        demonstrate_multiple_wavelengths()
        demonstrate_fresnel_zones()
        
        print(f"\n" + "="*70)
        print("🎓 Diffraction Patterns Summary:")
        print("="*70)
        print("Key Diffraction Types:")
        print("• Single-slit: I ∝ (sin β/β)² where β = ka sin θ/2")
        print("• Circular aperture: I ∝ (2J₁(x)/x)² (Airy pattern)")
        print("• Grating: Multiple-beam interference with envelope")
        print("• Fresnel zones: Near-field diffraction analysis")
        
        print(f"\n🔬 Resolution Limits:")
        print("• Rayleigh criterion: θ = 1.22λ/D (circular)")
        print("• Grating resolution: R = mN (order × number of lines)")
        print("• Single-slit width: Δθ = 2λ/a")
        
        print(f"\n🌟 Applications:")
        print("• Spectroscopy and wavelength measurement")
        print("• Optical resolution limits and design")
        print("• X-ray crystallography")
        print("• Radio astronomy and radar")
        print("• Holography and optical processing")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring diffraction patterns! ✨")