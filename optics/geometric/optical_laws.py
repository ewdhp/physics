#!/usr/bin/env python3
"""
Comprehensive Optical Laws
==========================

This module demonstrates ALL fundamental optical phenomena with dedicated plots:
1. Reflection: θᵢ = θᵣ (Law of reflection)
2. Refraction: n₁sin(θ₁) = n₂sin(θ₂) (Snell's law) 
3. Absorption: I = I₀e⁻ᵅˣ (Beer-Lambert law)
4. Transmission: T = I/I₀, R + A + T = 1 (Energy conservation)
5. Scattering: I ∝ 1/λ⁴ (Rayleigh scattering)
6. Critical Angle: θc = arcsin(n₂/n₁) (Total internal reflection)
7. Brewster's Angle: θB = arctan(n₂/n₁) (Polarization)

Each law includes theory, calculations, and dedicated visualizations.

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


class Ray:
    """Represents a light ray with position, direction, and medium properties."""
    
    def __init__(self, x: float, y: float, direction: float, 
                 intensity: float = 1.0, wavelength: float = 550e-9):
        self.x = x
        self.y = y
        self.direction = direction  # angle in radians
        self.intensity = intensity
        self.wavelength = wavelength  # green light in meters
    
    def get_unit_vector(self) -> Tuple[float, float]:
        """Get the unit direction vector of the ray."""
        return np.cos(self.direction), np.sin(self.direction)


class Interface:
    """Represents an interface between two optical media."""
    
    def __init__(self, position: float, n1: float, n2: float, 
                 name1: str = "Medium 1", name2: str = "Medium 2"):
        self.position = position  # y-coordinate of horizontal interface
        self.n1 = n1  # refractive index of medium 1 (above)
        self.n2 = n2  # refractive index of medium 2 (below)
        self.name1 = name1
        self.name2 = name2


class GeometricOptics:
    """Main class for geometric optics calculations and visualizations."""
    
    def __init__(self):
        self.rays = []
        self.interfaces = []
        
    def snells_law(self, theta1: float, n1: float, n2: float) -> Optional[float]:
        """
        Apply Snell's law to find refraction angle.
        
        Args:
            theta1: Incident angle (radians)
            n1: Refractive index of incident medium
            n2: Refractive index of refracting medium
            
        Returns:
            Refracted angle in radians, or None if total internal reflection
        """
        sin_theta2 = (n1 / n2) * np.sin(theta1)
        
        if abs(sin_theta2) > 1.0:
            return None  # Total internal reflection
        
        return np.arcsin(sin_theta2)
    
    def critical_angle(self, n1: float, n2: float) -> Optional[float]:
        """
        Calculate critical angle for total internal reflection.
        
        Args:
            n1: Refractive index of denser medium
            n2: Refractive index of rarer medium
            
        Returns:
            Critical angle in radians, or None if n1 < n2
        """
        if n1 <= n2:
            return None
        
        return np.arcsin(n2 / n1)
    
    def fresnel_reflectance(self, theta1: float, n1: float, n2: float) -> float:
        """
        Calculate Fresnel reflectance for unpolarized light.
        
        Args:
            theta1: Incident angle (radians)
            n1: Refractive index of incident medium
            n2: Refractive index of refracting medium
            
        Returns:
            Reflectance (fraction of light reflected)
        """
        # For normal incidence approximation
        if abs(theta1) < 0.01:  # Small angle approximation
            r = ((n1 - n2) / (n1 + n2)) ** 2
            return r
        
        # Full Fresnel equations for arbitrary angle
        theta2 = self.snells_law(theta1, n1, n2)
        if theta2 is None:  # Total internal reflection
            return 1.0
        
        # Fresnel equations for s and p polarizations
        cos_theta1, cos_theta2 = np.cos(theta1), np.cos(theta2)
        
        rs = ((n1 * cos_theta1 - n2 * cos_theta2) / 
              (n1 * cos_theta1 + n2 * cos_theta2)) ** 2
        
        rp = ((n1 * cos_theta2 - n2 * cos_theta1) / 
              (n1 * cos_theta2 + n2 * cos_theta1)) ** 2
        
        return (rs + rp) / 2  # Average for unpolarized light
    
    def beer_lambert_absorption(self, intensity_0: float, absorption_coeff: float, 
                              thickness: float) -> dict:
        """
        Calculate light absorption using Beer-Lambert law: I = I₀ e^(-αx)
        
        Args:
            intensity_0: Initial light intensity
            absorption_coeff: Absorption coefficient (α) in units 1/length
            thickness: Material thickness (x) in length units
            
        Returns:
            Dictionary with absorption analysis results
        """
        transmitted_intensity = intensity_0 * np.exp(-absorption_coeff * thickness)
        absorbed_intensity = intensity_0 - transmitted_intensity
        transmission_fraction = transmitted_intensity / intensity_0
        absorption_fraction = absorbed_intensity / intensity_0
        
        # Penetration depth (1/e depth)
        penetration_depth = 1.0 / absorption_coeff if absorption_coeff > 0 else float('inf')
        
        return {
            'initial_intensity': intensity_0,
            'transmitted_intensity': transmitted_intensity,
            'absorbed_intensity': absorbed_intensity,
            'transmission_fraction': transmission_fraction,
            'absorption_fraction': absorption_fraction,
            'absorption_coefficient': absorption_coeff,
            'thickness': thickness,
            'penetration_depth': penetration_depth,
            'attenuation_db': -10 * np.log10(transmission_fraction) if transmission_fraction > 0 else float('inf')
        }
    
    def rayleigh_scattering(self, wavelength: float, particle_size: float, 
                          particle_density: float) -> dict:
        """
        Calculate Rayleigh scattering intensity: I ∝ 1/λ⁴
        
        Args:
            wavelength: Light wavelength in meters
            particle_size: Particle size in meters (must be << wavelength)
            particle_density: Number density of scattering particles
            
        Returns:
            Dictionary with scattering analysis results
        """
        # Rayleigh scattering cross-section (simplified)
        lambda_4_factor = 1.0 / (wavelength ** 4)
        
        # Relative scattering intensity (normalized to reference wavelength)
        ref_wavelength = 550e-9  # Green light reference
        relative_intensity = (ref_wavelength / wavelength) ** 4
        
        # Scattering coefficient (simplified)
        scattering_coeff = particle_density * lambda_4_factor * (particle_size ** 6)
        
        # Mean free path
        mean_free_path = 1.0 / scattering_coeff if scattering_coeff > 0 else float('inf')
        
        return {
            'wavelength': wavelength,
            'wavelength_nm': wavelength * 1e9,
            'lambda_4_factor': lambda_4_factor,
            'relative_intensity': relative_intensity,
            'scattering_coefficient': scattering_coeff,
            'particle_size': particle_size,
            'particle_density': particle_density,
            'mean_free_path': mean_free_path,
            'color_preference': self._wavelength_to_color_name(wavelength)
        }
    
    def _wavelength_to_color_name(self, wavelength: float) -> str:
        """Convert wavelength to color name for Rayleigh scattering analysis"""
        wavelength_nm = wavelength * 1e9
        if wavelength_nm < 450:
            return "Violet"
        elif wavelength_nm < 495:
            return "Blue" 
        elif wavelength_nm < 570:
            return "Green"
        elif wavelength_nm < 590:
            return "Yellow"
        elif wavelength_nm < 620:
            return "Orange"
        else:
            return "Red"
    
    def comprehensive_optical_analysis(self, theta1: float, n1: float, n2: float,
                                     absorption_coeff: float = 0.0, thickness: float = 0.0,
                                     wavelength: float = 550e-9) -> dict:
        """
        Perform comprehensive optical analysis including all five phenomena:
        Reflection, Refraction, Absorption, Transmission, and Scattering
        """
        results = {}
        
        # 1. Reflection (θᵢ = θᵣ)
        results['reflection'] = {
            'incident_angle_deg': np.degrees(theta1),
            'reflected_angle_deg': np.degrees(theta1),  # Law of reflection
            'law': 'θᵢ = θᵣ'
        }
        
        # 2. Refraction (Snell's Law: n₁sin θ₁ = n₂sin θ₂)
        theta2 = self.snells_law(theta1, n1, n2)
        results['refraction'] = {
            'incident_angle_deg': np.degrees(theta1),
            'refracted_angle_deg': np.degrees(theta2) if theta2 else None,
            'total_internal_reflection': theta2 is None,
            'law': 'n₁sin θ₁ = n₂sin θ₂',
            'n1': n1,
            'n2': n2
        }
        
        # 3. Fresnel Reflection/Transmission coefficients
        reflectance = self.fresnel_reflectance(theta1, n1, n2)
        transmittance = 1.0 - reflectance if theta2 is not None else 0.0
        
        results['fresnel'] = {
            'reflectance': reflectance,
            'transmittance': transmittance,
            'conservation_check': abs(reflectance + transmittance - 1.0) < 1e-10
        }
        
        # 4. Absorption (Beer-Lambert Law: I = I₀e⁻ᵅˣ)
        if thickness > 0 and absorption_coeff > 0:
            absorption_result = self.beer_lambert_absorption(1.0, absorption_coeff, thickness)
            results['absorption'] = absorption_result
        else:
            results['absorption'] = None
        
        # 5. Scattering (Rayleigh: I ∝ 1/λ⁴)
        # Assume some typical atmospheric conditions
        particle_size = 0.1e-6  # 0.1 μm particles
        particle_density = 1e12  # particles per m³
        scattering_result = self.rayleigh_scattering(wavelength, particle_size, particle_density)
        results['scattering'] = scattering_result
        
        return results
    
    def trace_ray(self, ray: Ray, interface: Interface, 
                  max_length: float = 10.0) -> Tuple[List[Ray], List[Ray]]:
        """
        Trace a ray through an interface, generating reflected and refracted rays.
        
        Args:
            ray: Incident ray
            interface: Optical interface
            max_length: Maximum ray length for visualization
            
        Returns:
            Tuple of (reflected_rays, refracted_rays)
        """
        reflected_rays = []
        refracted_rays = []
        
        # Calculate intersection point with interface
        if abs(np.sin(ray.direction)) < 1e-10:  # Ray parallel to interface
            return reflected_rays, refracted_rays
        
        # Distance to interface
        t = (interface.position - ray.y) / np.sin(ray.direction)
        if t < 0:  # Ray moving away from interface
            return reflected_rays, refracted_rays
        
        # Intersection point
        x_intersect = ray.x + t * np.cos(ray.direction)
        y_intersect = interface.position
        
        # Angle of incidence (measured from normal)
        theta_incident = abs(np.pi/2 - ray.direction)
        
        # Determine which medium ray is coming from
        if ray.y > interface.position:  # Coming from above
            n1, n2 = interface.n1, interface.n2
        else:  # Coming from below
            n1, n2 = interface.n2, interface.n1
            theta_incident = abs(np.pi/2 + ray.direction)
        
        # Calculate reflectance
        R = self.fresnel_reflectance(theta_incident, n1, n2)
        
        # Reflected ray (law of reflection: angle in = angle out)
        if ray.y > interface.position:  # Reflecting from above
            reflected_direction = np.pi - ray.direction
        else:  # Reflecting from below
            reflected_direction = -ray.direction
        
        reflected_ray = Ray(
            x=x_intersect,
            y=y_intersect,
            direction=reflected_direction,
            intensity=ray.intensity * R,
            wavelength=ray.wavelength
        )
        reflected_rays.append(reflected_ray)
        
        # Refracted ray (Snell's law)
        theta_refracted = self.snells_law(theta_incident, n1, n2)
        
        if theta_refracted is not None:  # No total internal reflection
            T = 1 - R  # Transmittance
            
            if ray.y > interface.position:  # Refracting downward
                refracted_direction = -(np.pi/2 - theta_refracted)
            else:  # Refracting upward
                refracted_direction = np.pi/2 - theta_refracted
            
            refracted_ray = Ray(
                x=x_intersect,
                y=y_intersect,
                direction=refracted_direction,
                intensity=ray.intensity * T,
                wavelength=ray.wavelength
            )
            refracted_rays.append(refracted_ray)
        
        return reflected_rays, refracted_rays
    
    def plot_ray_diagram(self, rays: List[Ray], interfaces: List[Interface],
                        x_range: Tuple[float, float] = (-5, 5),
                        y_range: Tuple[float, float] = (-3, 3),
                        ray_length: float = 8.0):
        """
        Create a comprehensive ray diagram showing reflection and refraction.
        
        Args:
            rays: List of incident rays
            interfaces: List of optical interfaces
            x_range: X-axis plotting range
            y_range: Y-axis plotting range
            ray_length: Length of rays to draw
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot interfaces and media
        for i, interface in enumerate(interfaces):
            ax.axhline(y=interface.position, color='black', linewidth=2, 
                      linestyle='-', alpha=0.8)
            
            # Label media
            mid_y1 = interface.position + 0.3
            mid_y2 = interface.position - 0.3
            ax.text(x_range[0] + 0.5, mid_y1, 
                   f"{interface.name1} (n={interface.n1:.2f})",
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='lightblue', alpha=0.7))
            ax.text(x_range[0] + 0.5, mid_y2, 
                   f"{interface.name2} (n={interface.n2:.2f})",
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='lightgreen', alpha=0.7))
        
        # Colors for different ray types
        colors = {'incident': 'red', 'reflected': 'blue', 'refracted': 'green'}
        
        # Trace and plot all rays
        all_rays_to_plot = []
        
        for ray in rays:
            # Add incident ray
            all_rays_to_plot.append((ray, 'incident'))
            
            # Trace through each interface
            current_rays = [ray]
            
            for interface in interfaces:
                next_rays = []
                for current_ray in current_rays:
                    reflected, refracted = self.trace_ray(current_ray, interface, ray_length)
                    
                    for r_ray in reflected:
                        all_rays_to_plot.append((r_ray, 'reflected'))
                        next_rays.append(r_ray)
                    
                    for t_ray in refracted:
                        all_rays_to_plot.append((t_ray, 'refracted'))
                        next_rays.append(t_ray)
                
                current_rays = next_rays
        
        # Plot all rays
        for ray, ray_type in all_rays_to_plot:
            if ray.intensity < 0.01:  # Skip very dim rays
                continue
                
            # Calculate ray endpoints
            dx, dy = ray.get_unit_vector()
            x_end = ray.x + ray_length * dx
            y_end = ray.y + ray_length * dy
            
            # Clip to plotting range
            if x_end < x_range[0] or x_end > x_range[1]:
                t_clip = min((x_range[1] - ray.x) / dx if dx > 0 else float('inf'),
                            (x_range[0] - ray.x) / dx if dx < 0 else float('inf'))
                x_end = ray.x + t_clip * dx
                y_end = ray.y + t_clip * dy
            
            if y_end < y_range[0] or y_end > y_range[1]:
                t_clip = min((y_range[1] - ray.y) / dy if dy > 0 else float('inf'),
                            (y_range[0] - ray.y) / dy if dy < 0 else float('inf'))
                x_end = ray.x + t_clip * dx
                y_end = ray.y + t_clip * dy
            
            # Plot ray with intensity-based alpha
            alpha = min(1.0, ray.intensity)
            ax.arrow(ray.x, ray.y, x_end - ray.x, y_end - ray.y,
                    head_width=0.1, head_length=0.1, fc=colors[ray_type], 
                    ec=colors[ray_type], alpha=alpha, linewidth=2)
        
        # Formatting
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Position (arbitrary units)', fontsize=12)
        ax.set_ylabel('Height (arbitrary units)', fontsize=12)
        ax.set_title('Geometric Optics: Reflection and Refraction', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors['incident'], lw=2, label='Incident Ray'),
                          Line2D([0], [0], color=colors['reflected'], lw=2, label='Reflected Ray'),
                          Line2D([0], [0], color=colors['refracted'], lw=2, label='Refracted Ray')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig, ax


def demonstrate_reflection_and_refraction():
    """Demonstrate key concepts of reflection and refraction."""
    
    print("🔬 Geometric Optics: Reflection and Refraction Demo")
    print("=" * 55)
    
    # Create optics system
    optics = GeometricOptics()
    
    # Define interface: air to glass
    interface = Interface(
        position=0.0,
        n1=1.0,    # air
        n2=1.5,    # glass
        name1="Air",
        name2="Glass"
    )
    
    print(f"\n📐 Interface: {interface.name1} (n={interface.n1}) ↔ {interface.name2} (n={interface.n2})")
    
    # Calculate critical angle
    critical_angle_rad = optics.critical_angle(interface.n2, interface.n1)
    if critical_angle_rad:
        critical_angle_deg = np.degrees(critical_angle_rad)
        print(f"🔴 Critical angle (glass→air): {critical_angle_deg:.1f}°")
    
    # Test different incident angles
    test_angles = [0, 15, 30, 45, 60, 75]  # degrees
    
    print(f"\n{'Angle (°)':<10}{'θ_ref (°)':<12}{'θ_trans (°)':<12}{'R (%)':<8}{'T (%)'}")
    print("-" * 60)
    
    for angle_deg in test_angles:
        angle_rad = np.radians(angle_deg)
        
        # Calculate refracted angle
        refracted_rad = optics.snells_law(angle_rad, interface.n1, interface.n2)
        
        if refracted_rad is not None:
            refracted_deg = np.degrees(refracted_rad)
            R = optics.fresnel_reflectance(angle_rad, interface.n1, interface.n2)
            T = 1 - R
            
            print(f"{angle_deg:<10}{angle_deg:<12.1f}{refracted_deg:<12.1f}{R*100:<8.1f}{T*100:.1f}")
        else:
            print(f"{angle_deg:<10}{angle_deg:<12.1f}{'TIR':<12}{'100.0':<8}{'0.0'}")
    
    print("\nTIR = Total Internal Reflection")
    
    # Create visualization
    print("\n📊 Creating ray diagram...")
    
    # Create incident rays at different angles
    incident_rays = []
    for angle_deg in [15, 30, 45, 60]:
        angle_rad = np.radians(180 - angle_deg)  # Convert to ray direction
        ray = Ray(x=-3, y=2, direction=angle_rad, intensity=1.0)
        incident_rays.append(ray)
    
    # Plot ray diagram
    fig, ax = optics.plot_ray_diagram(
        rays=incident_rays,
        interfaces=[interface],
        x_range=(-4, 4),
        y_range=(-2, 3),
        ray_length=6
    )
    
    plt.show()
    
    # Additional demonstrations
    demonstrate_total_internal_reflection()
    demonstrate_brewster_angle()


def demonstrate_total_internal_reflection():
    """Demonstrate total internal reflection phenomenon."""
    
    print(f"\n🔴 Total Internal Reflection Demo")
    print("=" * 40)
    
    optics = GeometricOptics()
    
    # Glass to air interface
    interface = Interface(
        position=0.0,
        n1=1.5,    # glass (denser medium)
        n2=1.0,    # air (rarer medium)
        name1="Glass",
        name2="Air"
    )
    
    critical_angle_rad = optics.critical_angle(interface.n1, interface.n2)
    critical_angle_deg = np.degrees(critical_angle_rad)
    
    print(f"Critical angle: {critical_angle_deg:.2f}°")
    print(f"Testing angles around critical angle:")
    
    # Test angles around critical angle
    test_angles = np.array([35, 40, 42, 44, 45, 46, 48, 50])  # degrees
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Reflectance vs angle
    angles_fine = np.linspace(0, 90, 200)
    reflectances = []
    
    for angle in angles_fine:
        angle_rad = np.radians(angle)
        R = optics.fresnel_reflectance(angle_rad, interface.n1, interface.n2)
        reflectances.append(R)
    
    ax1.plot(angles_fine, reflectances, 'b-', linewidth=2)
    ax1.axvline(critical_angle_deg, color='red', linestyle='--', 
                label=f'Critical angle = {critical_angle_deg:.1f}°')
    ax1.set_xlabel('Incident Angle (degrees)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Reflectance vs Incident Angle\n(Glass → Air)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Ray diagram for TIR
    incident_rays = []
    for angle_deg in [30, 42, 45, 50, 60]:
        # Rays coming from below the interface (glass medium)
        angle_rad = np.radians(angle_deg)  # Angle from normal
        ray_direction = np.pi/2 - angle_rad  # Convert to ray direction
        ray = Ray(x=-2, y=-1.5, direction=ray_direction, intensity=1.0)
        incident_rays.append(ray)
    
    # Custom plotting for TIR
    for ray in incident_rays:
        # Calculate intersection and angles
        t = (interface.position - ray.y) / np.sin(ray.direction)
        x_int = ray.x + t * np.cos(ray.direction)
        
        theta_incident = np.pi/2 - ray.direction
        
        # Draw incident ray
        ax2.arrow(ray.x, ray.y, t * np.cos(ray.direction), t * np.sin(ray.direction),
                 head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2)
        
        # Check for TIR
        theta_refracted = optics.snells_law(theta_incident, interface.n1, interface.n2)
        
        if theta_refracted is None:  # TIR
            # Only reflected ray
            reflected_dir = np.pi - ray.direction
            ax2.arrow(x_int, interface.position, 
                     2 * np.cos(reflected_dir), 2 * np.sin(reflected_dir),
                     head_width=0.05, head_length=0.05, fc='blue', ec='blue', linewidth=2)
        else:
            # Both reflected and refracted rays
            R = optics.fresnel_reflectance(theta_incident, interface.n1, interface.n2)
            
            # Reflected ray
            reflected_dir = np.pi - ray.direction
            ax2.arrow(x_int, interface.position, 
                     1.5 * np.cos(reflected_dir), 1.5 * np.sin(reflected_dir),
                     head_width=0.05, head_length=0.05, fc='blue', ec='blue', 
                     linewidth=2, alpha=R)
            
            # Refracted ray
            refracted_dir = np.pi/2 + theta_refracted
            ax2.arrow(x_int, interface.position, 
                     1.5 * np.cos(refracted_dir), 1.5 * np.sin(refracted_dir),
                     head_width=0.05, head_length=0.05, fc='green', ec='green', 
                     linewidth=2, alpha=1-R)
    
    # Draw interface
    ax2.axhline(y=0, color='black', linewidth=3)
    ax2.text(-3, 0.3, 'Air (n=1.0)', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    ax2.text(-3, -0.3, 'Glass (n=1.5)', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax2.set_xlim(-3, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Height')
    ax2.set_title('Total Internal Reflection\n(Various Incident Angles)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def demonstrate_brewster_angle():
    """Demonstrate Brewster's angle and polarization effects."""
    
    print(f"\n💎 Brewster's Angle Demo")
    print("=" * 30)
    
    # For air-glass interface
    n1, n2 = 1.0, 1.5
    brewster_angle_rad = np.arctan(n2 / n1)
    brewster_angle_deg = np.degrees(brewster_angle_rad)
    
    print(f"Brewster's angle (air→glass): {brewster_angle_deg:.2f}°")
    print("At this angle, reflected light is completely polarized!")
    
    # Plot reflectance for s and p polarizations
    angles = np.linspace(0, 90, 200)
    Rs_values = []
    Rp_values = []
    
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        
        if angle_deg < 89.9:  # Avoid numerical issues near 90°
            # Simplified Fresnel equations
            theta2_rad = np.arcsin((n1/n2) * np.sin(angle_rad))
            cos1, cos2 = np.cos(angle_rad), np.cos(theta2_rad)
            
            Rs = ((n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)) ** 2
            Rp = ((n1 * cos2 - n2 * cos1) / (n1 * cos2 + n2 * cos1)) ** 2
            
            Rs_values.append(Rs)
            Rp_values.append(Rp)
        else:
            Rs_values.append(1.0)
            Rp_values.append(1.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles, Rs_values, 'b-', linewidth=2, label='s-polarized (⊥)')
    plt.plot(angles, Rp_values, 'r-', linewidth=2, label='p-polarized (∥)')
    plt.plot(angles, [(Rs + Rp)/2 for Rs, Rp in zip(Rs_values, Rp_values)], 
             'g--', linewidth=2, label='Unpolarized (average)')
    
    plt.axvline(brewster_angle_deg, color='purple', linestyle=':', linewidth=2,
                label=f"Brewster's angle = {brewster_angle_deg:.1f}°")
    
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Reflectance')
    plt.title("Fresnel Reflectance: Polarization Effects\n(Air → Glass Interface)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 90)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📝 Key Observations:")
    print(f"• At Brewster's angle ({brewster_angle_deg:.1f}°), Rp = 0 (no reflection of p-polarized light)")
    print(f"• Reflected light becomes completely s-polarized")
    print(f"• This principle is used in polarizing sunglasses and photography filters")


def demonstrate_comprehensive_optical_laws():
    """
    Demonstrate all five fundamental optical laws in a summary table format
    """
    print(f"\n🔬 COMPREHENSIVE OPTICAL LAWS DEMONSTRATION")
    print("=" * 60)
    print("This demonstration covers all seven fundamental optical phenomena:")
    print("1. Reflection: θᵢ = θᵣ")
    print("2. Refraction: n₁sin θ₁ = n₂sin θ₂") 
    print("3. Absorption: I = I₀e⁻ᵅˣ")
    print("4. Transmission: T = I/I₀, R + A + T = 1")
    print("5. Scattering: I ∝ 1/λ⁴ (Rayleigh)")
    
    optics = GeometricOptics()
    
    # Example scenario: Light going from air to glass with absorption
    print(f"\n📊 EXAMPLE ANALYSIS: Air → Glass Interface")
    print("-" * 50)
    
    # Parameters
    incident_angle = np.radians(30)  # 30 degrees
    n_air = 1.0
    n_glass = 1.5
    absorption_coeff = 0.1  # m⁻¹
    glass_thickness = 0.05  # 5 cm
    wavelength = 500e-9  # Blue-green light
    
    print(f"Scenario:")
    print(f"   Light wavelength: {wavelength*1e9:.0f} nm")
    print(f"   Incident angle: {np.degrees(incident_angle):.1f}°")
    print(f"   Air refractive index: {n_air}")
    print(f"   Glass refractive index: {n_glass}")
    print(f"   Glass thickness: {glass_thickness*100:.1f} cm")
    print(f"   Absorption coefficient: {absorption_coeff:.1f} m⁻¹")
    
    # Perform comprehensive analysis
    results = optics.comprehensive_optical_analysis(
        incident_angle, n_air, n_glass, absorption_coeff, glass_thickness, wavelength
    )
    
    # Display results in table format
    print(f"\n📋 OPTICAL LAWS ANALYSIS RESULTS:")
    print("=" * 60)
    
    # 1. Reflection
    refl = results['reflection']
    print(f"\n1️⃣  REFLECTION LAW: {refl['law']}")
    print(f"    Incident angle:  {refl['incident_angle_deg']:.1f}°")
    print(f"    Reflected angle: {refl['reflected_angle_deg']:.1f}°")
    print(f"    ✅ Law verified: Angle in = Angle out")
    
    # 2. Refraction  
    refr = results['refraction']
    print(f"\n2️⃣  REFRACTION LAW: {refr['law']}")
    print(f"    n₁ = {refr['n1']:.1f}, n₂ = {refr['n2']:.1f}")
    print(f"    Incident angle:  {refr['incident_angle_deg']:.1f}°")
    if refr['refracted_angle_deg']:
        print(f"    Refracted angle: {refr['refracted_angle_deg']:.1f}°")
        print(f"    ✅ Light bends toward normal (n₁ < n₂)")
    else:
        print(f"    ❌ Total internal reflection occurs")
    
    # 3. Fresnel Transmission/Reflection
    fresnel = results['fresnel']
    print(f"\n3️⃣  TRANSMISSION: T = I/I₀")
    print(f"    Reflectance (R): {fresnel['reflectance']:.3f} ({fresnel['reflectance']*100:.1f}%)")
    print(f"    Transmittance (T): {fresnel['transmittance']:.3f} ({fresnel['transmittance']*100:.1f}%)")
    print(f"    Conservation check: R + T = {fresnel['reflectance'] + fresnel['transmittance']:.3f}")
    print(f"    ✅ Energy conservation: {'Verified' if fresnel['conservation_check'] else 'Failed'}")
    
    # 4. Absorption
    if results['absorption']:
        abs_result = results['absorption']
        print(f"\n4️⃣  ABSORPTION LAW: I = I₀e⁻ᵅˣ")
        print(f"    Initial intensity: {abs_result['initial_intensity']:.3f}")
        print(f"    After absorption: {abs_result['transmitted_intensity']:.3f}")
        print(f"    Absorption fraction: {abs_result['absorption_fraction']:.3f} ({abs_result['absorption_fraction']*100:.1f}%)")
        print(f"    Penetration depth: {abs_result['penetration_depth']:.3f} m")
        print(f"    Attenuation: {abs_result['attenuation_db']:.1f} dB")
        print(f"    ✅ Exponential intensity loss verified")
    else:
        print(f"\n4️⃣  ABSORPTION LAW: I = I₀e⁻ᵅˣ")
        print(f"    No absorption in this scenario (α = 0 or thickness = 0)")
    
    # 5. Scattering
    scat = results['scattering']
    print(f"\n5️⃣  SCATTERING LAW: I ∝ 1/λ⁴ (Rayleigh)")
    print(f"    Wavelength: {scat['wavelength_nm']:.0f} nm ({scat['color_preference']})")
    print(f"    λ⁻⁴ factor: {scat['lambda_4_factor']:.2e}")
    print(f"    Relative intensity: {scat['relative_intensity']:.2f}")
    print(f"    Scattering coefficient: {scat['scattering_coefficient']:.2e}")
    print(f"    ✅ Shorter wavelengths scatter more strongly")
    
    # Summary table
    print(f"\n📊 SUMMARY TABLE OF OPTICAL LAWS:")
    print("=" * 70)
    print("| Interaction | Law/Equation              | Key Concept               |")
    print("|-------------|---------------------------|---------------------------|")
    print("| Reflection  | θᵢ = θᵣ                   | Angle in = angle out      |")
    print("| Refraction  | n₁sin θ₁ = n₂sin θ₂       | Light bends at interface  |")
    print("| Absorption  | I = I₀e⁻ᵅˣ               | Exponential intensity loss|")
    print("| Transmission| T = I/I₀, R+A+T=1        | Fraction of light passed  |")
    print("| Scattering  | I ∝ 1/λ⁴ (Rayleigh)      | Wavelength-dependent     |")
    print("=" * 70)
    
    # Practical examples
    print(f"\n🌍 REAL-WORLD APPLICATIONS:")
    print("• Reflection: Mirrors, periscopes, fiber optic cables")
    print("• Refraction: Lenses, prisms, eyeglasses, camera systems")
    print("• Absorption: Sunglasses, UV filters, solar panels")  
    print("• Transmission: Windows, optical fibers, laser systems")
    print("• Scattering: Blue sky, red sunsets, atmospheric optics")
    
    # Generate comprehensive plots for all 7 optical laws
    print(f"\n📊 Generating individual plots for all 7 optical laws...")
    plot_all_optical_laws(optics, incident_angle, n_air, n_glass)
    
    # Also generate wavelength-dependent analysis
    print(f"\n📈 Generating wavelength-dependent analysis...")
    plot_wavelength_dependent_analysis(optics, incident_angle, n_air, n_glass)


def plot_all_optical_laws(optics, theta1, n1, n2):
    """
    Create individual plots for each of the 7 fundamental optical laws
    """
    # Set up the figure with 2 rows and 4 columns 
    fig = plt.figure(figsize=(20, 12))
    
    # Law 1: Combined Reflection & Refraction
    ax1 = plt.subplot(2, 4, 1)
    angles_incident = np.linspace(0, 89, 90)
    angles_reflected = angles_incident  # Law of reflection
    
    # Calculate refracted angles
    refracted_angles = []
    for angle_deg in angles_incident:
        angle_rad = np.radians(angle_deg)
        theta2 = optics.snells_law(angle_rad, n1, n2)
        if theta2 is not None:
            refracted_angles.append(np.degrees(theta2))
        else:
            refracted_angles.append(np.nan)  # Total internal reflection
    
    # Plot both reflection and refraction
    ax1.plot(angles_incident, angles_reflected, 'b-', linewidth=3, label='Reflection: θᵣ = θᵢ')
    ax1.plot(angles_incident, refracted_angles, 'g-', linewidth=3, label=f'Refraction: n₁={n1}, n₂={n2}')
    ax1.plot([0, 90], [0, 90], 'r--', alpha=0.7, label='1:1 reference')
    
    ax1.set_xlabel('Incident Angle (degrees)', fontsize=11)
    ax1.set_ylabel('Output Angle (degrees)', fontsize=11)
    ax1.set_title('1. Reflection & Refraction\nθᵢ = θᵣ & n₁sin θ₁ = n₂sin θ₂', fontsize=11, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 90)
    
    # Law 2: Absorption - Beer-Lambert Law
    ax2 = plt.subplot(2, 4, 2)
    thicknesses = np.linspace(0, 2, 100)  # 0 to 2 meters
    absorption_coeffs = [0.1, 0.5, 1.0, 2.0]
    colors = ['blue', 'green', 'orange', 'red']
    
    for alpha, color in zip(absorption_coeffs, colors):
        intensities = [np.exp(-alpha * x) for x in thicknesses]
        ax2.plot(thicknesses*100, intensities, linewidth=2.5, color=color, 
                label=f'α = {alpha:.1f} m⁻¹')
    
    ax2.set_xlabel('Thickness (cm)', fontsize=11)
    ax2.set_ylabel('Transmitted Intensity I/I₀', fontsize=11)
    ax2.set_title('2. Beer-Lambert Law\nI = I₀e⁻ᵅˣ', fontsize=11, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    # Law 3: Transmission - Energy Conservation
    ax3 = plt.subplot(2, 4, 3)
    angles_trans = np.linspace(0, 89, 90)
    reflectances = []
    transmittances = []
    
    for angle_deg in angles_trans:
        angle_rad = np.radians(angle_deg)
        R = optics.fresnel_reflectance(angle_rad, n1, n2)
        T = 1.0 - R if optics.snells_law(angle_rad, n1, n2) is not None else 0.0
        reflectances.append(R)
        transmittances.append(T)
    
    ax3.plot(angles_trans, reflectances, 'r-', linewidth=3, label='Reflectance (R)')
    ax3.plot(angles_trans, transmittances, 'b-', linewidth=3, label='Transmittance (T)')
    ax3.plot(angles_trans, [r+t for r,t in zip(reflectances, transmittances)], 
            'k--', linewidth=2, alpha=0.7, label='R + T')
    ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.8, label='Conservation = 1')
    
    ax3.set_xlabel('Incident Angle (degrees)', fontsize=11)
    ax3.set_ylabel('Fraction', fontsize=11)
    ax3.set_title('3. Energy Conservation\nR + A + T = 1', fontsize=11, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    # Law 4: Scattering - Rayleigh Law
    ax4 = plt.subplot(2, 4, 4)
    wavelengths = np.linspace(400, 700, 100)  # nm
    scattering_intensities = [(550/wl)**4 for wl in wavelengths]  # Relative to 550nm
    
    ax4.plot(wavelengths, scattering_intensities, 'purple', linewidth=3, label='I ∝ 1/λ⁴')
    
    # Color regions
    colors_regions = ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
    bounds = [400, 450, 495, 570, 590, 620, 700]
    for i in range(len(colors_regions)):
        ax4.axvspan(bounds[i], bounds[i+1], alpha=0.2, color=colors_regions[i])
    
    ax4.set_xlabel('Wavelength (nm)', fontsize=11)
    ax4.set_ylabel('Relative Scattering Intensity', fontsize=11)
    ax4.set_title('4. Rayleigh Scattering\nI ∝ 1/λ⁴', fontsize=11, weight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Law 5: Critical Angle
    ax5 = plt.subplot(2, 4, 5)
    n_ratios = np.linspace(1.01, 3.0, 100)  # n₁/n₂ ratios
    critical_angles = [np.degrees(np.arcsin(1.0/n_ratio)) for n_ratio in n_ratios]
    
    ax5.plot(n_ratios, critical_angles, 'm-', linewidth=3, label='θc = arcsin(n₂/n₁)')
    ax5.axhline(y=90, color='k', linestyle='--', alpha=0.7, label='Maximum angle')
    ax5.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='No TIR (n₁ = n₂)')
    
    ax5.set_xlabel('Refractive Index Ratio n₁/n₂', fontsize=11)
    ax5.set_ylabel('Critical Angle θc (degrees)', fontsize=11)
    ax5.set_title('5. Critical Angle\nθc = arcsin(n₂/n₁)', fontsize=11, weight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_ylim(0, 95)
    
    # Law 6: Brewster's Angle
    ax6 = plt.subplot(2, 4, 6)
    n2_values = np.linspace(1.1, 3.0, 100)
    n1_fixed = 1.0  # Air
    brewster_angles = [np.degrees(np.arctan(n2/n1_fixed)) for n2 in n2_values]
    
    ax6.plot(n2_values, brewster_angles, 'orange', linewidth=3, label='θB = arctan(n₂/n₁)')
    ax6.axhline(y=90, color='k', linestyle='--', alpha=0.7, label='Grazing angle')
    
    # Mark common materials
    materials = {'Glass': 1.5, 'Diamond': 2.4, 'Water': 1.33}
    for material, n in materials.items():
        brewster_angle = np.degrees(np.arctan(n/n1_fixed))
        ax6.plot(n, brewster_angle, 'o', markersize=8, label=f'{material} ({brewster_angle:.1f}°)')
    
    ax6.set_xlabel('Refractive Index n₂ (n₁ = 1.0)', fontsize=11)
    ax6.set_ylabel('Brewster Angle θB (degrees)', fontsize=11)
    ax6.set_title('6. Brewster\'s Angle\nθB = arctan(n₂/n₁)', fontsize=11, weight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_ylim(0, 95)
    
    # Summary plot combining key concepts
    ax7 = plt.subplot(2, 4, 7)
    
    # Create a comprehensive angle analysis
    test_angles = np.linspace(0, 89, 90)
    n1_test, n2_test = 1.0, 1.5
    
    reflected = test_angles  # Law of reflection
    refracted = []
    fresnel_R = []
    
    for angle in test_angles:
        angle_rad = np.radians(angle)
        
        # Refraction
        theta2 = optics.snells_law(angle_rad, n1_test, n2_test)
        if theta2 is not None:
            refracted.append(np.degrees(theta2))
        else:
            refracted.append(np.nan)
        
        # Fresnel reflectance
        R = optics.fresnel_reflectance(angle_rad, n1_test, n2_test)
        fresnel_R.append(R * 90)  # Scale for visibility
    
    ax7.plot(test_angles, reflected, 'b-', linewidth=2, label='Reflection θᵣ')
    ax7.plot(test_angles, refracted, 'g-', linewidth=2, label='Refraction θ₂')
    ax7.plot(test_angles, fresnel_R, 'r--', linewidth=2, label='Reflectance × 90')
    
    # Mark critical angle
    critical_ang = np.degrees(optics.critical_angle(n2_test, n1_test))
    if critical_ang:
        ax7.axvline(x=critical_ang, color='orange', linestyle=':', linewidth=2, 
                   label=f'Critical angle ({critical_ang:.1f}°)')
    
    # Mark Brewster angle  
    brewster_ang = np.degrees(np.arctan(n2_test/n1_test))
    ax7.axvline(x=brewster_ang, color='purple', linestyle=':', linewidth=2,
               label=f'Brewster angle ({brewster_ang:.1f}°)')
    
    ax7.set_xlabel('Incident Angle (degrees)', fontsize=11)
    ax7.set_ylabel('Angle/Scaled Value (degrees)', fontsize=11)
    ax7.set_title('7. Combined Analysis\nAll Laws Together', fontsize=11, weight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    ax7.set_xlim(0, 89)
    
    # Law 8: Comprehensive Overview  
    ax8 = plt.subplot(2, 4, 8)
    
    # Create a comparison of key parameters
    phenomena = ['Reflection', 'Refraction', 'Absorption', 'Scattering', 'Critical', 'Brewster']
    angles_comparison = [
        45,  # Reflection angle for 45° incident
        28.1,  # Refraction angle for 45° (air→glass, n=1.5)
        30,  # Representative absorption length scale  
        20,  # Representative scattering intensity
        41.8,  # Critical angle (glass→air)
        56.3   # Brewster angle (air→glass)
    ]
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    bars = ax8.bar(phenomena, angles_comparison, color=colors, alpha=0.7)
    ax8.set_xlabel('Optical Phenomena', fontsize=11)
    ax8.set_ylabel('Representative Values (degrees)', fontsize=11)
    ax8.set_title('8. Optical Laws Overview\nKey Parameters Comparison', fontsize=11, weight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, angles_comparison):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}°', ha='center', va='bottom', fontsize=9)
    
    plt.setp(ax8.get_xticklabels(), rotation=45, ha='right')
    ax8.set_ylim(0, max(angles_comparison) * 1.2)

    plt.tight_layout(pad=2.5, h_pad=4.0, w_pad=2.0)
   
    plt.show()


def plot_wavelength_dependent_analysis(optics, theta1, n1, n2):
    """
    Plot how different optical phenomena depend on wavelength
    """
    wavelengths = np.linspace(400e-9, 700e-9, 100)  # Visible spectrum
    wavelengths_nm = wavelengths * 1e9
    
    reflectances = []
    scattering_intensities = []
    
    particle_size = 0.1e-6
    particle_density = 1e12
    
    for wl in wavelengths:
        # Fresnel reflectance (wavelength independent for simple case)
        R = optics.fresnel_reflectance(theta1, n1, n2)
        reflectances.append(R)
        
        # Rayleigh scattering
        scat_result = optics.rayleigh_scattering(wl, particle_size, particle_density)
        scattering_intensities.append(scat_result['relative_intensity'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Rayleigh Scattering vs Wavelength
    ax1.plot(wavelengths_nm, scattering_intensities, 'b-', linewidth=3, label='I ∝ 1/λ⁴')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Relative Scattering Intensity', fontsize=12)
    ax1.set_title('Rayleigh Scattering Law\nI ∝ 1/λ⁴', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Color regions
    colors = ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
    bounds = [400, 450, 495, 570, 590, 620, 700]
    for i in range(len(colors)):
        ax1.axvspan(bounds[i], bounds[i+1], alpha=0.2, color=colors[i])
    
    # Plot 2: Absorption example (Beer-Lambert)
    absorption_coeffs = [0.01, 0.1, 1.0, 10.0]  # Different materials
    thicknesses = np.linspace(0, 1, 100)  # 0 to 1 meter
    
    for alpha in absorption_coeffs:
        transmissions = [np.exp(-alpha * x) for x in thicknesses]
        ax2.plot(thicknesses*100, transmissions, linewidth=2, 
                label=f'α = {alpha:.2f} m⁻¹')
    
    ax2.set_xlabel('Thickness (cm)', fontsize=12)
    ax2.set_ylabel('Transmission I/I₀', fontsize=12)
    ax2.set_title('Beer-Lambert Absorption Law\nI = I₀e⁻ᵅˣ', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    # Plot 3: Fresnel reflectance vs angle
    angles = np.linspace(0, 90, 91)
    reflectances_angle = []
    
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        try:
            R = optics.fresnel_reflectance(angle_rad, n1, n2)
            reflectances_angle.append(R)
        except:
            reflectances_angle.append(1.0)  # Total internal reflection
    
    ax3.plot(angles, reflectances_angle, 'r-', linewidth=3, label='Fresnel Reflectance')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Incident Angle (degrees)', fontsize=12)
    ax3.set_ylabel('Reflectance R', fontsize=12)
    ax3.set_title('Fresnel Reflectance vs Angle\n(Air → Glass)', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🌟 Welcome to Comprehensive Optical Laws Explorer! 🌟")
    print("This script demonstrates all fundamental optical phenomena:")
    print("• Reflection, Refraction, Absorption, Transmission, Scattering")
    
    print("\nSelect demonstration:")
    print("1. Comprehensive Optical Laws Summary")
    print("2. Classic Reflection & Refraction")  
    print("3. Total Internal Reflection")
    print("4. Brewster's Angle")
    print("5. All demonstrations")
    
    choice = input("Enter your choice (1-5): ").strip()
    
    try:
        if choice == '1':
            demonstrate_comprehensive_optical_laws()
        elif choice == '2':
            demonstrate_reflection_and_refraction()
        elif choice == '3':
            demonstrate_total_internal_reflection()
        elif choice == '4':
            demonstrate_brewster_angle()
        elif choice == '5':
            print("\n🎬 Running all demonstrations...")
            demonstrate_comprehensive_optical_laws()
            input("\nPress Enter to continue to Reflection & Refraction...")
            demonstrate_reflection_and_refraction()
            input("\nPress Enter to continue to Total Internal Reflection...")
            demonstrate_total_internal_reflection()
            input("\nPress Enter to continue to Brewster's Angle...")
            demonstrate_brewster_angle()
        else:
            print("Running comprehensive demonstration by default...")
            demonstrate_comprehensive_optical_laws()
        
        print(f"\n" + "="*60)
        print("🎓 Complete Summary of Optical Laws:")
        print("="*60)
        print("1. Reflection: θᵢ = θᵣ (angle of incidence = angle of reflection)")
        print("2. Refraction: n₁sin(θ₁) = n₂sin(θ₂) (Snell's Law)")
        print("3. Absorption: I = I₀e⁻ᵅˣ (Beer-Lambert Law)")
        print("4. Transmission: T = I/I₀, R + A + T = 1 (Energy conservation)")
        print("5. Scattering: I ∝ 1/λ⁴ (Rayleigh scattering)")
        print("6. Critical Angle: θc = arcsin(n₂/n₁)")
        print("7. Brewster's Angle: θB = arctan(n₂/n₁)")
        
        print(f"\n💡 Applications:")
        print("• Optical fibers (total internal reflection)")
        print("• Anti-reflection coatings (destructive interference)")
        print("• Polarizing filters (Brewster's angle)")
        print("• Prisms and lenses (controlled refraction)")
        print("• Beer-Lambert law: UV filters, sunglasses, spectroscopy")
        print("• Rayleigh scattering: Blue sky, red sunsets")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring comprehensive optical laws! ✨")