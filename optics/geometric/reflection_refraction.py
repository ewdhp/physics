#!/usr/bin/env python3
"""
Geometric Optics: Reflection and Refraction
==========================================

This module demonstrates the fundamental concepts of geometric optics:
- Law of reflection
- Snell's law of refraction
- Total internal reflection
- Critical angle
- Ray tracing through different media

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


if __name__ == "__main__":
    print("🌟 Welcome to Geometric Optics Explorer! 🌟")
    print("This script demonstrates fundamental concepts of reflection and refraction.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_reflection_and_refraction()
        
        print(f"\n" + "="*60)
        print("🎓 Summary of Key Concepts:")
        print("="*60)
        print("1. Law of Reflection: θᵢ = θᵣ (angle of incidence = angle of reflection)")
        print("2. Snell's Law: n₁sin(θ₁) = n₂sin(θ₂)")
        print("3. Total Internal Reflection: occurs when n₁ > n₂ and θᵢ > θc")
        print("4. Critical Angle: θc = arcsin(n₂/n₁)")
        print("5. Fresnel Equations: describe reflection/transmission coefficients")
        print("6. Brewster's Angle: θB = arctan(n₂/n₁), gives complete polarization")
        
        print(f"\n💡 Applications:")
        print("• Optical fibers (total internal reflection)")
        print("• Anti-reflection coatings (destructive interference)")
        print("• Polarizing filters (Brewster's angle)")
        print("• Prisms and lenses (controlled refraction)")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring geometric optics! ✨")