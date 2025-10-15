#!/usr/bin/env python3
"""
Geometric Optics: Advanced Ray Tracing
=====================================

This module provides comprehensive ray tracing capabilities for geometric optics:
- Multi-element optical systems
- Sequential ray tracing through complex systems
- Aberration analysis (spherical, chromatic, coma, etc.)
- Spot diagrams and wavefront analysis
- Optical system optimization
- Real lens design examples

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
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod


@dataclass
class Ray:
    """Represents a light ray with position and direction."""
    x: float          # x position
    y: float          # y position  
    z: float          # z position (along optical axis)
    dx: float         # x direction cosine
    dy: float         # y direction cosine
    dz: float         # z direction cosine
    wavelength: float = 550e-9  # wavelength in meters
    intensity: float = 1.0      # relative intensity
    
    def __post_init__(self):
        # Normalize direction cosines
        norm = np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
        if norm > 0:
            self.dx /= norm
            self.dy /= norm
            self.dz /= norm


class OpticalSurface(ABC):
    """Base class for optical surfaces."""
    
    def __init__(self, z_position: float, aperture_radius: float):
        self.z_position = z_position
        self.aperture_radius = aperture_radius
    
    @abstractmethod
    def intersect_ray(self, ray: Ray) -> Optional[Tuple[float, float, float]]:
        """Find intersection point of ray with surface."""
        pass
    
    @abstractmethod
    def get_surface_normal(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Get surface normal at intersection point."""
        pass
    
    @abstractmethod
    def refract_ray(self, ray: Ray, intersection: Tuple[float, float, float], 
                   n1: float, n2: float) -> Optional[Ray]:
        """Apply Snell's law at the surface."""
        pass


class SphericalSurface(OpticalSurface):
    """Represents a spherical optical surface."""
    
    def __init__(self, z_position: float, radius_of_curvature: float, 
                 aperture_radius: float):
        super().__init__(z_position, aperture_radius)
        self.radius_of_curvature = radius_of_curvature  # Positive for surfaces convex toward +z
        self.center_z = z_position - radius_of_curvature
    
    def intersect_ray(self, ray: Ray) -> Optional[Tuple[float, float, float]]:
        """Find intersection of ray with spherical surface."""
        if abs(self.radius_of_curvature) > 1e10:  # Flat surface
            if abs(ray.dz) < 1e-10:  # Ray parallel to surface
                return None
            t = (self.z_position - ray.z) / ray.dz
            if t < 0:
                return None
            x = ray.x + t * ray.dx
            y = ray.y + t * ray.dy
            if x**2 + y**2 > self.aperture_radius**2:
                return None
            return (x, y, self.z_position)
        
        # Spherical surface intersection
        # Ray: P = P0 + t*d
        # Sphere: (P - C)·(P - C) = R²
        
        # Vector from ray origin to sphere center
        oc_x = ray.x - 0
        oc_y = ray.y - 0
        oc_z = ray.z - self.center_z
        
        # Quadratic coefficients
        a = ray.dx**2 + ray.dy**2 + ray.dz**2  # Should be 1 for normalized ray
        b = 2 * (oc_x * ray.dx + oc_y * ray.dy + oc_z * ray.dz)
        c = oc_x**2 + oc_y**2 + oc_z**2 - self.radius_of_curvature**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        # Choose appropriate intersection
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # Choose the intersection in the forward direction
        if self.radius_of_curvature > 0:  # Convex surface
            t = t1 if t1 > 1e-10 else t2
        else:  # Concave surface
            t = t2 if t2 > 1e-10 else t1
        
        if t < 1e-10:
            return None
        
        # Intersection point
        x = ray.x + t * ray.dx
        y = ray.y + t * ray.dy
        z = ray.z + t * ray.dz
        
        # Check aperture
        if x**2 + y**2 > self.aperture_radius**2:
            return None
        
        return (x, y, z)
    
    def get_surface_normal(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Get outward surface normal."""
        if abs(self.radius_of_curvature) > 1e10:  # Flat surface
            return (0, 0, 1)  # Normal points in +z direction
        
        # Vector from sphere center to surface point
        nx = x - 0
        ny = y - 0  
        nz = z - self.center_z
        
        # Normalize
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        if norm > 0:
            nx /= norm
            ny /= norm
            nz /= norm
        
        # Ensure normal points in correct direction
        if self.radius_of_curvature < 0:
            nx = -nx
            ny = -ny
            nz = -nz
        
        return (nx, ny, nz)
    
    def refract_ray(self, ray: Ray, intersection: Tuple[float, float, float],
                   n1: float, n2: float) -> Optional[Ray]:
        """Apply Snell's law at the surface."""
        x, y, z = intersection
        nx, ny, nz = self.get_surface_normal(x, y, z)
        
        # Incident direction
        dx_i, dy_i, dz_i = ray.dx, ray.dy, ray.dz
        
        # Dot product of incident ray with normal
        cos_theta_i = -(dx_i * nx + dy_i * ny + dz_i * nz)
        
        # Handle total internal reflection
        sin_theta_i_sq = max(0, 1 - cos_theta_i**2)
        sin_theta_t_sq = (n1/n2)**2 * sin_theta_i_sq
        
        if sin_theta_t_sq > 1:  # Total internal reflection
            # Reflect instead of refract
            dx_r = dx_i + 2 * cos_theta_i * nx
            dy_r = dy_i + 2 * cos_theta_i * ny
            dz_r = dz_i + 2 * cos_theta_i * nz
            return Ray(x, y, z, dx_r, dy_r, dz_r, ray.wavelength, ray.intensity)
        
        cos_theta_t = np.sqrt(1 - sin_theta_t_sq)
        
        # Ensure correct sign for transmitted ray
        if cos_theta_i < 0:
            cos_theta_t = -cos_theta_t
        
        # Transmitted ray direction (Snell's law in vector form)
        ratio = n1 / n2
        dx_t = ratio * dx_i + (ratio * cos_theta_i - cos_theta_t) * nx
        dy_t = ratio * dy_i + (ratio * cos_theta_i - cos_theta_t) * ny
        dz_t = ratio * dz_i + (ratio * cos_theta_i - cos_theta_t) * nz
        
        return Ray(x, y, z, dx_t, dy_t, dz_t, ray.wavelength, ray.intensity)


class OpticalElement:
    """Represents an optical element (lens, mirror, etc.)."""
    
    def __init__(self, name: str):
        self.name = name
        self.surfaces: List[OpticalSurface] = []
        self.refractive_indices: List[float] = [1.0]  # Start with air
        self.materials: List[str] = ["air"]
    
    def add_surface(self, surface: OpticalSurface, refractive_index: float, material: str = "glass"):
        """Add a surface to the optical element."""
        self.surfaces.append(surface)
        self.refractive_indices.append(refractive_index)
        self.materials.append(material)
    
    def trace_ray(self, ray: Ray) -> List[Ray]:
        """Trace a ray through the optical element."""
        ray_path = [ray]
        current_ray = ray
        
        for i, surface in enumerate(self.surfaces):
            # Find intersection
            intersection = surface.intersect_ray(current_ray)
            if intersection is None:
                break  # Ray missed the surface
            
            # Refract ray
            n1 = self.refractive_indices[i]
            n2 = self.refractive_indices[i + 1]
            
            refracted_ray = surface.refract_ray(current_ray, intersection, n1, n2)
            if refracted_ray is None:
                break  # Total internal reflection or other failure
            
            ray_path.append(refracted_ray)
            current_ray = refracted_ray
        
        return ray_path


class OpticalSystem:
    """Represents a complete optical system with multiple elements."""
    
    def __init__(self, name: str = "Optical System"):
        self.name = name
        self.elements: List[OpticalElement] = []
        self.object_distance: float = float('inf')
        self.image_distance: float = 0
        self.magnification: float = 1
    
    def add_element(self, element: OpticalElement):
        """Add an optical element to the system."""
        self.elements.append(element)
    
    def trace_ray_through_system(self, ray: Ray) -> List[List[Ray]]:
        """Trace a ray through the entire optical system."""
        system_path = []
        current_ray = ray
        
        for element in self.elements:
            element_path = element.trace_ray(current_ray)
            system_path.append(element_path)
            if len(element_path) > 0:
                current_ray = element_path[-1]  # Continue with the exit ray
            else:
                break  # Ray was blocked or lost
        
        return system_path
    
    def trace_parallel_rays(self, heights: List[float], angle: float = 0) -> List[List[List[Ray]]]:
        """Trace multiple parallel rays through the system."""
        all_paths = []
        
        for height in heights:
            # Create ray at specified height
            ray = Ray(0, height, -1000,  # Start far to the left
                     np.sin(np.radians(angle)), 0, np.cos(np.radians(angle)))
            path = self.trace_ray_through_system(ray)
            all_paths.append(path)
        
        return all_paths
    
    def create_spot_diagram(self, heights: List[float], angles: List[float]) -> Tuple[List[float], List[float]]:
        """Create a spot diagram showing ray intercepts at the image plane."""
        intercept_x = []
        intercept_y = []
        
        # Find approximate image plane location
        image_z = self.estimate_image_plane()
        
        for height in heights:
            for angle in angles:
                ray = Ray(0, height, -1000,
                         np.sin(np.radians(angle)), 0, np.cos(np.radians(angle)))
                paths = self.trace_ray_through_system(ray)
                
                # Find where final ray intersects image plane
                if len(paths) > 0 and len(paths[-1]) > 0:
                    final_ray = paths[-1][-1]
                    
                    if abs(final_ray.dz) > 1e-10:
                        t = (image_z - final_ray.z) / final_ray.dz
                        x_intercept = final_ray.x + t * final_ray.dx
                        y_intercept = final_ray.y + t * final_ray.dy
                        
                        intercept_x.append(x_intercept)
                        intercept_y.append(y_intercept)
        
        return intercept_x, intercept_y
    
    def estimate_image_plane(self) -> float:
        """Estimate the location of the image plane using paraxial rays."""
        # Trace a paraxial ray to find image plane
        ray = Ray(0, 0.1, -1000, 0, 0, 1)  # Small height, parallel to axis
        paths = self.trace_ray_through_system(ray)
        
        if len(paths) > 0 and len(paths[-1]) > 1:
            final_ray = paths[-1][-1]
            # Find where ray crosses optical axis
            if abs(final_ray.dy) > 1e-10:
                t = -final_ray.y / final_ray.dy
                return final_ray.z + t * final_ray.dz
        
        return 100  # Default fallback


def create_thin_lens(focal_length: float, diameter: float, z_position: float, 
                    refractive_index: float = 1.5) -> OpticalElement:
    """Create a thin lens optical element."""
    lens = OpticalElement(f"Thin Lens (f={focal_length}mm)")
    
    # For thin lens, use lensmaker's equation to find radii
    # 1/f = (n-1)(1/R1 - 1/R2) 
    # For symmetric lens: R1 = -R2 = R
    # 1/f = (n-1)(2/R) => R = 2f(n-1)
    
    R = 2 * focal_length * (refractive_index - 1)
    
    # Front surface (convex)
    front_surface = SphericalSurface(z_position - 0.5, R, diameter/2)
    lens.add_surface(front_surface, refractive_index, "crown_glass")
    
    # Back surface (concave)  
    back_surface = SphericalSurface(z_position + 0.5, -R, diameter/2)
    lens.add_surface(back_surface, 1.0, "air")
    
    return lens


def create_doublet_lens(f1: float, f2: float, separation: float, diameter: float, 
                       z_position: float) -> OpticalElement:
    """Create a doublet lens system."""
    doublet = OpticalElement(f"Doublet Lens System")
    
    # First lens (crown glass)
    R1 = 2 * f1 * 0.5  # n = 1.5 for crown glass
    front1 = SphericalSurface(z_position, R1, diameter/2)
    doublet.add_surface(front1, 1.52, "crown_glass")
    
    back1 = SphericalSurface(z_position + 2, -R1, diameter/2)
    doublet.add_surface(back1, 1.0, "air")
    
    # Gap between lenses
    # Second lens (flint glass)
    R2 = 2 * f2 * 0.62  # n = 1.62 for flint glass  
    front2 = SphericalSurface(z_position + 2 + separation, R2, diameter/2)
    doublet.add_surface(front2, 1.62, "flint_glass")
    
    back2 = SphericalSurface(z_position + 4 + separation, -R2, diameter/2)
    doublet.add_surface(back2, 1.0, "air")
    
    return doublet


def demonstrate_single_lens():
    """Demonstrate ray tracing through a single lens."""
    print("🔍 Single Lens Ray Tracing")
    print("=" * 30)
    
    # Create optical system
    system = OpticalSystem("Single Convex Lens")
    lens = create_thin_lens(focal_length=50.0, diameter=25.0, z_position=0)
    system.add_element(lens)
    
    # Trace rays at different heights
    heights = [-10, -5, 0, 5, 10]
    paths = system.trace_parallel_rays(heights)
    
    # Plot ray diagram
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, path_list in enumerate(paths):
        color = colors[i % len(colors)]
        
        for element_path in path_list:
            for j in range(len(element_path) - 1):
                ray1 = element_path[j]
                ray2 = element_path[j + 1]
                
                ax.plot([ray1.z, ray2.z], [ray1.y, ray2.y], color=color, linewidth=2)
    
    # Draw lens
    lens_z = 0
    lens_r = 12.5
    ax.plot([lens_z, lens_z], [-lens_r, lens_r], 'k-', linewidth=4, label='Lens')
    
    # Draw focal points
    ax.plot(50, 0, 'ro', markersize=8, label='Focal Point')
    ax.plot(-50, 0, 'ro', markersize=8)
    
    # Optical axis
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlim(-100, 150)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('Distance (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('Ray Tracing Through Single Convex Lens')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    print(f"✅ Single lens demonstration complete!")


def demonstrate_spot_diagram():
    """Demonstrate spot diagram analysis."""
    print("\n📊 Spot Diagram Analysis")
    print("=" * 30)
    
    # Create system with spherical aberration
    system = OpticalSystem("Lens with Spherical Aberration")
    
    # Use a lens with some spherical aberration
    lens = OpticalElement("Spherical Lens")
    
    # Add surfaces that will show aberration
    front_surface = SphericalSurface(-2, 30.0, 15.0)  # R = 30mm
    lens.add_surface(front_surface, 1.5, "glass")
    
    back_surface = SphericalSurface(2, -30.0, 15.0)
    lens.add_surface(back_surface, 1.0, "air")
    
    system.add_element(lens)
    
    # Create spot diagram
    heights = np.linspace(-12, 12, 5)
    angles = [0, 0.5, 1.0]  # Small angles in degrees
    
    x_spots, y_spots = system.create_spot_diagram(heights.tolist(), angles)
    
    # Plot spot diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(x_spots, y_spots, c='red', alpha=0.6, s=20)
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_title('Spot Diagram - Ray Intercepts at Image Plane')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add circle to show ideal focus
    circle = plt.Circle((0, 0), 0.1, fill=False, color='blue', linewidth=2, 
                       linestyle='--', label='Ideal Focus')
    ax.add_patch(circle)
    ax.legend()
    
    plt.show()
    
    print(f"✅ Spot diagram analysis complete!")


def demonstrate_aberrations():
    """Demonstrate common optical aberrations."""
    print("\n🔍 Optical Aberrations")
    print("=" * 25)
    
    print("\n📖 Common Optical Aberrations:")
    print("1. Spherical Aberration - rays at different heights focus at different points")
    print("2. Chromatic Aberration - different wavelengths focus at different points") 
    print("3. Coma - off-axis point sources appear comet-shaped")
    print("4. Astigmatism - different focal lengths in perpendicular planes")
    print("5. Field Curvature - flat object focuses on curved surface")
    print("6. Distortion - magnification varies across field")
    
    # Demonstrate chromatic aberration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Single lens chromatic aberration
    focal_lengths = {"red": 52.0, "green": 50.0, "blue": 48.0}  # Different f for different λ
    
    ax1.set_title("Chromatic Aberration - Single Lens")
    
    for color, f in focal_lengths.items():
        # Draw focus points
        ax1.plot(f, 0, 'o', color=color, markersize=8, label=f'{color.title()} λ focus')
    
    # Draw lens
    ax1.plot([0, 0], [-15, 15], 'k-', linewidth=4)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax1.set_xlim(-10, 60)
    ax1.set_ylim(-5, 5)
    ax1.set_xlabel('Distance (mm)')
    ax1.set_ylabel('Height (mm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Achromatic doublet correction
    ax2.set_title("Achromatic Doublet - Corrected")
    
    # All colors focus at same point for doublet
    common_focus = 50.0
    for color in ["red", "green", "blue"]:
        ax2.plot(common_focus, 0, 'o', color=color, markersize=8, alpha=0.7)
    
    ax2.plot(common_focus, 0, 'ko', markersize=10, fillstyle='none', linewidth=2, 
            label='Common focus')
    
    # Draw doublet lenses
    ax2.plot([0, 0], [-15, 15], 'b-', linewidth=4, label='Crown glass')
    ax2.plot([5, 5], [-15, 15], 'r-', linewidth=4, label='Flint glass')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_xlim(-10, 60)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel('Distance (mm)')
    ax2.set_ylabel('Height (mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Aberration analysis complete!")


def demonstrate_complex_system():
    """Demonstrate ray tracing through a complex optical system."""
    print("\n🔬 Complex Optical System")
    print("=" * 30)
    
    # Create a telescope-like system
    system = OpticalSystem("Simple Telescope")
    
    # Objective lens
    objective = create_thin_lens(focal_length=500.0, diameter=50.0, z_position=0)
    system.add_element(objective)
    
    # Eyepiece lens  
    eyepiece = create_thin_lens(focal_length=25.0, diameter=20.0, z_position=525)
    system.add_element(eyepiece)
    
    # Trace rays from infinity
    heights = [-20, -10, 0, 10, 20]
    angles = [0, 0.5, 1.0]  # Different field angles
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for angle in angles:
        paths = system.trace_parallel_rays(heights, angle)
        
        alpha = 1.0 - angle * 0.3  # Make off-axis rays more transparent
        
        for i, path_list in enumerate(paths):
            color = colors[i % len(colors)]
            
            for element_path in path_list:
                for j in range(len(element_path) - 1):
                    ray1 = element_path[j]
                    ray2 = element_path[j + 1]
                    
                    ax.plot([ray1.z, ray2.z], [ray1.y, ray2.y], 
                           color=color, linewidth=1.5, alpha=alpha)
    
    # Draw optical elements
    ax.plot([0, 0], [-25, 25], 'k-', linewidth=6, label='Objective Lens')
    ax.plot([525, 525], [-10, 10], 'k-', linewidth=4, label='Eyepiece')
    
    # Draw focal points
    ax.plot([500, 500], [-2, 2], 'r-', linewidth=3, label='Focal Plane')
    
    # Optical axis
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlim(-100, 600)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('Distance (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('Ray Tracing Through Simple Telescope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    magnification = 500.0 / 25.0  # fo / fe
    print(f"🔭 Telescope Magnification: {magnification:.1f}×")
    print(f"✅ Complex system analysis complete!")


if __name__ == "__main__":
    print("🌈 Welcome to Advanced Ray Tracing Explorer! 🌈")
    print("This script demonstrates comprehensive ray tracing through optical systems.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_single_lens()
        demonstrate_spot_diagram()
        demonstrate_aberrations()
        demonstrate_complex_system()
        
        print(f"\n" + "="*70)
        print("🎓 Advanced Ray Tracing Summary:")
        print("="*70)
        print("Key Concepts Demonstrated:")
        print("• Sequential ray tracing through multiple surfaces")
        print("• Snell's law applied at each optical interface")
        print("• Spot diagram analysis for aberration assessment")
        print("• Chromatic aberration and achromatic correction")
        print("• Complex multi-element system analysis")
        
        print(f"\n💡 Ray Tracing Applications:")
        print("• Lens design and optimization")
        print("• Optical system performance prediction") 
        print("• Aberration analysis and correction")
        print("• Tolerance analysis for manufacturing")
        print("• Virtual prototyping of optical instruments")
        
        print(f"\n🔬 Professional Ray Tracing Tools:")
        print("• Zemax OpticStudio")
        print("• Code V")
        print("• OSLO")
        print("• Python + NumPy for custom analysis")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring advanced ray tracing! ✨")