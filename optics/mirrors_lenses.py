#!/usr/bin/env python3
"""
Geometric Optics: Mirrors and Lenses
====================================

This module demonstrates the behavior of mirrors and lenses in geometric optics:
- Plane mirrors
- Spherical mirrors (concave and convex)
- Thin lenses (converging and diverging)
- Lens equation and mirror equation
- Image formation and magnification
- Ray diagrams for mirrors and lenses

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
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
import math


@dataclass
class OpticalElement:
    """Base class for optical elements (mirrors and lenses)."""
    x_position: float  # Position along optical axis
    focal_length: float  # Focal length (positive for converging, negative for diverging)
    aperture: float  # Height of the element
    name: str = "Optical Element"
    
    @property
    def is_converging(self) -> bool:
        """Check if the element is converging (positive focal length)."""
        return self.focal_length > 0
    
    @property
    def power(self) -> float:
        """Optical power in diopters (1/focal_length in meters)."""
        return 1.0 / self.focal_length if self.focal_length != 0 else float('inf')


@dataclass
class Mirror(OpticalElement):
    """Represents a mirror (plane or spherical)."""
    radius_of_curvature: float = float('inf')  # R = 2f for spherical mirrors
    
    def __post_init__(self):
        if self.radius_of_curvature != float('inf'):
            # For spherical mirrors: f = R/2
            self.focal_length = self.radius_of_curvature / 2
    
    @property
    def mirror_type(self) -> str:
        """Determine mirror type based on focal length."""
        if abs(self.focal_length) == float('inf'):
            return "plane"
        elif self.focal_length > 0:
            return "concave"
        else:
            return "convex"


@dataclass
class Lens(OpticalElement):
    """Represents a thin lens (converging or diverging)."""
    refractive_index: float = 1.5  # Typical glass
    
    @property
    def lens_type(self) -> str:
        """Determine lens type based on focal length."""
        if self.focal_length > 0:
            return "converging"
        else:
            return "diverging"


@dataclass
class LightRay:
    """Represents a light ray for ray tracing."""
    x: float  # Starting x position
    y: float  # Starting y position
    direction_x: float  # Direction cosine in x
    direction_y: float  # Direction cosine in y
    intensity: float = 1.0
    color: str = 'red'
    
    def propagate(self, distance: float) -> 'LightRay':
        """Propagate ray by a given distance."""
        new_x = self.x + distance * self.direction_x
        new_y = self.y + distance * self.direction_y
        return LightRay(new_x, new_y, self.direction_x, self.direction_y, 
                       self.intensity, self.color)


class GeometricOpticsSystem:
    """Main class for analyzing mirrors and lenses."""
    
    def __init__(self):
        self.elements: List[OpticalElement] = []
        self.rays: List[LightRay] = []
    
    def add_element(self, element: OpticalElement):
        """Add an optical element to the system."""
        self.elements.append(element)
    
    def lens_equation(self, focal_length: float, object_distance: float) -> Tuple[float, float]:
        """
        Apply thin lens equation: 1/f = 1/do + 1/di
        
        Args:
            focal_length: Focal length of lens
            object_distance: Distance from object to lens (positive)
            
        Returns:
            Tuple of (image_distance, magnification)
        """
        if object_distance == 0:
            return float('inf'), float('inf')
        
        # Lens equation: 1/f = 1/do + 1/di → di = (do * f) / (do - f)
        if abs(object_distance - focal_length) < 1e-10:
            return float('inf'), float('inf')  # Object at focal point
        
        image_distance = (object_distance * focal_length) / (object_distance - focal_length)
        magnification = -image_distance / object_distance
        
        return image_distance, magnification
    
    def mirror_equation(self, focal_length: float, object_distance: float) -> Tuple[float, float]:
        """
        Apply mirror equation: 1/f = 1/do + 1/di
        Same as lens equation but with different sign conventions.
        """
        return self.lens_equation(focal_length, object_distance)
    
    def trace_ray_through_lens(self, ray: LightRay, lens: Lens) -> List[LightRay]:
        """
        Trace a ray through a thin lens using ray tracing rules.
        
        Args:
            ray: Incident ray
            lens: Lens to trace through
            
        Returns:
            List containing the refracted ray
        """
        # Calculate intersection with lens
        if abs(ray.direction_x) < 1e-10:
            return []  # Ray parallel to optical axis
        
        t = (lens.x_position - ray.x) / ray.direction_x
        if t < 0:
            return []  # Ray moving away from lens
        
        # Intersection point
        intersection_y = ray.y + t * ray.direction_y
        
        # Apply thin lens ray tracing rules
        if abs(intersection_y) > lens.aperture / 2:
            return []  # Ray blocked by aperture
        
        # For thin lens, use ray tracing rules:
        # 1. Ray parallel to axis → passes through focal point
        # 2. Ray through optical center → continues straight
        # 3. Ray through focal point → emerges parallel to axis
        
        if abs(ray.direction_y) < 1e-6:  # Ray parallel to optical axis
            # Emerges toward focal point
            if lens.focal_length != 0:
                new_direction_y = -intersection_y / lens.focal_length
                new_direction_x = 1.0 / np.sqrt(1 + new_direction_y**2)
                new_direction_y = new_direction_y * new_direction_x
            else:
                new_direction_x, new_direction_y = ray.direction_x, ray.direction_y
        elif abs(intersection_y) < 1e-6:  # Ray through optical center
            # Continues straight
            new_direction_x, new_direction_y = ray.direction_x, ray.direction_y
        else:
            # General case: use lens equation approximation
            # For small angles: angle ≈ tan(angle) ≈ y/f
            if lens.focal_length != 0:
                deflection = -intersection_y / lens.focal_length
                new_direction_y = ray.direction_y + deflection
                # Normalize direction vector
                norm = np.sqrt(ray.direction_x**2 + new_direction_y**2)
                new_direction_x = ray.direction_x / norm
                new_direction_y = new_direction_y / norm
            else:
                new_direction_x, new_direction_y = ray.direction_x, ray.direction_y
        
        # Create refracted ray
        refracted_ray = LightRay(
            x=lens.x_position,
            y=intersection_y,
            direction_x=new_direction_x,
            direction_y=new_direction_y,
            intensity=ray.intensity * 0.95,  # Small loss due to reflection
            color=ray.color
        )
        
        return [refracted_ray]
    
    def trace_ray_through_mirror(self, ray: LightRay, mirror: Mirror) -> List[LightRay]:
        """
        Trace a ray reflecting from a mirror.
        
        Args:
            ray: Incident ray
            mirror: Mirror to reflect from
            
        Returns:
            List containing the reflected ray
        """
        # Calculate intersection with mirror
        if abs(ray.direction_x) < 1e-10:
            return []
        
        t = (mirror.x_position - ray.x) / ray.direction_x
        if t < 0:
            return []
        
        intersection_y = ray.y + t * ray.direction_y
        
        if abs(intersection_y) > mirror.aperture / 2:
            return []  # Ray blocked by aperture
        
        # Apply mirror reflection rules
        if mirror.mirror_type == "plane":
            # Plane mirror: simple reflection
            new_direction_x = -ray.direction_x
            new_direction_y = ray.direction_y
        else:
            # Spherical mirror: use focal point
            if abs(ray.direction_y) < 1e-6:  # Ray parallel to axis
                if mirror.focal_length != 0:
                    # Reflects toward/away from focal point
                    focal_y_offset = intersection_y
                    focal_distance = abs(mirror.focal_length - 0)  # Distance to focal point
                    if focal_distance > 0:
                        new_direction_y = -focal_y_offset / abs(mirror.focal_length) * np.sign(mirror.focal_length)
                        new_direction_x = -np.sqrt(1 - new_direction_y**2) * np.sign(ray.direction_x)
                    else:
                        new_direction_x = -ray.direction_x
                        new_direction_y = ray.direction_y
                else:
                    new_direction_x = -ray.direction_x
                    new_direction_y = ray.direction_y
            else:
                # Simple reflection for now (can be improved)
                new_direction_x = -ray.direction_x
                new_direction_y = ray.direction_y
        
        reflected_ray = LightRay(
            x=mirror.x_position,
            y=intersection_y,
            direction_x=new_direction_x,
            direction_y=new_direction_y,
            intensity=ray.intensity * 0.9,  # Loss due to absorption
            color=ray.color
        )
        
        return [reflected_ray]
    
    def plot_optical_system(self, x_range: Tuple[float, float] = (-10, 10),
                           y_range: Tuple[float, float] = (-5, 5),
                           object_height: float = 2.0,
                           object_distance: float = 6.0):
        """
        Plot the optical system with elements, object, and ray traces.
        
        Args:
            x_range: Range for x-axis plotting
            y_range: Range for y-axis plotting
            object_height: Height of the object
            object_distance: Distance of object from first element
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw optical axis
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Draw optical elements
        for element in self.elements:
            if isinstance(element, Lens):
                self._draw_lens(ax, element)
            elif isinstance(element, Mirror):
                self._draw_mirror(ax, element)
        
        # Draw object
        if self.elements:
            object_x = self.elements[0].x_position - object_distance
            ax.arrow(object_x, 0, 0, object_height, head_width=0.2, head_length=0.2,
                    fc='green', ec='green', linewidth=3)
            ax.text(object_x - 0.5, object_height/2, 'Object', fontsize=10,
                   verticalalignment='center')
        
        # Trace principal rays
        if self.elements and len(self.elements) == 1:
            element = self.elements[0]
            object_x = element.x_position - object_distance
            
            # Ray 1: Parallel to axis
            ray1 = LightRay(object_x, object_height, 1.0, 0.0, color='red')
            self._trace_and_plot_ray(ax, ray1, x_range)
            
            # Ray 2: Through optical center (lens) or toward focal point (mirror)
            if isinstance(element, Lens):
                # Ray through center
                direction_y = -object_height / object_distance
                direction_x = np.sqrt(1 - direction_y**2)
                ray2 = LightRay(object_x, object_height, direction_x, direction_y, color='blue')
            else:  # Mirror
                # Ray toward focal point
                focal_x = element.x_position + element.focal_length
                direction_x = focal_x - object_x
                direction_y = -object_height
                norm = np.sqrt(direction_x**2 + direction_y**2)
                ray2 = LightRay(object_x, object_height, direction_x/norm, direction_y/norm, color='blue')
            
            self._trace_and_plot_ray(ax, ray2, x_range)
            
            # Calculate and draw image
            di, mag = (self.lens_equation if isinstance(element, Lens) 
                      else self.mirror_equation)(element.focal_length, object_distance)
            
            if abs(di) != float('inf'):
                image_x = element.x_position + di
                image_height = mag * object_height
                
                if x_range[0] <= image_x <= x_range[1]:
                    ax.arrow(image_x, 0, 0, image_height, head_width=0.2, head_length=0.2,
                            fc='orange', ec='orange', linewidth=3, alpha=0.8)
                    
                    image_type = "Real" if di > 0 else "Virtual"
                    orientation = "Inverted" if mag < 0 else "Upright"
                    ax.text(image_x + 0.5, image_height/2, f'Image\n({image_type}, {orientation})',
                           fontsize=10, verticalalignment='center')
        
        # Format plot
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('Distance along optical axis')
        ax.set_ylabel('Height')
        ax.set_title('Geometric Optics: Mirrors and Lenses')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Legend
        legend_elements = []
        if self.elements:
            element = self.elements[0]
            element_name = f"{element.lens_type.title()} Lens" if isinstance(element, Lens) else f"{element.mirror_type.title()} Mirror"
            legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2, label=f'{element_name} (f={element.focal_length:.1f})'))
        
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', linewidth=2, label='Principal Ray 1'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Principal Ray 2'),
            plt.Line2D([0], [0], color='green', linewidth=3, label='Object'),
            plt.Line2D([0], [0], color='orange', linewidth=3, label='Image')
        ])
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig, ax
    
    def _draw_lens(self, ax, lens: Lens):
        """Draw a lens on the plot."""
        x = lens.x_position
        h = lens.aperture / 2
        
        if lens.is_converging:
            # Converging lens (biconvex shape)
            lens_curve = 0.3
            left_curve_x = np.linspace(x - lens_curve, x, 50)
            left_curve_y_top = h * np.sqrt(1 - ((left_curve_x - x) / lens_curve)**2)
            left_curve_y_bottom = -left_curve_y_top
            
            right_curve_x = np.linspace(x, x + lens_curve, 50)
            right_curve_y_top = h * np.sqrt(1 - ((right_curve_x - x) / lens_curve)**2)
            right_curve_y_bottom = -right_curve_y_top
            
            ax.plot(left_curve_x, left_curve_y_top, 'b-', linewidth=3)
            ax.plot(left_curve_x, left_curve_y_bottom, 'b-', linewidth=3)
            ax.plot(right_curve_x, right_curve_y_top, 'b-', linewidth=3)
            ax.plot(right_curve_x, right_curve_y_bottom, 'b-', linewidth=3)
        else:
            # Diverging lens (biconcave shape)
            lens_curve = 0.3
            ax.plot([x - lens_curve, x, x + lens_curve], [h, 0, h], 'b-', linewidth=3)
            ax.plot([x - lens_curve, x, x + lens_curve], [-h, 0, -h], 'b-', linewidth=3)
        
        # Draw focal points
        if abs(lens.focal_length) != float('inf'):
            ax.plot(x - abs(lens.focal_length), 0, 'bo', markersize=6)
            ax.plot(x + abs(lens.focal_length), 0, 'bo', markersize=6)
            ax.text(x + abs(lens.focal_length), -0.5, f'F\n({lens.focal_length:.1f})', 
                   fontsize=9, ha='center')
    
    def _draw_mirror(self, ax, mirror: Mirror):
        """Draw a mirror on the plot."""
        x = mirror.x_position
        h = mirror.aperture / 2
        
        if mirror.mirror_type == "plane":
            # Plane mirror
            ax.plot([x, x], [-h, h], 'k-', linewidth=4)
            # Add hatching to show reflecting surface
            for i in range(int(h * 4)):
                y = -h + i * 0.5
                if abs(y) <= h:
                    ax.plot([x, x + 0.2], [y, y + 0.2], 'k-', linewidth=1)
        elif mirror.mirror_type == "concave":
            # Concave mirror (curved inward)
            curve_x = np.linspace(x - 0.3, x, 50)
            curve_y_top = h * np.sqrt(1 - ((curve_x - x) / 0.3)**2)
            curve_y_bottom = -curve_y_top
            ax.plot(curve_x, curve_y_top, 'k-', linewidth=4)
            ax.plot(curve_x, curve_y_bottom, 'k-', linewidth=4)
        else:  # convex
            # Convex mirror (curved outward)
            curve_x = np.linspace(x, x + 0.3, 50)
            curve_y_top = h * np.sqrt(1 - ((curve_x - x) / 0.3)**2)
            curve_y_bottom = -curve_y_top
            ax.plot(curve_x, curve_y_top, 'k-', linewidth=4)
            ax.plot(curve_x, curve_y_bottom, 'k-', linewidth=4)
        
        # Draw focal point
        if abs(mirror.focal_length) != float('inf'):
            ax.plot(x + mirror.focal_length, 0, 'ko', markersize=6)
            ax.text(x + mirror.focal_length, -0.5, f'F\n({mirror.focal_length:.1f})', 
                   fontsize=9, ha='center')
    
    def _trace_and_plot_ray(self, ax, ray: LightRay, x_range: Tuple[float, float]):
        """Trace a ray through the system and plot it."""
        current_ray = ray
        ray_segments = [(ray.x, ray.y)]
        
        for element in self.elements:
            if isinstance(element, Lens):
                traced_rays = self.trace_ray_through_lens(current_ray, element)
            else:  # Mirror
                traced_rays = self.trace_ray_through_mirror(current_ray, element)
            
            if traced_rays:
                next_ray = traced_rays[0]
                ray_segments.append((element.x_position, next_ray.y))
                current_ray = next_ray
            else:
                break
        
        # Extend ray to plot boundaries
        if len(ray_segments) >= 2:
            last_x, last_y = ray_segments[-1]
            # Extend the final ray segment
            if abs(current_ray.direction_x) > 1e-10:
                if current_ray.direction_x > 0:
                    final_x = x_range[1]
                else:
                    final_x = x_range[0]
                final_y = last_y + (final_x - last_x) * current_ray.direction_y / current_ray.direction_x
                ray_segments.append((final_x, final_y))
        
        # Plot ray segments
        for i in range(len(ray_segments) - 1):
            x1, y1 = ray_segments[i]
            x2, y2 = ray_segments[i + 1]
            ax.plot([x1, x2], [y1, y2], color=ray.color, linewidth=2)
            
            # Add arrowhead
            if i == len(ray_segments) - 2:  # Last segment
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx, dy = dx / length, dy / length
                    ax.arrow(x1 + 0.7 * (x2 - x1), y1 + 0.7 * (y2 - y1), 
                            0.3 * dx, 0.3 * dy, head_width=0.2, head_length=0.3,
                            fc=ray.color, ec=ray.color)


def demonstrate_lenses():
    """Demonstrate different types of lenses."""
    print("🔍 Lens Optics Demonstration")
    print("=" * 35)
    
    # Converging lens
    print("\n📐 Converging Lens (f = +3.0)")
    system1 = GeometricOpticsSystem()
    converging_lens = Lens(x_position=0, focal_length=3.0, aperture=4.0, name="Converging Lens")
    system1.add_element(converging_lens)
    
    # Test different object distances
    object_distances = [6.0, 4.5, 3.0, 2.0, 1.5]
    
    print(f"{'Object Dist':<12}{'Image Dist':<12}{'Magnification':<14}{'Image Type'}")
    print("-" * 55)
    
    for do in object_distances:
        di, mag = system1.lens_equation(converging_lens.focal_length, do)
        
        if abs(di) == float('inf'):
            image_type = "At infinity"
            print(f"{do:<12.1f}{'∞':<12}{'∞':<14}{image_type}")
        else:
            image_type = "Real, inverted" if di > 0 and mag < 0 else "Virtual, upright"
            print(f"{do:<12.1f}{di:<12.1f}{mag:<14.2f}{image_type}")
    
    # Create visualization for converging lens
    fig1, ax1 = system1.plot_optical_system(x_range=(-8, 8), y_range=(-4, 4), 
                                           object_distance=4.5, object_height=2.0)
    ax1.set_title("Converging Lens: Real Image Formation")
    plt.show()
    
    # Diverging lens
    print(f"\n📐 Diverging Lens (f = -3.0)")
    system2 = GeometricOpticsSystem()
    diverging_lens = Lens(x_position=0, focal_length=-3.0, aperture=4.0, name="Diverging Lens")
    system2.add_element(diverging_lens)
    
    print(f"{'Object Dist':<12}{'Image Dist':<12}{'Magnification':<14}{'Image Type'}")
    print("-" * 55)
    
    for do in object_distances:
        di, mag = system2.lens_equation(diverging_lens.focal_length, do)
        image_type = "Virtual, upright"
        print(f"{do:<12.1f}{di:<12.1f}{mag:<14.2f}{image_type}")
    
    # Create visualization for diverging lens
    fig2, ax2 = system2.plot_optical_system(x_range=(-8, 8), y_range=(-4, 4), 
                                           object_distance=4.5, object_height=2.0)
    ax2.set_title("Diverging Lens: Virtual Image Formation")
    plt.show()


def demonstrate_mirrors():
    """Demonstrate different types of mirrors."""
    print("\n🪞 Mirror Optics Demonstration")
    print("=" * 35)
    
    # Concave mirror
    print("\n📐 Concave Mirror (f = +2.5)")
    system1 = GeometricOpticsSystem()
    concave_mirror = Mirror(x_position=0, focal_length=2.5, aperture=4.0, name="Concave Mirror")
    system1.add_element(concave_mirror)
    
    object_distances = [5.0, 3.75, 2.5, 1.5, 1.0]
    
    print(f"{'Object Dist':<12}{'Image Dist':<12}{'Magnification':<14}{'Image Type'}")
    print("-" * 55)
    
    for do in object_distances:
        di, mag = system1.mirror_equation(concave_mirror.focal_length, do)
        
        if abs(di) == float('inf'):
            image_type = "At infinity"
            print(f"{do:<12.1f}{'∞':<12}{'∞':<14}{image_type}")
        elif di > 0:
            image_type = "Real, inverted" if mag < 0 else "Real, upright"
            print(f"{do:<12.1f}{di:<12.1f}{mag:<14.2f}{image_type}")
        else:
            image_type = "Virtual, upright"
            print(f"{do:<12.1f}{di:<12.1f}{mag:<14.2f}{image_type}")
    
    # Visualization for concave mirror
    fig1, ax1 = system1.plot_optical_system(x_range=(-8, 6), y_range=(-4, 4), 
                                           object_distance=3.75, object_height=2.0)
    ax1.set_title("Concave Mirror: Real Image Formation")
    plt.show()
    
    # Convex mirror
    print(f"\n📐 Convex Mirror (f = -2.5)")
    system2 = GeometricOpticsSystem()
    convex_mirror = Mirror(x_position=0, focal_length=-2.5, aperture=4.0, name="Convex Mirror")
    system2.add_element(convex_mirror)
    
    print(f"{'Object Dist':<12}{'Image Dist':<12}{'Magnification':<14}{'Image Type'}")
    print("-" * 55)
    
    for do in object_distances:
        di, mag = system2.mirror_equation(convex_mirror.focal_length, do)
        image_type = "Virtual, upright"
        print(f"{do:<12.1f}{di:<12.1f}{mag:<14.2f}{image_type}")
    
    # Visualization for convex mirror
    fig2, ax2 = system2.plot_optical_system(x_range=(-8, 6), y_range=(-4, 4), 
                                           object_distance=3.75, object_height=2.0)
    ax2.set_title("Convex Mirror: Virtual Image Formation")
    plt.show()


if __name__ == "__main__":
    print("🔍 Welcome to Mirrors and Lenses Explorer! 🔍")
    print("This script demonstrates image formation with mirrors and lenses.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_lenses()
        demonstrate_mirrors()
        
        print(f"\n" + "="*60)
        print("🎓 Summary of Key Concepts:")
        print("="*60)
        print("1. Thin Lens Equation: 1/f = 1/do + 1/di")
        print("2. Mirror Equation: 1/f = 1/do + 1/di (same form as lens equation)")
        print("3. Magnification: m = -di/do")
        print("4. Sign Conventions:")
        print("   • Focal length: positive for converging, negative for diverging")
        print("   • Object distance: always positive")
        print("   • Image distance: positive for real images, negative for virtual")
        print("5. Ray Tracing Rules:")
        print("   • Ray parallel to axis → through focal point")
        print("   • Ray through optical center → continues straight (lenses)")
        print("   • Ray through focal point → emerges parallel")
        
        print(f"\n💡 Applications:")
        print("• Camera lenses (converging lens systems)")
        print("• Telescope mirrors (large concave mirrors)")
        print("• Car mirrors (convex mirrors for wide field of view)")
        print("• Magnifying glasses (converging lenses)")
        print("• Corrective lenses (eyeglasses, contact lenses)")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring mirrors and lenses! ✨")