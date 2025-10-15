#!/usr/bin/env python3
"""
Geometric Optics: Optical Instruments
=====================================

This module demonstrates the principles and design of common optical instruments:
- Telescopes (refracting and reflecting)
- Microscopes (compound microscope)
- Cameras and projectors
- Magnifying glasses
- Binoculars
- Fiber optics principles

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
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class OpticalComponent:
    """Represents a component in an optical instrument."""
    name: str
    focal_length: float
    diameter: float
    position: float
    component_type: str  # 'lens', 'mirror', 'eyepiece', 'objective'


class OpticalInstrument:
    """Base class for optical instruments."""
    
    def __init__(self, name: str):
        self.name = name
        self.components: List[OpticalComponent] = []
        self.specifications: Dict = {}
    
    def add_component(self, component: OpticalComponent):
        """Add an optical component to the instrument."""
        self.components.append(component)
    
    def calculate_magnification(self) -> float:
        """Calculate total magnification of the instrument."""
        return 1.0  # Override in subclasses
    
    def calculate_resolving_power(self, wavelength: float = 550e-9) -> float:
        """Calculate resolving power based on aperture."""
        return wavelength  # Override in subclasses


class Telescope(OpticalInstrument):
    """Represents a telescope (refracting or reflecting)."""
    
    def __init__(self, objective_focal_length: float, eyepiece_focal_length: float, 
                 objective_diameter: float, telescope_type: str = "refracting"):
        super().__init__(f"{telescope_type.title()} Telescope")
        
        self.objective_fl = objective_focal_length
        self.eyepiece_fl = eyepiece_focal_length
        self.objective_diameter = objective_diameter
        self.telescope_type = telescope_type
        
        # Add components
        if telescope_type == "refracting":
            objective = OpticalComponent("Objective Lens", objective_focal_length, 
                                       objective_diameter, 0, "lens")
        else:  # reflecting
            objective = OpticalComponent("Primary Mirror", objective_focal_length, 
                                       objective_diameter, 0, "mirror")
        
        eyepiece = OpticalComponent("Eyepiece", eyepiece_focal_length, 
                                  min(25.0, objective_diameter/4), 
                                  objective_focal_length + eyepiece_focal_length, "lens")
        
        self.add_component(objective)
        self.add_component(eyepiece)
        
        self.specifications = {
            "Magnification": self.calculate_magnification(),
            "Light Gathering Power": self.calculate_light_gathering_power(),
            "Resolving Power": self.calculate_resolving_power(),
            "F-ratio": objective_focal_length / objective_diameter
        }
    
    def calculate_magnification(self) -> float:
        """Calculate angular magnification of telescope."""
        return self.objective_fl / self.eyepiece_fl
    
    def calculate_light_gathering_power(self) -> float:
        """Calculate light gathering power relative to naked eye (7mm pupil)."""
        return (self.objective_diameter / 7.0) ** 2
    
    def calculate_resolving_power(self, wavelength: float = 550e-9) -> float:
        """Calculate angular resolution in arcseconds."""
        # Rayleigh criterion: θ = 1.22λ/D
        resolution_rad = 1.22 * wavelength / (self.objective_diameter / 1000)  # Convert mm to m
        resolution_arcsec = resolution_rad * (180 * 3600 / np.pi)  # Convert to arcseconds
        return resolution_arcsec
    
    def plot_telescope_diagram(self):
        """Plot ray diagram for the telescope."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Telescope tube length
        tube_length = self.objective_fl + self.eyepiece_fl
        
        # Draw optical axis
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Draw objective
        obj_x = 0
        obj_r = self.objective_diameter / 2
        
        if self.telescope_type == "refracting":
            # Draw lens
            ax.plot([obj_x, obj_x], [-obj_r, obj_r], 'b-', linewidth=4, label='Objective Lens')
        else:
            # Draw concave mirror
            mirror_curve = np.linspace(-obj_r, obj_r, 50)
            mirror_x = obj_x - 0.5 * (1 - np.sqrt(1 - (mirror_curve / obj_r)**2))
            ax.plot(mirror_x, mirror_curve, 'k-', linewidth=4, label='Primary Mirror')
        
        # Draw eyepiece
        eye_x = tube_length
        eye_r = min(15.0, obj_r / 2)
        ax.plot([eye_x, eye_x], [-eye_r, eye_r], 'g-', linewidth=3, label='Eyepiece')
        
        # Draw focal points
        ax.plot(self.objective_fl, 0, 'ro', markersize=8, label='Objective Focus')
        ax.plot(eye_x - self.eyepiece_fl, 0, 'go', markersize=6, label='Eyepiece Focus')
        
        # Draw parallel rays from distant object
        ray_heights = [-obj_r * 0.8, -obj_r * 0.4, obj_r * 0.4, obj_r * 0.8]
        colors = ['red', 'orange', 'blue', 'purple']
        
        for i, height in enumerate(ray_heights):
            # Incident parallel rays
            ax.arrow(-tube_length * 0.3, height, tube_length * 0.3, 0,
                    head_width=obj_r * 0.05, head_length=tube_length * 0.02,
                    fc=colors[i], ec=colors[i], linewidth=2)
            
            # Rays converge to objective focal point
            focal_y = height * self.eyepiece_fl / self.objective_fl  # Intermediate image height
            ax.plot([obj_x, self.objective_fl], [height, focal_y], colors[i], linewidth=2)
            
            # Rays from objective focus to eyepiece
            ax.plot([self.objective_fl, eye_x - self.eyepiece_fl], [focal_y, focal_y], 
                   colors[i], linewidth=2)
            
            # Parallel rays exit eyepiece
            exit_angle = focal_y / self.eyepiece_fl * self.calculate_magnification()
            exit_y = eye_x + tube_length * 0.2
            ax.arrow(eye_x, focal_y, tube_length * 0.2, 
                    exit_angle * tube_length * 0.2,
                    head_width=obj_r * 0.05, head_length=tube_length * 0.02,
                    fc=colors[i], ec=colors[i], linewidth=2, alpha=0.7)
        
        # Annotations
        ax.text(self.objective_fl / 2, -obj_r * 1.2, f'fo = {self.objective_fl:.0f}mm', 
               fontsize=12, ha='center')
        ax.text(eye_x - self.eyepiece_fl / 2, obj_r * 1.2, f'fe = {self.eyepiece_fl:.0f}mm', 
               fontsize=12, ha='center')
        ax.text(tube_length / 2, obj_r * 1.5, 
               f'Magnification = fo/fe = {self.calculate_magnification():.1f}×', 
               fontsize=14, ha='center', weight='bold')
        
        # Format plot
        ax.set_xlim(-tube_length * 0.4, tube_length * 1.3)
        ax.set_ylim(-obj_r * 1.8, obj_r * 1.8)
        ax.set_xlabel('Distance (mm)', fontsize=12)
        ax.set_ylabel('Height (mm)', fontsize=12)
        ax.set_title(f'{self.name} - Ray Diagram', fontsize=14, weight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax


class Microscope(OpticalInstrument):
    """Represents a compound microscope."""
    
    def __init__(self, objective_focal_length: float, eyepiece_focal_length: float,
                 tube_length: float = 160.0):  # Standard tube length
        super().__init__("Compound Microscope")
        
        self.objective_fl = objective_focal_length
        self.eyepiece_fl = eyepiece_focal_length
        self.tube_length = tube_length
        
        # Add components
        objective = OpticalComponent("Objective Lens", objective_focal_length, 
                                   8.0, 0, "lens")
        eyepiece = OpticalComponent("Eyepiece", eyepiece_focal_length, 
                                  12.0, tube_length, "lens")
        
        self.add_component(objective)
        self.add_component(eyepiece)
        
        self.specifications = {
            "Total Magnification": self.calculate_magnification(),
            "Objective Magnification": self.calculate_objective_magnification(),
            "Eyepiece Magnification": self.calculate_eyepiece_magnification(),
            "Numerical Aperture": self.calculate_numerical_aperture(),
            "Resolution Limit": self.calculate_resolving_power()
        }
    
    def calculate_objective_magnification(self) -> float:
        """Calculate magnification of objective lens."""
        return self.tube_length / self.objective_fl
    
    def calculate_eyepiece_magnification(self) -> float:
        """Calculate magnification of eyepiece (assumes 25cm viewing distance)."""
        return 250.0 / self.eyepiece_fl  # 25cm = 250mm
    
    def calculate_magnification(self) -> float:
        """Calculate total magnification."""
        return self.calculate_objective_magnification() * self.calculate_eyepiece_magnification()
    
    def calculate_numerical_aperture(self, medium_index: float = 1.0) -> float:
        """Calculate numerical aperture (simplified)."""
        # For air, typical NA ranges from 0.1 to 0.95
        # This is a simplified calculation
        return min(0.95, medium_index * 0.5)
    
    def calculate_resolving_power(self, wavelength: float = 550e-9) -> float:
        """Calculate resolution limit using Abbe criterion."""
        NA = self.calculate_numerical_aperture()
        resolution = 0.61 * wavelength / NA  # Abbe limit
        return resolution * 1e6  # Convert to micrometers
    
    def plot_microscope_diagram(self):
        """Plot ray diagram for the microscope."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Draw optical axis
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Object position (very close to objective)
        object_distance = self.objective_fl * 1.1  # Slightly outside focal length
        object_height = 0.5  # Small object
        
        # Draw object
        ax.arrow(0, 0, 0, object_height, head_width=2, head_length=0.1,
                fc='red', ec='red', linewidth=3)
        ax.text(-5, object_height/2, 'Object', fontsize=10, va='center')
        
        # Draw objective lens
        obj_x = object_distance
        obj_r = 4.0
        ax.plot([obj_x, obj_x], [-obj_r, obj_r], 'b-', linewidth=4, label='Objective Lens')
        
        # Calculate intermediate image
        di_obj = (object_distance * self.objective_fl) / (object_distance - self.objective_fl)
        mag_obj = -di_obj / object_distance
        intermediate_image_height = mag_obj * object_height
        intermediate_image_pos = obj_x + di_obj
        
        # Draw intermediate image
        ax.arrow(intermediate_image_pos, 0, 0, intermediate_image_height, 
                head_width=2, head_length=abs(intermediate_image_height)*0.1,
                fc='orange', ec='orange', linewidth=3)
        ax.text(intermediate_image_pos + 5, intermediate_image_height/2, 
               'Intermediate\nImage', fontsize=10, va='center')
        
        # Draw eyepiece
        eye_x = obj_x + self.tube_length
        eye_r = 6.0
        ax.plot([eye_x, eye_x], [-eye_r, eye_r], 'g-', linewidth=4, label='Eyepiece')
        
        # Draw principal rays
        ray_colors = ['red', 'blue', 'green']
        
        # Ray 1: Top of object through center of objective
        ax.plot([0, obj_x], [object_height, object_height], ray_colors[0], linewidth=2)
        ax.plot([obj_x, intermediate_image_pos], [object_height, intermediate_image_height], 
               ray_colors[0], linewidth=2)
        
        # Ray 2: Top of object parallel to axis, then through objective focus
        ax.plot([0, obj_x], [object_height, object_height], ray_colors[1], linewidth=2)
        ax.plot([obj_x, obj_x + self.objective_fl], [object_height, 0], ray_colors[1], linewidth=2)
        
        # Continue rays through eyepiece (simplified)
        eye_object_dist = eye_x - intermediate_image_pos
        
        # Virtual image formation by eyepiece
        virtual_image_dist = -(eye_object_dist * self.eyepiece_fl) / (eye_object_dist - self.eyepiece_fl)
        virtual_mag = -virtual_image_dist / eye_object_dist
        final_image_height = virtual_mag * intermediate_image_height
        final_image_pos = eye_x + virtual_image_dist
        
        # Draw rays exiting eyepiece
        for i, color in enumerate(ray_colors[:2]):
            exit_height = intermediate_image_height * (1 - i * 0.5)
            ax.arrow(eye_x, exit_height, 30, final_image_height * 0.3 * (1 - i * 0.5),
                    head_width=2, head_length=3, fc=color, ec=color, 
                    linewidth=2, alpha=0.7)
        
        # Draw focal points
        ax.plot(obj_x + self.objective_fl, 0, 'bo', markersize=6)
        ax.plot(eye_x + self.eyepiece_fl, 0, 'go', markersize=6)
        ax.plot(eye_x - self.eyepiece_fl, 0, 'go', markersize=6)
        
        # Annotations
        ax.text(obj_x/2, -obj_r*1.5, f'Object distance ≈ {object_distance:.1f}mm', 
               fontsize=10, ha='center')
        ax.text(eye_x, -eye_r*1.5, f'Tube length = {self.tube_length:.0f}mm', 
               fontsize=10, ha='center')
        ax.text(eye_x + 40, final_image_height/2, 
               f'Virtual Image\n(at 25cm from eye)', fontsize=10, va='center')
        
        # Specifications box
        specs_text = (f'Objective: f = {self.objective_fl:.1f}mm\n'
                     f'Eyepiece: f = {self.eyepiece_fl:.1f}mm\n'
                     f'Total Magnification = {self.calculate_magnification():.0f}×\n'
                     f'Resolution ≈ {self.calculate_resolving_power():.2f} μm')
        
        ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Format plot
        ax.set_xlim(-20, eye_x + 80)
        ax.set_ylim(-obj_r*2, max(final_image_height, intermediate_image_height) + 5)
        ax.set_xlabel('Distance (mm)', fontsize=12)
        ax.set_ylabel('Height (mm)', fontsize=12)
        ax.set_title('Compound Microscope - Ray Diagram', fontsize=14, weight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


class Camera(OpticalInstrument):
    """Represents a camera system."""
    
    def __init__(self, focal_length: float, aperture_diameter: float, sensor_size: Tuple[float, float]):
        super().__init__("Camera")
        
        self.focal_length = focal_length
        self.aperture_diameter = aperture_diameter
        self.sensor_width, self.sensor_height = sensor_size
        
        # Add lens component
        lens = OpticalComponent("Camera Lens", focal_length, aperture_diameter, 0, "lens")
        self.add_component(lens)
        
        self.specifications = {
            "Focal Length": focal_length,
            "F-number": self.calculate_f_number(),
            "Field of View": self.calculate_field_of_view(),
            "Depth of Field": "Variable with aperture",
            "Sensor Size": f"{sensor_size[0]}×{sensor_size[1]}mm"
        }
    
    def calculate_f_number(self) -> float:
        """Calculate f-number (f-stop)."""
        return self.focal_length / self.aperture_diameter
    
    def calculate_field_of_view(self) -> float:
        """Calculate horizontal field of view in degrees."""
        fov_rad = 2 * np.arctan(self.sensor_width / (2 * self.focal_length))
        return np.degrees(fov_rad)


def demonstrate_telescope():
    """Demonstrate telescope optics."""
    print("🔭 Telescope Optics")
    print("=" * 25)
    
    # Refracting telescope
    refractor = Telescope(objective_focal_length=1000, eyepiece_focal_length=25, 
                         objective_diameter=100, telescope_type="refracting")
    
    print(f"\n📡 {refractor.name}:")
    for key, value in refractor.specifications.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Plot refracting telescope
    fig1, ax1 = refractor.plot_telescope_diagram()
    plt.show()
    
    # Reflecting telescope
    reflector = Telescope(objective_focal_length=1200, eyepiece_focal_length=20, 
                         objective_diameter=200, telescope_type="reflecting")
    
    print(f"\n📡 {reflector.name}:")
    for key, value in reflector.specifications.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Plot reflecting telescope
    fig2, ax2 = reflector.plot_telescope_diagram()
    plt.show()


def demonstrate_microscope():
    """Demonstrate microscope optics."""
    print("\n🔬 Microscope Optics")
    print("=" * 25)
    
    # High-power microscope
    microscope = Microscope(objective_focal_length=4.0, eyepiece_focal_length=10.0, 
                           tube_length=160.0)
    
    print(f"\n🔍 {microscope.name}:")
    for key, value in microscope.specifications.items():
        if isinstance(value, float):
            if "Resolution" in key:
                print(f"   {key}: {value:.3f} μm")
            else:
                print(f"   {key}: {value:.1f}")
        else:
            print(f"   {key}: {value}")
    
    # Plot microscope
    fig, ax = microscope.plot_microscope_diagram()
    plt.show()


def demonstrate_camera():
    """Demonstrate camera optics."""
    print("\n📷 Camera Optics")
    print("=" * 20)
    
    # DSLR camera with 50mm lens
    camera = Camera(focal_length=50.0, aperture_diameter=12.5, sensor_size=(36.0, 24.0))  # Full frame
    
    print(f"\n📸 {camera.name}:")
    for key, value in camera.specifications.items():
        if isinstance(value, float):
            if "Field" in key:
                print(f"   {key}: {value:.1f}°")
            else:
                print(f"   {key}: {value:.1f}")
        else:
            print(f"   {key}: {value}")
    
    # Compare different focal lengths
    focal_lengths = [24, 35, 50, 85, 135, 200]  # mm
    print(f"\n📏 Field of View Comparison:")
    print(f"{'Focal Length (mm)':<18}{'F-number':<12}{'Field of View (°)'}")
    print("-" * 45)
    
    for fl in focal_lengths:
        cam = Camera(focal_length=fl, aperture_diameter=fl/2.8, sensor_size=(36.0, 24.0))
        print(f"{fl:<18}{cam.calculate_f_number():<12.1f}{cam.calculate_field_of_view():<12.1f}")


def demonstrate_fiber_optics():
    """Demonstrate fiber optics principles."""
    print("\n🌐 Fiber Optics")
    print("=" * 20)
    
    print("\n💡 Fiber Optics Principles:")
    print("1. Total Internal Reflection confines light within the fiber core")
    print("2. Critical angle determines the numerical aperture")
    print("3. Step-index vs gradient-index fibers")
    
    # Fiber parameters
    core_index = 1.50      # Glass core
    cladding_index = 1.46  # Lower index cladding
    
    # Calculate numerical aperture
    NA = np.sqrt(core_index**2 - cladding_index**2)
    acceptance_angle = np.degrees(np.arcsin(NA))
    critical_angle = np.degrees(np.arcsin(cladding_index / core_index))
    
    print(f"\n📊 Fiber Specifications:")
    print(f"   Core Index: {core_index}")
    print(f"   Cladding Index: {cladding_index}")
    print(f"   Numerical Aperture: {NA:.3f}")
    print(f"   Acceptance Angle: ±{acceptance_angle:.1f}°")
    print(f"   Critical Angle: {critical_angle:.1f}°")
    
    # Plot fiber cross-section and ray paths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Fiber cross-section
    core_circle = plt.Circle((0, 0), 25, color='lightblue', alpha=0.7, label='Core')
    cladding_circle = plt.Circle((0, 0), 62.5, fill=False, color='blue', linewidth=2, label='Cladding')
    
    ax1.add_patch(core_circle)
    ax1.add_patch(cladding_circle)
    ax1.set_xlim(-80, 80)
    ax1.set_ylim(-80, 80)
    ax1.set_aspect('equal')
    ax1.set_title('Optical Fiber Cross-Section')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ray propagation in fiber
    fiber_length = 100
    core_radius = 25
    
    # Draw fiber
    ax2.fill_between([0, fiber_length], [-core_radius, -core_radius], 
                    [core_radius, core_radius], color='lightblue', alpha=0.7, label='Core')
    ax2.plot([0, fiber_length], [core_radius, core_radius], 'b-', linewidth=2)
    ax2.plot([0, fiber_length], [-core_radius, -core_radius], 'b-', linewidth=2, label='Cladding Interface')
    
    # Draw light rays
    angles = [0, 10, 20]  # degrees from axis
    colors = ['red', 'green', 'blue']
    
    for angle, color in zip(angles, colors):
        angle_rad = np.radians(angle)
        if angle <= acceptance_angle:
            # Ray propagates by total internal reflection
            y_start = 0
            y = y_start
            x = 0
            
            ray_x = [x]
            ray_y = [y]
            
            while x < fiber_length:
                # Propagate until hitting boundary
                if np.tan(angle_rad) != 0:
                    if np.tan(angle_rad) > 0:
                        x_boundary = (core_radius - y) / np.tan(angle_rad) + x
                        y_boundary = core_radius
                    else:
                        x_boundary = (-core_radius - y) / np.tan(angle_rad) + x
                        y_boundary = -core_radius
                    
                    if x_boundary > fiber_length:
                        ray_x.append(fiber_length)
                        ray_y.append(y + (fiber_length - x) * np.tan(angle_rad))
                        break
                    else:
                        ray_x.append(x_boundary)
                        ray_y.append(y_boundary)
                        x, y = x_boundary, y_boundary
                        angle_rad = -angle_rad  # Reflect
                else:
                    ray_x.append(fiber_length)
                    ray_y.append(y)
                    break
            
            ax2.plot(ray_x, ray_y, color=color, linewidth=2, 
                    label=f'Ray at {angle}°' + (' (TIR)' if angle > 0 else ' (axial)'))
    
    ax2.set_xlim(0, fiber_length)
    ax2.set_ylim(-40, 40)
    ax2.set_xlabel('Distance along fiber')
    ax2.set_ylabel('Radial position')
    ax2.set_title('Light Propagation in Optical Fiber')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🔬 Welcome to Optical Instruments Explorer! 🔬")
    print("This script demonstrates the principles of various optical instruments.")
    print("\nPress Enter to start the demonstrations...")
    input()
    
    try:
        demonstrate_telescope()
        demonstrate_microscope()
        demonstrate_camera()
        demonstrate_fiber_optics()
        
        print(f"\n" + "="*70)
        print("🎓 Summary of Optical Instruments:")
        print("="*70)
        print("1. Telescopes:")
        print("   • Magnification = fo/fe (objective focal length / eyepiece focal length)")
        print("   • Light gathering ∝ (objective diameter)²")
        print("   • Resolution ∝ 1/(objective diameter)")
        
        print("\n2. Microscopes:")
        print("   • Total magnification = Mo × Me")
        print("   • Resolution limited by wavelength and numerical aperture")
        print("   • Working distance decreases with magnification")
        
        print("\n3. Cameras:")
        print("   • Field of view ∝ sensor size / focal length")
        print("   • Depth of field controlled by aperture (f-number)")
        print("   • Exposure = aperture area × shutter time")
        
        print("\n4. Fiber Optics:")
        print("   • Total internal reflection confines light")
        print("   • Numerical aperture determines light acceptance")
        print("   • Applications: telecommunications, medical endoscopy")
        
        print(f"\n💡 Key Design Principles:")
        print("• Aberration correction (spherical, chromatic, etc.)")
        print("• Light throughput vs resolution trade-offs")
        print("• Mechanical stability and alignment")
        print("• Coatings to reduce reflections and improve transmission")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring optical instruments! ✨")