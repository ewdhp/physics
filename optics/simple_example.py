#!/usr/bin/env python3
"""
Simple Geometric Optics Example
==============================

A quick demonstration of the geometric_optics module showing:
- Basic reflection and refraction
- How to create and use Ray and Interface objects
- Simple calculations with Snell's law

Run this script to see basic optics calculations without the full demo.
"""

import numpy as np
import matplotlib
# Set backend for GUI display
try:
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reflection_refraction import GeometricOptics, Ray, Interface


def simple_optics_example():
    """Simple example showing basic optics calculations."""
    
    print("🔬 Simple Geometric Optics Example")
    print("=" * 40)
    
    # Create an optics system
    optics = GeometricOptics()
    
    # Define an air-water interface
    interface = Interface(
        position=0.0,
        n1=1.0,    # air
        n2=1.33,   # water  
        name1="Air",
        name2="Water"
    )
    
    print(f"Interface: {interface.name1} (n={interface.n1}) → {interface.name2} (n={interface.n2})")
    
    # Test different incident angles
    incident_angles = [0, 15, 30, 45, 60]  # degrees
    
    print(f"\n{'Incident':<10}{'Refracted':<12}{'Reflectance':<12}")
    print(f"{'Angle (°)':<10}{'Angle (°)':<12}{'(%)':<12}")
    print("-" * 34)
    
    for angle_deg in incident_angles:
        angle_rad = np.radians(angle_deg)
        
        # Apply Snell's law
        refracted_rad = optics.snells_law(angle_rad, interface.n1, interface.n2)
        refracted_deg = np.degrees(refracted_rad)
        
        # Calculate reflectance
        reflectance = optics.fresnel_reflectance(angle_rad, interface.n1, interface.n2)
        
        print(f"{angle_deg:<10.0f}{refracted_deg:<12.1f}{reflectance*100:<12.1f}")
    
    # Calculate critical angle for water to air
    critical_rad = optics.critical_angle(interface.n2, interface.n1)
    critical_deg = np.degrees(critical_rad)
    
    print(f"\n🔴 Critical angle (water→air): {critical_deg:.1f}°")
    
    # Create a simple visualization
    print("\n📊 Creating simple ray diagram...")
    
    # Create some incident rays
    rays = [
        Ray(x=-2, y=1, direction=np.radians(135), intensity=1.0),  # 45° incident
        Ray(x=-1, y=1, direction=np.radians(150), intensity=1.0),  # 30° incident
    ]
    
    # Plot the rays
    fig, ax = optics.plot_ray_diagram(
        rays=rays,
        interfaces=[interface],
        x_range=(-3, 3),
        y_range=(-1.5, 1.5),
        ray_length=4
    )
    
    plt.show()
    
    print("\n✨ Example completed!")


if __name__ == "__main__":
    simple_optics_example()