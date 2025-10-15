#!/usr/bin/env python3
"""
Optics Calculator
================

A command-line calculator for quick geometric optics calculations.
Useful for homework, lab work, or quick checks of optical calculations.

Usage examples:
  python optics_calculator.py snell 30 1.0 1.5
  python optics_calculator.py critical 1.5 1.0  
  python optics_calculator.py brewster 1.0 1.5
  python optics_calculator.py reflectance 45 1.0 1.5
"""

import sys
import argparse
import numpy as np
from reflection_refraction import GeometricOptics


def calculate_snells_law(angle_deg, n1, n2):
    """Calculate refraction angle using Snell's law."""
    optics = GeometricOptics()
    angle_rad = np.radians(angle_deg)
    
    refracted_rad = optics.snells_law(angle_rad, n1, n2)
    
    if refracted_rad is None:
        print(f"❌ Total Internal Reflection!")
        print(f"   Incident angle ({angle_deg}°) exceeds critical angle")
        critical_rad = optics.critical_angle(n1, n2)
        if critical_rad:
            print(f"   Critical angle: {np.degrees(critical_rad):.2f}°")
    else:
        refracted_deg = np.degrees(refracted_rad)
        print(f"✅ Snell's Law Calculation:")
        print(f"   Incident medium: n₁ = {n1}")
        print(f"   Refracting medium: n₂ = {n2}")
        print(f"   Incident angle: θ₁ = {angle_deg}°")
        print(f"   Refracted angle: θ₂ = {refracted_deg:.2f}°")


def calculate_critical_angle(n1, n2):
    """Calculate critical angle for total internal reflection."""
    optics = GeometricOptics()
    
    if n1 <= n2:
        print(f"❌ No critical angle exists!")
        print(f"   Critical angle only exists when n₁ > n₂")
        print(f"   Given: n₁ = {n1}, n₂ = {n2}")
        return
    
    critical_rad = optics.critical_angle(n1, n2)
    critical_deg = np.degrees(critical_rad)
    
    print(f"✅ Critical Angle Calculation:")
    print(f"   Denser medium: n₁ = {n1}")
    print(f"   Rarer medium: n₂ = {n2}")
    print(f"   Critical angle: θc = {critical_deg:.2f}°")
    print(f"   Formula: θc = arcsin(n₂/n₁)")


def calculate_brewster_angle(n1, n2):
    """Calculate Brewster's angle for polarization."""
    brewster_rad = np.arctan(n2 / n1)
    brewster_deg = np.degrees(brewster_rad)
    
    print(f"✅ Brewster's Angle Calculation:")
    print(f"   Incident medium: n₁ = {n1}")
    print(f"   Refracting medium: n₂ = {n2}")
    print(f"   Brewster's angle: θB = {brewster_deg:.2f}°")
    print(f"   Formula: θB = arctan(n₂/n₁)")
    print(f"   At this angle, reflected light is completely p-polarized")


def calculate_reflectance(angle_deg, n1, n2):
    """Calculate Fresnel reflectance."""
    optics = GeometricOptics()
    angle_rad = np.radians(angle_deg)
    
    reflectance = optics.fresnel_reflectance(angle_rad, n1, n2)
    transmittance = 1 - reflectance
    
    print(f"✅ Fresnel Reflectance Calculation:")
    print(f"   Incident medium: n₁ = {n1}")
    print(f"   Refracting medium: n₂ = {n2}")
    print(f"   Incident angle: θ = {angle_deg}°")
    print(f"   Reflectance (R): {reflectance:.4f} ({reflectance*100:.2f}%)")
    print(f"   Transmittance (T): {transmittance:.4f} ({transmittance*100:.2f}%)")


def main():
    """Main calculator interface."""
    
    parser = argparse.ArgumentParser(
        description="Geometric Optics Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s snell 30 1.0 1.5           # Snell's law: 30° from air to glass
  %(prog)s critical 1.5 1.0           # Critical angle: glass to air  
  %(prog)s brewster 1.0 1.5           # Brewster's angle: air to glass
  %(prog)s reflectance 45 1.0 1.5     # Fresnel reflectance at 45°
  
Common refractive indices:
  Vacuum/Air: 1.0
  Water: 1.33
  Glass (typical): 1.5
  Diamond: 2.42
        """
    )
    
    subparsers = parser.add_subparsers(dest='calculation', help='Type of calculation')
    
    # Snell's law parser
    snell_parser = subparsers.add_parser('snell', help='Calculate refraction using Snell\'s law')
    snell_parser.add_argument('angle', type=float, help='Incident angle (degrees)')
    snell_parser.add_argument('n1', type=float, help='Refractive index of incident medium')
    snell_parser.add_argument('n2', type=float, help='Refractive index of refracting medium')
    
    # Critical angle parser
    critical_parser = subparsers.add_parser('critical', help='Calculate critical angle')
    critical_parser.add_argument('n1', type=float, help='Refractive index of denser medium')
    critical_parser.add_argument('n2', type=float, help='Refractive index of rarer medium')
    
    # Brewster's angle parser
    brewster_parser = subparsers.add_parser('brewster', help='Calculate Brewster\'s angle')
    brewster_parser.add_argument('n1', type=float, help='Refractive index of incident medium')
    brewster_parser.add_argument('n2', type=float, help='Refractive index of refracting medium')
    
    # Reflectance parser
    reflectance_parser = subparsers.add_parser('reflectance', help='Calculate Fresnel reflectance')
    reflectance_parser.add_argument('angle', type=float, help='Incident angle (degrees)')
    reflectance_parser.add_argument('n1', type=float, help='Refractive index of incident medium')
    reflectance_parser.add_argument('n2', type=float, help='Refractive index of refracting medium')
    
    args = parser.parse_args()
    
    if args.calculation is None:
        parser.print_help()
        return
    
    print("🧮 Geometric Optics Calculator")
    print("=" * 35)
    
    try:
        if args.calculation == 'snell':
            calculate_snells_law(args.angle, args.n1, args.n2)
        elif args.calculation == 'critical':
            calculate_critical_angle(args.n1, args.n2)
        elif args.calculation == 'brewster':
            calculate_brewster_angle(args.n1, args.n2)
        elif args.calculation == 'reflectance':
            calculate_reflectance(args.angle, args.n1, args.n2)
            
    except Exception as e:
        print(f"❌ Calculation error: {e}")
        return
    
    print(f"\n📚 For more detailed analysis, run: python geometric_optics.py")


if __name__ == "__main__":
    main()