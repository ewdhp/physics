#!/usr/bin/env python3
"""
PENDULUM EQUATION OF MOTION - Complete Mathematical Analysis
============================================================

This script demonstrates the complete mathematical derivation and solution
of the pendulum equation of motion, including both nonlinear and linearized cases.

Mathematical Theory:
-------------------

1. LAGRANGIAN MECHANICS APPROACH:
   - Kinetic Energy: T = (1/2) * m * L² * θ̇²
   - Potential Energy: V = m * g * L * (1 - cos θ)
   - Lagrangian: L = T - V = (1/2) * m * L² * θ̇² - m * g * L * (1 - cos θ)
   
2. EULER-LAGRANGE EQUATION:
   d/dt(∂L/∂θ̇) - ∂L/∂θ = 0
   
   Leads to: θ̈ + (g/L) * sin θ = 0  (Nonlinear equation)

3. LINEARIZATION (Small angle approximation):
   For small θ: sin θ ≈ θ
   Therefore: θ̈ + (g/L) * θ = 0  (Simple harmonic oscillator)
   
   Solution: θ(t) = A * cos(ωt + φ), where ω = √(g/L)

4. ENERGY CONSERVATION:
   Total Energy: E = (1/2) * m * L² * θ̇² + m * g * L * (1 - cos θ)
   For conservative system: E = constant

5. PERIOD FORMULAS:
   - Linear approximation: T₀ = 2π * √(L/g)
   - Nonlinear correction: T ≈ T₀ * [1 + (1/16) * θ₀² + (11/3072) * θ₀⁴ + ...]
   
Author: Physics Education Project
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import math

class PendulumAnalysis:
    """
    Complete pendulum analysis with mathematical theory
    """
    
    def __init__(self, L=1.0, g=9.81, mass=1.0):
        """
        Initialize pendulum parameters
        
        Args:
            L: Length of pendulum (m)
            g: Gravitational acceleration (m/s²)
            mass: Mass of bob (kg)
        """
        self.L = L
        self.g = g
        self.mass = mass
        self.omega0 = np.sqrt(g/L)  # Natural frequency for small oscillations
        
        # Mathematical constants for period correction
        self.period_linear = 2 * np.pi * np.sqrt(L/g)
    
    def nonlinear_equation(self, t, y):
        """
        NONLINEAR EQUATION OF MOTION: θ̈ + (g/L) * sin θ = 0
        
        Mathematical Derivation (Lagrangian Mechanics):
        ----------------------------------------------
        1. Position of bob: x = L sin θ, y = -L cos θ
        2. Velocity: ẋ = L θ̇ cos θ, ẏ = L θ̇ sin θ  
        3. Kinetic Energy: T = (1/2) m (ẋ² + ẏ²) = (1/2) m L² θ̇²
        4. Potential Energy: V = m g y = -m g L cos θ (taking y=0 at pivot)
           Or equivalently: V = m g L (1 - cos θ) taking V=0 at bottom
        5. Lagrangian: L = T - V = (1/2) m L² θ̇² - m g L (1 - cos θ)
        
        Euler-Lagrange Equation: d/dt(∂L/∂θ̇) - ∂L/∂θ = 0
        
        ∂L/∂θ̇ = m L² θ̇
        d/dt(∂L/∂θ̇) = m L² θ̈
        ∂L/∂θ = -m g L sin θ
        
        Therefore: m L² θ̈ - (-m g L sin θ) = 0
                  m L² θ̈ + m g L sin θ = 0
                  θ̈ + (g/L) sin θ = 0  ← NONLINEAR EQUATION OF MOTION
        
        For numerical integration, convert to first-order system:
        Let y[0] = θ, y[1] = θ̇
        Then: dy[0]/dt = y[1]
              dy[1]/dt = -(g/L) * sin(y[0])
        """
        theta, theta_dot = y
        theta_ddot = -(self.g / self.L) * np.sin(theta)
        return [theta_dot, theta_ddot]
    
    def linear_equation(self, t, y):
        """
        LINEARIZED EQUATION OF MOTION: θ̈ + (g/L) * θ = 0
        
        Mathematical Approximation (Small Angle):
        ----------------------------------------
        For small angles θ (in radians): sin θ ≈ θ - θ³/6 + θ⁵/120 - ...
        
        First-order approximation: sin θ ≈ θ
        
        Therefore: θ̈ + (g/L) θ = 0  ← LINEAR SIMPLE HARMONIC OSCILLATOR
        
        Analytical Solution:
        θ(t) = A cos(ω₀t + φ), where ω₀ = √(g/L)
        
        With initial conditions θ(0) = θ₀, θ̇(0) = 0:
        θ(t) = θ₀ cos(ω₀t)
        
        Period: T = 2π/ω₀ = 2π√(L/g)  ← INDEPENDENT OF AMPLITUDE!
        """
        theta, theta_dot = y
        theta_ddot = -(self.g / self.L) * theta  # Linear approximation
        return [theta_dot, theta_ddot]
    
    def analytical_solution(self, t, theta0, theta_dot0=0):
        """
        ANALYTICAL SOLUTION for linear case
        
        General solution: θ(t) = A cos(ω₀t) + B sin(ω₀t)
        
        With initial conditions:
        θ(0) = θ₀  →  A = θ₀
        θ̇(0) = θ̇₀  →  B = θ̇₀/ω₀
        
        Therefore: θ(t) = θ₀ cos(ω₀t) + (θ̇₀/ω₀) sin(ω₀t)
        """
        omega0 = self.omega0
        return theta0 * np.cos(omega0 * t) + (theta_dot0 / omega0) * np.sin(omega0 * t)
    
    def energy(self, theta, theta_dot):
        """
        TOTAL MECHANICAL ENERGY (Conservative System)
        
        Energy Components:
        -----------------
        1. Kinetic Energy: T = (1/2) m L² θ̇²
        2. Potential Energy: V = m g L (1 - cos θ)
           - At θ = 0 (bottom): V = 0
           - At θ = π/2 (horizontal): V = m g L
           - At θ = π (top): V = 2 m g L
        
        Total Energy: E = T + V = (1/2) m L² θ̇² + m g L (1 - cos θ)
        
        For conservative system: dE/dt = 0 (Energy is conserved)
        """
        kinetic = 0.5 * self.mass * self.L**2 * theta_dot**2
        potential = self.mass * self.g * self.L * (1 - np.cos(theta))
        return kinetic + potential
    
    def period_correction(self, theta0):
        """
        NONLINEAR PERIOD CORRECTION
        
        Exact Period (Elliptic Integral):
        T = 4 * √(L/g) * K(k), where k = sin(θ₀/2)
        K(k) is the complete elliptic integral of the first kind
        
        Series Expansion for moderate amplitudes:
        T ≈ T₀ * [1 + (1/16)θ₀² + (11/3072)θ₀⁴ + (173/737280)θ₀⁶ + ...]
        
        Where T₀ = 2π√(L/g) is the linear period
        
        Physical Interpretation:
        - For θ₀ = 15° ≈ 0.26 rad: T/T₀ ≈ 1.004 (0.4% increase)
        - For θ₀ = 45° ≈ 0.79 rad: T/T₀ ≈ 1.073 (7.3% increase)  
        - For θ₀ = 90° ≈ 1.57 rad: T/T₀ ≈ 1.181 (18.1% increase)
        """
        # Series expansion coefficients
        c1 = 1/16
        c2 = 11/3072
        c3 = 173/737280
        
        correction_factor = 1 + c1*theta0**2 + c2*theta0**4 + c3*theta0**6
        return self.period_linear * correction_factor


def plot_pendulum_analysis():
    """
    Create comprehensive plots showing pendulum equation of motion analysis
    """
    pendulum = PendulumAnalysis(L=1.0, g=9.81, mass=1.0)
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Time parameters
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)
    
    # Different initial conditions for comparison
    initial_angles = [np.pi/6, np.pi/4, np.pi/3, np.pi/2]  # 30°, 45°, 60°, 90°
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot 1: Nonlinear vs Linear Solutions
    ax1 = plt.subplot(2, 3, 1)
    
    for i, (theta0, color) in enumerate(zip(initial_angles, colors)):
        # Solve nonlinear equation
        sol_nonlinear = solve_ivp(pendulum.nonlinear_equation, t_span, [theta0, 0], 
                                 t_eval=t_eval, rtol=1e-8)
        
        # Solve linear equation
        sol_linear = solve_ivp(pendulum.linear_equation, t_span, [theta0, 0], 
                              t_eval=t_eval, rtol=1e-8)
        
        # Analytical solution
        theta_analytical = pendulum.analytical_solution(t_eval, theta0)
        
        if i == 0:  # Only show labels for first curve
            ax1.plot(t_eval, np.degrees(sol_nonlinear.y[0]), color=color, 
                    linewidth=2, label='Nonlinear')
            ax1.plot(t_eval, np.degrees(sol_linear.y[0]), '--', color=color, 
                    linewidth=2, alpha=0.7, label='Linear')
            ax1.plot(t_eval, np.degrees(theta_analytical), ':', color=color, 
                    linewidth=1, alpha=0.5, label='Analytical')
        else:
            ax1.plot(t_eval, np.degrees(sol_nonlinear.y[0]), color=color, linewidth=2)
            ax1.plot(t_eval, np.degrees(sol_linear.y[0]), '--', color=color, 
                    linewidth=2, alpha=0.7)
            ax1.plot(t_eval, np.degrees(theta_analytical), ':', color=color, 
                    linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Angle (degrees)', fontsize=11)
    ax1.set_title('1. Nonlinear vs Linear Solutions\nθ̈ + (g/L)sin θ = 0 vs θ̈ + (g/L)θ = 0', 
                  fontsize=11, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Phase Portrait
    ax2 = plt.subplot(2, 3, 2)
    
    for theta0, color in zip(initial_angles, colors):
        sol = solve_ivp(pendulum.nonlinear_equation, t_span, [theta0, 0], 
                       t_eval=t_eval, rtol=1e-8)
        ax2.plot(np.degrees(sol.y[0]), np.degrees(sol.y[1]), color=color, 
                linewidth=2, label=f'θ₀ = {np.degrees(theta0):.0f}°')
    
    ax2.set_xlabel('Angle θ (degrees)', fontsize=11)
    ax2.set_ylabel('Angular Velocity θ̇ (deg/s)', fontsize=11)
    ax2.set_title('2. Phase Portrait\nθ̇ vs θ trajectories', fontsize=11, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Energy Conservation
    ax3 = plt.subplot(2, 3, 3)
    
    theta0 = np.pi/4  # 45 degrees
    sol = solve_ivp(pendulum.nonlinear_equation, t_span, [theta0, 0], 
                   t_eval=t_eval, rtol=1e-8)
    
    # Calculate energy components
    kinetic_energy = 0.5 * pendulum.mass * pendulum.L**2 * sol.y[1]**2
    potential_energy = pendulum.mass * pendulum.g * pendulum.L * (1 - np.cos(sol.y[0]))
    total_energy = kinetic_energy + potential_energy
    
    ax3.plot(t_eval, kinetic_energy, 'r-', linewidth=2, label='Kinetic: ½mL²θ̇²')
    ax3.plot(t_eval, potential_energy, 'b-', linewidth=2, label='Potential: mgL(1-cos θ)')
    ax3.plot(t_eval, total_energy, 'k--', linewidth=2, label='Total Energy')
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Energy (J)', fontsize=11)
    ax3.set_title('3. Energy Conservation\nE = T + V = constant', fontsize=11, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Period vs Amplitude
    ax4 = plt.subplot(2, 3, 4)
    
    angles_deg = np.linspace(1, 90, 50)
    angles_rad = np.radians(angles_deg)
    
    periods_theoretical = []
    periods_numerical = []
    
    for theta0 in angles_rad:
        # Theoretical period correction
        period_theory = pendulum.period_correction(theta0)
        periods_theoretical.append(period_theory)
        
        # Numerical period (measure time for one complete oscillation)
        sol = solve_ivp(pendulum.nonlinear_equation, (0, 20), [theta0, 0], 
                       dense_output=True, rtol=1e-10)
        
        # Find period by detecting zero crossings
        t_fine = np.linspace(0, 20, 10000)
        theta_fine = sol.sol(t_fine)[0]
        
        # Find first return to initial angle (approximately)
        crossings = []
        for i in range(1, len(theta_fine)):
            if theta_fine[i-1] * theta_fine[i] < 0:  # Sign change
                crossings.append(t_fine[i])
        
        if len(crossings) >= 2:
            period_numerical = 2 * crossings[1]  # Half period * 2
            periods_numerical.append(period_numerical)
        else:
            periods_numerical.append(np.nan)
    
    # Linear period (constant)
    linear_period = np.full_like(angles_deg, pendulum.period_linear)
    
    ax4.plot(angles_deg, linear_period, 'k--', linewidth=2, alpha=0.7, 
            label=f'Linear: T₀ = 2π√(L/g) = {pendulum.period_linear:.3f}s')
    ax4.plot(angles_deg, periods_theoretical, 'r-', linewidth=2, 
            label='Nonlinear (Theory)')
    ax4.plot(angles_deg[:len(periods_numerical)], periods_numerical, 'bo', 
            markersize=4, alpha=0.7, label='Nonlinear (Numerical)')
    
    ax4.set_xlabel('Initial Amplitude θ₀ (degrees)', fontsize=11)
    ax4.set_ylabel('Period T (s)', fontsize=11)
    ax4.set_title('4. Period vs Amplitude\nT ≈ T₀[1 + θ₀²/16 + ...]', fontsize=11, weight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Angular Acceleration
    ax5 = plt.subplot(2, 3, 5)
    
    theta0 = np.pi/3  # 60 degrees
    sol = solve_ivp(pendulum.nonlinear_equation, (0, 5), [theta0, 0], 
                   t_eval=np.linspace(0, 5, 500), rtol=1e-8)
    
    # Calculate angular acceleration
    alpha_nonlinear = -(pendulum.g/pendulum.L) * np.sin(sol.y[0])
    alpha_linear = -(pendulum.g/pendulum.L) * sol.y[0]
    
    ax5.plot(t_eval[:len(alpha_nonlinear)], alpha_nonlinear, 'r-', linewidth=2, 
            label='Nonlinear: α = -(g/L)sin θ')
    ax5.plot(t_eval[:len(alpha_linear)], alpha_linear, 'b--', linewidth=2, 
            label='Linear: α = -(g/L)θ')
    
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Angular Acceleration α (rad/s²)', fontsize=11)
    ax5.set_title('5. Angular Acceleration\nα = θ̈', fontsize=11, weight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Mathematical Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = """
🔬 PENDULUM EQUATION OF MOTION SUMMARY

📐 LAGRANGIAN DERIVATION:
   T = ½mL²θ̇²  (Kinetic Energy)
   V = mgL(1-cos θ)  (Potential Energy)
   L = T - V  (Lagrangian)
   
   Euler-Lagrange: d/dt(∂L/∂θ̇) - ∂L/∂θ = 0
   
📊 EQUATIONS OF MOTION:
   Nonlinear: θ̈ + (g/L)sin θ = 0
   Linear:    θ̈ + (g/L)θ = 0
   
⚡ SOLUTIONS:
   Linear: θ(t) = θ₀cos(ω₀t), ω₀ = √(g/L)
   Period: T₀ = 2π√(L/g)
   
🎯 NONLINEAR EFFECTS:
   • Period increases with amplitude
   • T ≈ T₀[1 + θ₀²/16 + 11θ₀⁴/3072 + ...]
   • Energy conservation: E = T + V = constant
   • Phase space trajectories are closed curves
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    plt.show()


def demonstrate_pendulum_motion():
    """
    Main demonstration function with detailed mathematical explanations
    """
    print("🎯 PENDULUM EQUATION OF MOTION - Mathematical Analysis")
    print("="*60)
    print("This demonstration shows the complete derivation and solution")
    print("of the pendulum equation of motion using Lagrangian mechanics.")
    print()
    
    pendulum = PendulumAnalysis()
    
    print("📐 MATHEMATICAL DERIVATION:")
    print("-" * 30)
    print("1. Position: x = L sin θ, y = -L cos θ")
    print("2. Kinetic Energy: T = ½mL²θ̇²")
    print("3. Potential Energy: V = mgL(1 - cos θ)")
    print("4. Lagrangian: L = T - V")
    print("5. Euler-Lagrange → θ̈ + (g/L)sin θ = 0")
    print()
    
    print("🔬 EXAMPLE CALCULATIONS:")
    print("=" * 40)
    
    # Parameters
    L, g = 1.0, 9.81
    theta0_deg = 45
    theta0_rad = np.radians(theta0_deg)
    
    print(f"Pendulum Length: L = {L:.1f} m")
    print(f"Initial Angle: θ₀ = {theta0_deg}° = {theta0_rad:.3f} rad")
    print()
    
    # Linear analysis
    omega0 = np.sqrt(g/L)
    period_linear = 2 * np.pi / omega0
    
    print(f"LINEAR ANALYSIS (sin θ ≈ θ):")
    print(f"  • Natural Frequency: ω₀ = √(g/L) = {omega0:.3f} rad/s")
    print(f"  • Period: T₀ = 2π/ω₀ = {period_linear:.3f} s")
    print(f"  • Frequency: f₀ = 1/T₀ = {1/period_linear:.3f} Hz")
    print()
    
    # Nonlinear correction
    period_nonlinear = pendulum.period_correction(theta0_rad)
    error_percent = ((period_nonlinear - period_linear) / period_linear) * 100
    
    print(f"NONLINEAR CORRECTION:")
    print(f"  • Corrected Period: T = {period_nonlinear:.3f} s")
    print(f"  • Period Increase: {error_percent:.2f}%")
    print(f"  • Formula: T ≈ T₀[1 + θ₀²/16 + 11θ₀⁴/3072 + ...]")
    print()
    
    # Energy analysis
    initial_energy = pendulum.energy(theta0_rad, 0)
    max_ke = initial_energy  # All potential converts to kinetic at bottom
    max_speed = np.sqrt(2 * max_ke / (pendulum.mass * pendulum.L**2))
    
    print(f"ENERGY ANALYSIS:")
    print(f"  • Total Energy: E = {initial_energy:.3f} J")
    print(f"  • Maximum Speed: v_max = {max_speed:.3f} m/s")
    print(f"  • At θ = 0: KE = {initial_energy:.3f} J, PE = 0 J")
    print(f"  • At θ = θ₀: KE = 0 J, PE = {initial_energy:.3f} J")
    print()
    
    print("📈 Generating pendulum analysis plots...")
    plot_pendulum_analysis()
    
    print("\n🎓 KEY PHYSICS INSIGHTS:")
    print("=" * 40)
    print("• Nonlinear equation: θ̈ + (g/L)sin θ = 0")
    print("• Linear approximation valid for small amplitudes (< 15°)")
    print("• Period increases with amplitude (nonlinear effect)")
    print("• Energy is conserved in ideal pendulum")
    print("• Phase space trajectories are closed ellipses")
    print("• Lagrangian mechanics provides elegant derivation")


if __name__ == "__main__":
    print("🌟 Welcome to Pendulum Equation of Motion Explorer! 🌟")
    print("This script demonstrates the complete mathematical analysis:")
    print("• Lagrangian derivation of equation of motion")
    print("• Nonlinear vs linear solutions")
    print("• Energy conservation and phase portraits")
    print("• Period corrections for large amplitudes")
    
    try:
        demonstrate_pendulum_motion()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    
    print(f"\n✨ Thanks for exploring pendulum mechanics! ✨")