# Wave Optics

This directory contains comprehensive implementations of wave optics phenomena, covering all the key topics mentioned in the main optics README.

## 📂 Available Scripts

### 1. **`interference_phenomena.py`** - Interference Phenomena
- **Two-source interference** with adjustable separation and phase
- **Thin film interference** (soap bubbles, oil films, anti-reflection coatings)
- **Newton's rings** pattern and analysis
- **Michelson interferometer** principles and applications
- **Temporal and spatial coherence** effects on interference visibility
- **Interactive visualizations** of interference patterns

**Key Physics:**
- Superposition principle and path difference calculations
- Constructive/destructive interference conditions
- Phase relationships and optical path differences
- Coherence requirements for high-contrast fringes

### 2. **`double_slit_experiment.py`** - Young's Double-Slit Experiment
- **Classic double-slit setup** with variable parameters
- **Parameter effects** (slit separation, width, wavelength, screen distance)
- **Near-field vs far-field** diffraction patterns (Fresnel vs Fraunhofer)
- **Coherence requirements** for interference visibility
- **Modern applications** in precision measurement and spectroscopy
- **Educational demonstrations** of wave-particle duality

**Key Physics:**
- Fringe spacing: Δy = λL/d
- Single-slit envelope modulation
- Fresnel number analysis: F = d²/(λL)
- Coherence criteria for fringe visibility

### 3. **`diffraction_patterns.py`** - Diffraction Patterns
- **Single-slit diffraction** (Fraunhofer and Fresnel patterns)
- **Circular aperture diffraction** (Airy disk pattern)
- **Diffraction gratings** (spectroscopy applications)
- **Multiple-wavelength dispersion** analysis
- **Fresnel zones** and zone plate theory
- **Resolution limits** and optical system design

**Key Physics:**
- Single-slit: I ∝ (sin β/β)² where β = ka sin θ/2
- Airy disk: I ∝ (2J₁(x)/x)² with resolution θ = 1.22λ/D
- Grating equation: d sin θ = mλ
- Spectral resolution: R = mN (order × number of lines)

### 4. **`polarization.py`** - Polarization
- **Malus's law** demonstration with linear polarizers
- **Brewster's angle** and polarization by reflection
- **Wave plates** (quarter-wave, half-wave) and circular polarization
- **Optical activity** and polarization rotation in chiral media
- **LCD display principles** and polarization control
- **Elliptical and circular polarization** analysis

**Key Physics:**
- Malus's law: I = I₀ cos²(θ)
- Brewster's angle: θB = arctan(n₂/n₁)
- Wave plate retardation effects on polarization states
- Stokes parameters and polarization ellipse

### 5. **`coherence.py`** - Coherence
- **Temporal coherence** and coherence length measurement
- **Spatial coherence** and van Cittert-Zernike theorem
- **Stellar interferometry** applications and angular resolution
- **Coherence measurement techniques** (Michelson interferometer)
- **OCT (Optical Coherence Tomography)** principles
- **Light source comparison** (laser, LED, thermal sources)

**Key Physics:**
- Temporal coherence: Lc = λ²/Δλ
- Spatial coherence: lc ≈ λD/d (van Cittert-Zernike)
- Fringe visibility vs path difference
- Angular resolution: θ = λ/B (interferometer baseline)

## 🚀 Quick Start

```bash
# Activate the virtual environment (if not already active)
source /home/aoru/github/ewdhp/.venv/bin/activate

# Navigate to wave optics directory
cd python/physics/optics/wave

# Test all modules
python test_wave_optics.py

# Run individual demonstrations
python interference_phenomena.py
python double_slit_experiment.py
python diffraction_patterns.py
python polarization.py
python coherence.py
```

## 🎓 Educational Features

All scripts include:
- **Interactive demonstrations** with matplotlib visualizations
- **Parameter exploration** to understand physical relationships
- **Real-world applications** and practical examples
- **Quantitative analysis** with actual calculations
- **Professional-quality plots** with proper labels and annotations
- **Step-by-step explanations** of the underlying physics

## 🔬 Applications Covered

### Scientific Applications
- **Precision metrology** and interferometric measurements
- **Spectroscopy** and wavelength analysis
- **Microscopy** and resolution enhancement
- **Astronomical observations** (stellar interferometry)
- **Medical imaging** (OCT, polarization-sensitive techniques)

### Industrial Applications
- **Optical system design** and testing
- **Anti-reflection coatings** and thin film technology
- **LCD displays** and polarization optics
- **Laser systems** and beam quality analysis
- **Fiber optics** and telecommunications

## 📊 Technical Specifications

### Computational Features
- **NumPy-based** calculations for high performance
- **Matplotlib** visualizations with TkAgg backend support
- **Modular design** for easy extension and modification
- **Error handling** and robust parameter validation
- **Cross-platform compatibility** (Linux, macOS, Windows)

### Physics Accuracy
- **Rigorous mathematical models** based on Maxwell's equations
- **Proper units and scaling** for realistic simulations
- **Multiple approximation levels** (paraxial, small-angle, exact)
- **Experimental validation** against known results
- **Literature-based parameters** for real materials and systems

## 🛠️ Dependencies

All scripts use the same dependencies as the main physics package:
- `numpy` - Numerical calculations
- `matplotlib` - Plotting and visualization
- `scipy` (where needed) - Special functions (Bessel, Fresnel integrals)

## 📚 Further Reading

### Recommended Textbooks
- Hecht, E. "Optics" (5th Edition)
- Born, M. & Wolf, E. "Principles of Optics"
- Saleh, B.E.A. & Teich, M.C. "Fundamentals of Photonics"
- Goodman, J.W. "Introduction to Fourier Optics"

### Key Concepts Covered
1. **Wave Nature of Light** - Electromagnetic wave theory and propagation
2. **Superposition Principle** - Linear combination of wave amplitudes
3. **Interference** - Constructive and destructive wave combination
4. **Diffraction** - Wave behavior at apertures and obstacles
5. **Polarization** - Vector nature of electromagnetic fields
6. **Coherence** - Temporal and spatial correlation properties

## 🎯 Learning Objectives

After working through these modules, you should understand:
- How wave properties of light lead to interference and diffraction
- The relationship between coherence and interference visibility
- How polarization affects light propagation and detection
- Practical applications in modern optical systems
- Quantitative analysis techniques for wave optics phenomena
- Design principles for optical instruments and systems

---

*Part of the Physics Education Project - Comprehensive Python implementations for optics education and research.*