# Optics

Optics is the study of light and its interactions with matter.

## Key Topics

### Geometric Optics
- Reflection and refraction
- Snell's law
- Mirrors and lenses
- Optical instruments
- Ray tracing

### Wave Optics
- Interference phenomena
- Young's double-slit experiment
- Diffraction patterns
- Polarization
- Coherence

### Physical Optics
- Electromagnetic nature of light
- Dispersion and color
- Scattering
- Absorption and emission
- Laser physics

### Optical Instruments
- Microscopes and telescopes
- Cameras and projectors
- Spectroscopes
- Interferometers
- Fiber optics

## Fundamental Equations

- Snell's law: **n₁sin(θ₁) = n₂sin(θ₂)**
- Lens equation: **1/f = 1/d₀ + 1/dᵢ**
- Diffraction grating: **d sin(θ) = mλ**
- Malus's law: **I = I₀cos²(θ)**

## Applications

- Optical communication
- Medical diagnostics and surgery
- Astronomy and telescopy
- Photography and imaging
- Holography and displays

## 🐍 Python Implementations

The optics module is organized into two main subdirectories:

### 📂 Directory Structure

```
optics/
├── geometric/          # Geometric optics (ray optics)
│   ├── reflection_refraction.py
│   ├── mirrors_lenses.py
│   ├── optical_instruments.py
│   ├── ray_tracing.py
│   ├── optics_calculator.py
│   └── simple_example.py
└── wave/              # Wave optics
    ├── interference_phenomena.py
    ├── double_slit_experiment.py
    ├── diffraction_patterns.py
    ├── polarization.py
    ├── coherence.py
    └── test_wave_optics.py
```

### 🔍 Geometric Optics Scripts

1. **`reflection_refraction.py`** - Comprehensive geometric optics module
   - Snell's law calculations
   - Fresnel reflectance equations
   - Total internal reflection
   - Critical angle calculations
   - Ray tracing and visualization
   - Interactive demonstrations

2. **`mirrors_lenses.py`** - Complete mirrors and lenses implementation
   - Lens equation (1/f = 1/do + 1/di) with ray tracing
   - Mirror equation and curved mirror analysis
   - Converging and diverging lens simulations
   - OpticalElement classes with full ray tracing capabilities

3. **`optical_instruments.py`** - Optical instruments simulation
   - Telescopes (refracting and reflecting) with magnification calculations
   - Compound microscopes with resolution analysis
   - Camera systems and field of view calculations
   - Fiber optics principles and numerical aperture analysis

4. **`ray_tracing.py`** - Advanced ray tracing system
   - Professional-grade sequential ray tracing through multiple surfaces
   - Multi-element optical system analysis and design
   - Spot diagram generation for aberration analysis
   - Spherical surface intersection algorithms and complex system design

5. **`optics_calculator.py`** - Command-line calculator
   - Quick Snell's law calculations
   - Critical angle calculations  
   - Brewster's angle calculations
   - Fresnel reflectance calculations

6. **`simple_example.py`** - Quick start example
   - Basic reflection and refraction calculations
   - Simple ray diagram generation
   - Educational demonstrations

### 🌊 Wave Optics Scripts

1. **`interference_phenomena.py`** - Interference phenomena
   - Two-source interference with adjustable parameters
   - Thin film interference (soap bubbles, oil films, AR coatings)
   - Newton's rings pattern analysis
   - Michelson interferometer principles and applications
   - Temporal and spatial coherence effects

2. **`double_slit_experiment.py`** - Young's double-slit experiment
   - Classic double-slit setup with variable parameters
   - Parameter effects (slit separation, width, wavelength, distance)
   - Near-field vs far-field diffraction patterns
   - Coherence requirements and modern applications

3. **`diffraction_patterns.py`** - Diffraction patterns
   - Single-slit diffraction (Fraunhofer and Fresnel)
   - Circular aperture diffraction (Airy disk pattern)
   - Diffraction gratings and spectroscopy applications
   - Multiple-wavelength dispersion analysis
   - Fresnel zones and resolution limits

4. **`polarization.py`** - Polarization phenomena
   - Malus's law demonstration with linear polarizers
   - Brewster's angle and polarization by reflection
   - Wave plates (quarter-wave, half-wave) and circular polarization
   - Optical activity and polarization rotation
   - LCD display principles and polarization control

5. **`coherence.py`** - Coherence properties
   - Temporal coherence and coherence length measurement
   - Spatial coherence and van Cittert-Zernike theorem
   - Stellar interferometry applications and angular resolution
   - OCT (Optical Coherence Tomography) principles
   - Light source comparison and coherence measurement techniques

### Global Configuration Files (in `/python/`)

4. **`../backend_config.py`** - Matplotlib backend configuration
   - Automatic backend detection and setup for all physics modules
   - Test matplotlib display capabilities
   - Configure optimal settings for your system

5. **`../matplotlibrc`** - Matplotlib configuration file
   - Optimized settings for all physics plots
   - Better default colors and styling
   - Enhanced readability

### Quick Start

```bash
# Activate the virtual environment
source /home/aoru/github/ewdhp/.venv/bin/activate

# Navigate to optics directory
cd physics/optics

# GEOMETRIC OPTICS
cd geometric/

# Run comprehensive geometric optics demos
python reflection_refraction.py               # Reflection, refraction, Snell's law
python mirrors_lenses.py                      # Mirrors and lenses analysis  
python optical_instruments.py                 # Telescopes, microscopes, cameras
python ray_tracing.py                         # Advanced ray tracing systems

# Quick calculations
python optics_calculator.py snell 30 1.0 1.5      # Snell's law
python optics_calculator.py critical 1.5 1.0       # Critical angle
python optics_calculator.py brewster 1.0 1.5       # Brewster's angle
python optics_calculator.py reflectance 45 1.0 1.5 # Reflectance

# Simple introduction
python simple_example.py                      # Basic demonstrations

# WAVE OPTICS  
cd ../wave/

# Test all wave optics modules
python test_wave_optics.py

# Run comprehensive wave optics demos
python interference_phenomena.py              # Interference patterns and coherence
python double_slit_experiment.py             # Young's double-slit experiment
python diffraction_patterns.py               # Single-slit, Airy disk, gratings
python polarization.py                        # Polarization and wave plates
python coherence.py                          # Temporal and spatial coherence

# Configure and test matplotlib backend (from main python directory)
cd ../../ && python backend_config.py                  # Auto-detect best backend
cd ../../ && python backend_config.py TkAgg            # Force TkAgg backend
```

### Features

#### Geometric Optics
- **Ray Tracing**: Professional-grade sequential ray tracing through complex optical systems
- **Optical Design**: Complete lens and mirror systems with aberration analysis
- **Instruments**: Telescopes, microscopes, cameras with realistic parameters
- **Interactive Visualizations**: Ray diagrams and optical system layouts
- **Physical Accuracy**: Based on Fermat's principle, Snell's law, and Fresnel equations

#### Wave Optics  
- **Interference**: Two-source, thin film, Newton's rings, Michelson interferometry
- **Diffraction**: Single-slit, Airy disk, gratings with spectroscopic applications
- **Polarization**: Linear, circular, elliptical polarization with wave plates
- **Coherence**: Temporal and spatial coherence measurement and applications
- **Modern Applications**: OCT, stellar interferometry, precision metrology

#### Technical Features
- **Educational Focus**: Clear explanations and step-by-step calculations
- **Multiple Scenarios**: Complete coverage of geometric and wave optics phenomena  
- **GUI Support**: Interactive plot windows with TkAgg backend on OpenSUSE
- **Cross-Platform**: Automatic backend detection and configuration
- **Modular Design**: Organized structure for easy navigation and extension