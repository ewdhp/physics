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

### Available Scripts

1. **`geometric_optics.py`** - Comprehensive geometric optics module
   - Snell's law calculations
   - Fresnel reflectance equations
   - Total internal reflection
   - Critical angle calculations
   - Ray tracing and visualization
   - Interactive demonstrations

2. **`simple_example.py`** - Quick start example
   - Basic reflection and refraction calculations
   - Simple ray diagram generation
   - Educational demonstrations

3. **`optics_calculator.py`** - Command-line calculator
   - Quick Snell's law calculations
   - Critical angle calculations  
   - Brewster's angle calculations
   - Fresnel reflectance calculations

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

# Run the comprehensive demo
python geometric_optics.py

# Or run the simple example
python simple_example.py

# Use the command-line calculator for quick calculations
python optics_calculator.py snell 30 1.0 1.5      # Snell's law
python optics_calculator.py critical 1.5 1.0       # Critical angle
python optics_calculator.py brewster 1.0 1.5       # Brewster's angle
python optics_calculator.py reflectance 45 1.0 1.5 # Reflectance

# Configure and test matplotlib backend (from main python directory)
cd .. && python backend_config.py                  # Auto-detect best backend
cd .. && python backend_config.py TkAgg            # Force TkAgg backend
```

### Features

- **Interactive Visualizations**: Ray diagrams with reflection and refraction
- **Physical Accuracy**: Based on Fresnel equations and Snell's law
- **Educational Focus**: Clear explanations and step-by-step calculations
- **Multiple Scenarios**: Normal incidence, oblique incidence, total internal reflection
- **Polarization Effects**: Demonstrates Brewster's angle and polarization
- **GUI Support**: Interactive plot windows with TkAgg backend on OpenSUSE
- **Cross-Platform**: Automatic backend detection and configuration