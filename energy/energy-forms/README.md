# Energy Forms

This section explores the different forms that energy can take across various physics domains. Each form has its own characteristics, mathematical expressions, and physical significance.

## 🎯 Overview

Energy manifests in many forms throughout the physical universe:

- **Mechanical Energy**: Motion and position-based energy
- **Thermal Energy**: Energy associated with temperature and heat
- **Electromagnetic Energy**: Energy in electric and magnetic fields
- **Nuclear Energy**: Energy stored in atomic nuclei
- **Chemical Energy**: Energy in molecular bonds
- **Wave Energy**: Energy carried by oscillating systems

## 🔧 Mechanical Energy

### Kinetic Energy
Energy associated with motion:
$$KE = \frac{1}{2}mv^2$$

**Key Properties:**
- Always positive
- Frame-dependent
- Scales quadratically with velocity

### Potential Energy
Energy associated with position in a force field:

**Gravitational:** $U_g = mgh$ (near Earth's surface)
**Elastic:** $U_e = \frac{1}{2}kx^2$ (Hooke's law)
**General:** $U = -\int \vec{F} \cdot d\vec{r}$ (conservative forces)

## 🌡️ Thermal Energy

### Internal Energy
Microscopic kinetic and potential energy of particles:
$$U = \langle KE_{microscopic} \rangle + \langle PE_{microscopic} \rangle$$

### Heat
Energy transfer due to temperature difference:
$$Q = mc\Delta T$$ (for specific heat)
$$Q = mL$$ (for phase changes)

### Temperature Relation
For ideal gas: $\langle KE \rangle = \frac{3}{2}k_BT$ per particle

## ⚡ Electromagnetic Energy

### Electric Field Energy
Energy density in electric fields:
$$u_E = \frac{1}{2}\epsilon_0 E^2$$

Total energy: $U_E = \int u_E \, dV$

### Magnetic Field Energy
Energy density in magnetic fields:
$$u_B = \frac{1}{2\mu_0} B^2$$

### Electromagnetic Waves
Energy in light and radio waves:
- **Energy density**: $u = u_E + u_B = \epsilon_0 E^2$
- **Intensity**: $I = \langle \vec{S} \rangle = \frac{1}{2\mu_0 c} E_0^2$
- **Photon energy**: $E = h\nu = \hbar\omega$

## ☢️ Nuclear Energy

### Binding Energy
Energy required to separate nucleus into constituent nucleons:
$$BE = (Zm_p + Nm_n - M_{nucleus})c^2$$

### Mass-Energy Equivalence
Einstein's famous relation:
$$E = mc^2$$

### Nuclear Reactions
- **Fission**: Heavy nuclei split, release energy
- **Fusion**: Light nuclei combine, release energy
- **Radioactive decay**: Spontaneous nuclear transformation

## 🌊 Wave Energy

### Mechanical Waves
Energy in sound, seismic waves, water waves:
$$E = \frac{1}{2}\rho A^2 \omega^2$$ (per unit volume)

Where:
- $\rho$ = medium density
- $A$ = amplitude
- $\omega$ = angular frequency

### Energy Transport
Power transmitted by waves:
$$P = -F \cdot v = \vec{S} \cdot \hat{n}$$

## 🧪 Chemical Energy

### Bond Energy
Energy stored in molecular bonds:
- **Formation energy**: Energy to form bonds
- **Dissociation energy**: Energy to break bonds
- **Activation energy**: Energy barrier for reactions

### Thermochemical Relations
$$\Delta H = \sum E_{bonds \, broken} - \sum E_{bonds \, formed}$$

## 📊 Energy Scales

Understanding the typical magnitudes:

| Energy Form | Typical Scale | Example |
|-------------|---------------|---------|
| Nuclear | MeV (10⁶ eV) | Nuclear fission |
| Chemical | eV (1.6×10⁻¹⁹ J) | Molecular bonds |
| Thermal | kT ≈ 0.025 eV | Room temperature |
| Mechanical | Joules | Macroscopic motion |
| Gravitational | Variable | Depends on height/mass |

## 🔗 Interconnections

### Energy Cascades
Large-scale energy often cascades to smaller scales:
- **Stars**: Nuclear → Thermal → Electromagnetic
- **Hydroelectric**: Gravitational → Kinetic → Electrical
- **Photosynthesis**: Electromagnetic → Chemical

### Reversibility
Some transformations are more reversible than others:
- **Highly Reversible**: Mechanical ↔ Electrical
- **Moderately Reversible**: Chemical ↔ Thermal
- **Irreversible**: High-grade → Low-grade thermal

## 🎓 Key Insights

### Unification
Modern physics reveals deep connections:
- **Electromagnetism**: Electric and magnetic energy unified
- **Mass-Energy**: Matter and energy are equivalent
- **Quantum**: Energy comes in discrete packets (quanta)

### Conservation
Despite different forms, total energy is conserved:
- Local conservation (in small regions)
- Global conservation (in isolated systems)
- Apparent violations resolved by including all forms

### Quality
Not all energy forms are equally useful:
- **High-quality**: Mechanical, electrical (fully convertible)
- **Low-quality**: Heat at low temperature (limited conversion)

## 📚 Implementation Files

- `mechanical.py`: Kinetic and potential energy calculations
- `thermal.py`: Heat, temperature, and internal energy
- `electromagnetic.py`: Field energies and electromagnetic waves
- `nuclear.py`: Binding energy and nuclear reactions
- `wave.py`: Wave energy and propagation
- `chemical.py`: Bond energies and reaction energetics

Each file contains both computational tools and educational demonstrations to help you understand these fundamental energy forms! ⚡