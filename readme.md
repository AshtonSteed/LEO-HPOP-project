# High Precision Orbital Propagator (HPOP)

![Gravitational Potential](https://latex.codecogs.com/svg.latex?-\frac{GM_{\oplus}}{r}%20+%20\sum_{n=2}^{N_z}%20\frac{J_n%20P_n^{0}(\cos\theta)}{r^{n+1}}%20+%20\sum_{n=2}^{N_t}%20\sum_{m=1}^{n}%20\frac{P_n^{m}(\cos\theta)\,(C_n^m%20\cos(m\phi)%20+%20S_n^m%20\sin(m\phi))}{r^{n+1}})

A Python-based high-precision orbital propagator designed for simulating the trajectories of Low Earth Orbit (LEO) satellites with exceptional accuracy. This project implements advanced numerical integration techniques and accounts for multiple environmental perturbations to predict satellite positions and velocities over time.

## Overview

Orbital propagation involves predicting the future position and velocity of a satellite based on initial conditions and the forces acting upon it. For high-precision applications, such as satellite tracking, rendezvous missions, or space situational awareness, it's crucial to model not just the central gravitational force of Earth but also various perturbations that affect the orbit.

HPOP uses the Runge-Kutta integration method (RK8) to solve the equations of motion, providing accurate trajectory predictions for LEO satellites.

## Key Features

### Numerical Integration
- **RK8 Integration**: Implements the 8th-order Runge-Kutta method for high-accuracy numerical integration of orbital dynamics.
- **State Vector**: Tracks position (r) and velocity (v) in Cartesian coordinates.

### Coordinate Systems
The propagator handles conversions between spherical and Cartesian coordinate systems:

**Spherical to Cartesian:**
```
x = r * cos(θ) * cos(φ)
y = r * cos(θ) * sin(φ)
z = r * sin(θ)
```

**Cartesian to Spherical:**
```
r = √(x² + y² + z²)
θ = arcsin(z/r)
φ = arctan2(y, x)
```

## Perturbations

HPOP incorporates several key perturbations to achieve high-precision orbital propagation:

### 1. Gravitational Harmonics (EGM2008)

The Earth's gravitational field is not perfectly spherical. The EGM2008 model provides a high-resolution representation of Earth's gravity field using spherical harmonics.

**Mathematical Formulation:**
The gravitational potential V is expressed as:

![Gravitational Potential Equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}&space;U=\frac{GM_{\oplus}}{r}%20&plus;%20\sum_{n=2}^{N_z}%20\frac{J_n%20P_n^{0}(\cos\theta)}{r^{n&plus;1}}%20&plus;%20\sum_{n=2}^{N_t}%20\sum_{m=1}^{n}%20\frac{P_n^{m}(\cos\theta)\,(C_n^m%20\cos(m\phi)%20&plus;%20S_n^m%20\sin(m\phi))}{r^{n&plus;1}})

Where:
- \( $GM_{\oplus}$ \): Earth's gravitational parameter (μ ≈ 3.986004418 × 10¹⁴ m³/s²)
- \( $r, \theta, \phi$ \): Spherical coordinates (radius, latitude, longitude)
- \( $P_n^m$ \): Associated Legendre functions
- \($ J_n$ \): Zonal harmonics (axisymmetric perturbations)
- \( $C_n^m, S_n^m$ \): Tesseral and sectoral harmonics (asymmetric perturbations from EGM2008)

**Implementation:**
- Loads EGM2008 coefficients from `data/EGM2008NM1000.csv`
- Computes higher-order harmonics up to degree/order 1000
- Accounts for Earth's oblateness and mass distribution irregularities

### 2. Exospheric Drag

Atmospheric drag becomes significant in LEO, especially at altitudes below 1000 km. The exosphere (upper atmosphere) exerts a drag force opposing the satellite's motion.

**Key Concepts:**
- **Density Model**: Uses atmospheric density models (e.g., NRLMSISE-00) based on altitude, solar activity, and geomagnetic conditions
- **Drag Force**: \( \$vec{F_d} = -\frac{1}{2} C_d A \rho v^2 \hat{v}$ \)
  - \( $C_d$ \): Drag coefficient
  - \( $A$ \): Satellite cross-sectional area
  - \( $\rho$ \): Atmospheric density
  - \( $v$ \): Satellite velocity relative to atmosphere
- **Acceleration**: \( $\vec{a_d} = \frac{\vec{F_d}}{m}$ \)

**Implementation:**
- Calculates atmospheric density based on position and time
- Applies drag acceleration in the velocity direction

### 3. External Body Gravity

Gravitational perturbations from celestial bodies other than Earth, primarily the Moon and Sun, affect LEO satellite orbits.

**Third-Body Perturbation:**
- **Lunar Gravity**: The Moon's gravity causes periodic perturbations, especially during lunar passages
- **Solar Gravity**: The Sun's influence is smaller but still significant for precise calculations

**Mathematical Approach:**
- Computes gravitational acceleration from Moon and Sun positions
- Adds perturbation acceleration: \( $\vec{a_{ext}} = -\mu_{body} \frac{\vec{r_{sat}} - \vec{r_{body}}}{|\vec{r_{sat}} - \vec{r_{body}}|^3}$ \)
  - \( $\mu_{body}$ \): Gravitational parameter of the perturbing body
  - \( $\vec{r_{sat}}, \vec{r_{body}}$ \): Position vectors of satellite and perturbing body

**Implementation:**
- Tracks Moon and Sun positions using ephemeris data
- Calculates and sums gravitational perturbations

### 4. Solar Radiation Pressure (SRP)

Solar radiation exerts pressure on satellite surfaces due to photon momentum transfer. This is particularly important for satellites with large solar panels or high area-to-mass ratios.

**Key Concepts:**
- **Radiation Pressure**: \( $P_{SRP} = \frac{F_{solar}}{c}$ \) where F_solar is solar flux and c is speed of light
- **Force on Satellite**: Depends on surface properties, orientation, and shadowing
- **Acceleration**: \( $\vec{a_{SRP}} = \frac{P_{SRP} A_{eff}}{m} \hat{n}$ \)
  - \( $A_{eff}$ \): Effective cross-sectional area
  - \( $m$ \): Satellite mass
  - \( $\hat{n}$ \): Surface normal direction

**Implementation:**
- Models SRP based on satellite attitude and solar position
- Accounts for eclipses and varying solar distance
- Applies pressure acceleration considering surface reflectivity

## Project Structure

```
LEO-HPOP-project/
├── main.py                 # Main simulation script
├── state.py                # State vector management
├── Gravity/
│   └── gravity.py          # Gravitational force calculations
├── Drag/
│   └── drag.py             # Atmospheric drag calculations
├── Integration and Conversion/
│   ├── butchertab_ex.py    # Butcher tableau for RK integration
│   └── state.py            # Additional state utilities
├── data/
│   └── EGM2008NM1000.csv   # EGM2008 spherical harmonic coefficients
└── readme.md               # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/LEO-HPOP-project.git
   cd LEO-HPOP-project
   ```

2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```

## Usage
 - TBD

## Future Developments

- Implement additional perturbations (e.g., Earth's magnetic field, relativistic effects)
- Add support for different orbit types (MEO, GEO)
- Integrate with real-time tracking data
- Optimize for parallel computing

## References



## License

idk