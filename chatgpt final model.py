import numpy as np

# Constants
G = 6.67430e-20  # km^3/kg/s^2
M_Earth = 5.972e24  # kg
R_E = 6378.137  # km
J2 = 1.08263e-3
J3 = -2.5327e-6
J4 = -1.6196e-6
M_Sun = 1.989e30  # kg
M_Jupiter = 1.898e27  # kg
AU = 149597870.7  # km (1 AU)
Jupiter_Dist = 778.5e6  # km (Jupiter-Earth average distance)

# Approximate Sun & Jupiter positions (assuming simple circular orbits)
def sun_position(t_days):
    theta = (2 * np.pi * t_days / 365.25) % (2 * np.pi)
    return np.array([AU * np.cos(theta), AU * np.sin(theta), 0])

def jupiter_position(t_days):
    theta = (2 * np.pi * t_days / 4332.6) % (2 * np.pi)  # Jupiter's 12-year orbit
    return np.array([Jupiter_Dist * np.cos(theta), Jupiter_Dist * np.sin(theta), 0])

def j2_acceleration(r):
    r_mag = np.linalg.norm(r)
    z = r[2]

    factor = -1.5 * J2 * (G * M_Earth * R_E**2) / r_mag**5
    j2_x = factor * (1 - 5 * (z**2 / r_mag**2)) * r[0]
    j2_y = factor * (1 - 5 * (z**2 / r_mag**2)) * r[1]
    j2_z = factor * (3 - 5 * (z**2 / r_mag**2)) * z

    return np.array([j2_x, j2_y, j2_z])

# Compute J3 & J4 perturbation acceleration
def j3_j4_acceleration(r):
    r_mag = np.linalg.norm(r)
    z = r[2]

    j3_factor = (-5/2) * J3 * (G * M_Earth * R_E**3) / r_mag**7
    j4_factor = (5/8) * J4 * (G * M_Earth * R_E**4) / r_mag**9

    z_j3 = (7 * z**3 / r_mag**3) - (3 * z / r_mag)
    z_j4 = (35 * z**3 / r_mag**3) - (15 * z / r_mag)

    return j3_factor * (z_j3 * r - 3 * z**2 * np.array([0, 0, 1])) + \
           j4_factor * ((35 * z**4 / r_mag**4) - (30 * z**2 / r_mag**2) + 3) * r - \
           j4_factor * z_j4 * np.array([0, 0, 1])

# Compute tidal acceleration
def tidal_acceleration(r, t_days):
    r_sun = sun_position(t_days)
    r_jupiter = jupiter_position(t_days)

    a_sun = G * M_Sun * ((r_sun - r) / np.linalg.norm(r_sun - r)**3 - r_sun / np.linalg.norm(r_sun)**3)
    a_jupiter = G * M_Jupiter * ((r_jupiter - r) / np.linalg.norm(r_jupiter - r)**3 - r_jupiter / np.linalg.norm(r_jupiter)**3)

    return a_sun + a_jupiter

# Compute total acceleration
def acceleration(r, v, t_days):
    r_mag = np.linalg.norm(r)

    # Primary gravity
    a_gravity = -G * M_Earth * r / r_mag**3

    # J2, J3, J4 perturbations
    a_j2 = j2_acceleration(r)  # <--- J2 IS BACK!
    a_j3_j4 = j3_j4_acceleration(r)

    # Tidal Forces
    a_tidal = tidal_acceleration(r, t_days)

    return a_gravity + a_j2 + a_j3_j4 + a_tidal


# RK8 Solver (same as previous, but now includes J3, J4, and tidal forces)
def rk8_step(r, v, dt, t_days):
    k_r = np.zeros((13, 3))
    k_v = np.zeros((13, 3))

    for i in range(13):
        r_temp = r + dt * sum(A[i][j] * k_r[j] for j in range(i))
        v_temp = v + dt * sum(A[i][j] * k_v[j] for j in range(i))
        k_r[i] = v_temp
        k_v[i] = acceleration(r_temp, v_temp, t_days)

    r_new = r + dt * sum(B[i] * k_r[i] for i in range(13))
    v_new = v + dt * sum(B[i] * k_v[i] for i in range(13))

    return r_new, v_new

# Example simulation (same setup as before)
