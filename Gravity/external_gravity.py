import numpy as np
import skyfield.api as sf
from skyfield.api import load
# DE 421 uses these standard Planetary constants (units: km^3 / s^2):
GM_VALUES = {
    'mercury': 22032.09,
    'venus': 324858.59,
    'earth': 398600.436,
    'mars': 42828.37,
    'jupiter': 126712764.8,
    'saturn': 37940585.2,
    'uranus': 5794548.6,
    'neptune': 6836527.1,
    'sun': 132712440041.94,
    'moon': 403503.236310
}

class Bodies:
    def __init__(self, name='de421.bsp'):
        eph = load(name)

        # Standard names in Skyfield for DE421
        # Bodies to be considered gravitationally!
        names = [
            'sun', 'moon'
        ]
        
        # Dictionary mapping planet objects and their GM
        self.planet_data = {}
        for name in names:
            clean_name = name.split()[0] # e.g., 'jupiter'
            self.planet_data[clean_name] = {
                'body': eph[name],
                'gm': GM_VALUES.get(clean_name)
            }
    
    # Returns a summation of the gravitational force
    def get_ext_gravity(self, position, time):
        
        total_force = np.array([0.0, 0.0, 0.0])
        for body, gm in self.planet_data.items():
            # Compute the gravitational acceleration due to this body
            r = body.position(time) - position
            r_mag = r.magnitude()
            if r_mag > 1e-12:  # Avoid division by zero
                force = -gm * r / (r_mag ** 3)
                total_force += force
        return total_force