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
        
        self.earth = eph['earth']

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
    def get_ext_gravity(self, position, t, timescale):
        # Find Sum External body acceleration at a time from an epoc ts
        # Position is Cartesian Vector (x,y,z) in km in GCRF
        # T is time in seconds since Simulation Start
        # timescale is a Skyfield timescale object to convert t to a time object for ephemeris lookup
        time = timescale.tdb(jd=t/86400)
        total_acceleration = np.array([0.0, 0.0, 0.0])
        for body_name, data in self.planet_data.items():
            body = data['body']
            gm = data['gm']
            # Compute the gravitational acceleration due to this body
            # Skyfield body is relative to solar system barycenter (ICRF), so use Earth position to find relative position to ITRS
            
            r_body = (self.earth - body).at(time).position.km
            r_body_mag = np.linalg.norm(r_body)
            
            r_sat_to_body = r_body - position
            r_sat_to_body_mag = np.linalg.norm(r_sat_to_body)
            
            if r_body_mag > 1e-12 and r_sat_to_body_mag > 1e-12:  # Avoid division by zero
                # Calculate acceleration with respect to Earth-centered inertial frame
                # So subtract acceleration of body onto Earth
                direct_term = r_sat_to_body / (r_sat_to_body_mag ** 3)
                indirect_term = r_body / (r_body_mag ** 3)
                
                acceleration = gm * (direct_term - indirect_term)
                total_acceleration += acceleration
        return total_acceleration

    
    def get_srp(self, position, t, timescale, area=1e-6, reflectivity=0.3, mass=1.0):
        time = timescale.tdb(jd=t/86400)
        
        sun = self.planet_data['sun']
        
        # Find distance to sun from earth and to sun from satellite
        r_body = (self.earth - sun).at(time).position.km
        r_sat_to_body = r_body - position
        
        dist_to_sun = np.linalg.norm(r_sat_to_body)
        
        
        # Calc Force
        # Placeholder for Cannonball Solar Radiation Pressure acceleration 
        s = 1367e6 # Solar Constant, W/km^2
        c = 299792.458 # Speed of light, km/s
        v = 1 #TODO: Add method to check visibility to sun
        a = -v * s/c * reflectivity * area / mass * r_sat_to_body / dist_to_sun
        
        return a