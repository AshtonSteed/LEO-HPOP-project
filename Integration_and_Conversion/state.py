import numpy as np
from datetime import datetime, timezone
from astropy.time import Time
from astropy.coordinates import ITRS, GCRS
import astropy.units as u
from Gravity.gravity import Gravity as grav
from skyfield.framelib import itrs
from skyfield.functions import T


# U = -(G*M_earth)/radius) + {a}
#    sum(N_z)(n=2)[(J_n*P_n^0*sin(theta))/(radius^(n+1)] + {b}
#    sum(N_t)(n=2)[sum(n)(m=1)[(P_n^m*sin(theta)* {c}
#                              (C_n^m*cos(m*phi)+S_n^m*sin(m*phi)))/(radius^(n+1))]] {d}

# Before you begin typing ANY code, you MUST write a comment of what you are intending to do with the function.
# An example would be:
# Name of Function
# Arguments / variables used, and what they mean
# What the intention of the function is

# Make it easier for the computer and you to figure out what to do with the data.
# Another important thing is to use the libraries given. numpy has all the functions you need for individual data;
# redundancies just take up more space and are less efficient.

# We need to initialize the numpy array.
# We don't have the file that we are getting the data from, but preferably we can export it as a csv.
#leo_data = pd.read_csv('~~.csv', ',')

# We need to declare all the variables needed in the equations - by applying those variables to parts of
# the numpy array:

# We need to define functions for each section, so that when we combine them all, it's less of a stress on the system,
# and is more readable for readers.

# TODO: Implement other methods to load sat data,
#  Might be reasonable to make State as a large array rather than little objects, not sure
class State:

    # Makes a list of states, we can use this to get the most recent state (such as state_list[-1])
    state_list = []
    # R: Position vector [km]
    # V: Velocity Vector [km/s]
    # t: Initial time [julian]

    # conversion consts
    A = 6378.137  # equatorial radius of the earth, km
    B = 6356.7523142  # polar radius of the earth, km
    Am = A * 1000  # eq radius, m
    Bm = B * 1000  # pol radius, m
    ECCm_2 = (Am**2 - Bm**2) / (Am**2)

    def __init__(self, rv, t_0):
        # state update vars
        self.state = rv #HAS to be in GCRF for the integration methods
        self.t = t_0
        self.state_list.append(self.state)
        
    """Utility Functions"""
    # Get position vector
    def r(self):
        return self.state[:3]
    # Get velocity vector
    def v(self):    
        return self.state[3:]
    
    # Convert state to Keplerian Orbital Elements with singularity handling
    def to_koe(self, mu):
        # Calculate specific angular momentum vector
        r = self.r()
        v = self.v()
        h_vec = np.cross(r, v)
        
        # Calculate unit Eccentricity vector and value
        e_vec = np.cross(v, h_vec) / mu - r / np.linalg.norm(r)
        e = np.linalg.norm(e_vec)
        
        # Find vector toward ascending node
        n_vec = np.cross([0, 0, 1], h_vec)
        
        # Calculate Inclination
        i = np.arccos(h_vec[2] / np.linalg.norm(h_vec))
        
        # Handle singularities based on eccentricity and inclination
        tolerance = 1e-10  # Small tolerance for numerical stability
        
        # Case 1: Circular orbit (e ≈ 0)
        if e < tolerance:
            # For circular orbits, argument of periapsis is undefined
            # Return argument of latitude (u = ω + ν) instead
            if np.linalg.norm(n_vec) > tolerance:
                # Non-equatorial circular orbit - RAAN is still defined
                # Calculate RAAN
                if n_vec[1] >= 0:
                    raan = np.arccos(n_vec[0] / np.linalg.norm(n_vec))
                else:
                    raan = 2 * np.pi - np.arccos(n_vec[0] / np.linalg.norm(n_vec))
                
                # Calculate argument of latitude (u = ω + ν)
                u = np.arccos(np.dot(n_vec, r) / (np.linalg.norm(n_vec) * np.linalg.norm(r)))
                if r[2] >= 0:
                    u = u
                else:
                    u = 2 * np.pi - u
                arg_peri = 0.0  # Undefined for circular orbits
                theta = u  # Argument of latitude
            else:
                # Equatorial circular orbit - both RAAN and argument of periapsis are undefined
                raan = 0.0  # Reference direction
                theta = np.arctan2(r[1], r[0])  # True longitude
                arg_peri = 0.0  # Undefined for circular orbits
        else:
            # Case 2: Elliptical orbit (e > 0)
            
            # Calculate RAAN (Right Ascension of Ascending Node)
            if np.linalg.norm(n_vec) > tolerance:
                # Non-equatorial orbit
                if n_vec[1] >= 0:
                    raan = np.arccos(n_vec[0] / np.linalg.norm(n_vec))
                else:
                    raan = 2 * np.pi - np.arccos(n_vec[0] / np.linalg.norm(n_vec))
            else:
                # Equatorial orbit - RAAN is undefined, set to 0
                raan = 0.0
            
            # Calculate Argument of Periapsis
            if np.linalg.norm(n_vec) > tolerance:
                # Non-equatorial elliptical orbit
                if e_vec[2] >= 0:
                    arg_peri = np.arccos(np.dot(n_vec, e_vec) / (np.linalg.norm(n_vec) * e))
                else:
                    arg_peri = 2 * np.pi - np.arccos(np.dot(n_vec, e_vec) / (np.linalg.norm(n_vec) * e))
            else:
                # Equatorial elliptical orbit - use longitude of perigee
                # Longitude of perigee = RAAN + ω, but RAAN is undefined for equatorial orbits
                # So we return the longitude of perigee directly
                arg_peri = np.arctan2(e_vec[1], e_vec[0])
                if arg_peri < 0:
                    arg_peri += 2 * np.pi
            
            # Calculate True Anomaly
            if np.dot(r, v) >= 0:
                # Moving away from periapsis
                theta = np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
            else:
                # Moving toward periapsis
                theta = 2 * np.pi - np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
        
        # Calculate Semi-Major Axis
        a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v)**2 / mu)
        
        return a, e, i, raan, arg_peri, theta
        
        
        
        
    """
    STATE UPDATE FUNCTIONS
    """
    # Test using array of positions

    # Runge-Kutta 4th Order (just for testing purposes)
    def rk4(self, dt):
        k1 = self.deriv(self.t, self.state)
        k2 = self.deriv(self.t + 0.5 * dt, self.state + 0.5 * k1)
        k3 = self.deriv(self.t + 0.5 * dt, self.state + 0.5 * k2)
        k4 = self.deriv(self.t + dt, self.state + k3)

        self.state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return self.state_list.append(State(self.state[:3], self.state[3:], self.t + dt))

    # Runge-Kutta 8th Order
    #
    #     t (universal time at position) is included for the acceleration, because we have to turn the Cartesian GCRF
    #     coords into Spherical ECEF coords for the derivation, and then back into Cartesian GCRF for actually using
    #     the acceleration. This means that we need to include the conversions in the deriv function.
    #
    def rk8(self, dt):
        k_1 = self.deriv(self.t, self.state)
        k_2 = self.deriv(self.t + dt * (4 / 27), self.state + (dt * 4 / 27) * k_1)
        k_3 = self.deriv(self.t + dt * (2 / 9), self.state + (dt / 18) * (k_1 + 3 * k_2))
        k_4 = self.deriv(self.t + dt * (1 / 3), self.state + (dt / 12) * (k_1 + 3 * k_3))
        k_5 = self.deriv(self.t + dt * (1 / 2), self.state + (dt / 8) * (k_1 + 3 * k_4))
        k_6 = self.deriv(self.t + dt * (2 / 3), self.state + (dt / 54) * (13 * k_1 - 27 * k_3 + 42 * k_4 + 8 * k_5))
        k_7 = self.deriv(self.t + dt * (1 / 6), self.state + (dt / 4320) * (389 * k_1 - 54 * k_3 + 966 * k_4 - 824 *
                                                                            k_5 + 243 * k_6))
        k_8 = self.deriv(self.t + dt, self.state + (dt / 20) * (-234 * k_1 + 81 * k_3 - 1164 * k_4 + 656 * k_5 -
                                                                122 * k_6 + 800 * k_7))
        k_9 = self.deriv(self.t + dt * (5 / 6), self.state + (dt / 288) * (-127 * k_1 + 18 * k_3 - 678 * k_4 + 456 *
                                                                           k_5 - 9 * k_6 + 576 * k_7 + 4 * k_8))
        k_10 = self.deriv(self.t + dt, self.state + (dt / 820) * (1481 * k_1 - 81 * k_3 + 7104 * k_4 - 3376 * k_5 +
                                                                  72 * k_6 - 5040 * k_7 - 60 * k_8 + 720 * k_9))

        self.state = self.state + dt / 840 * (41 * k_1 + 27 * k_4 + 272 * k_5 + 27 * k_6 + 216 * k_7 + 216 * k_9 + 41
                                              * k_10)

        return self.state_list.append(State(self.r(), self.v(), self.t + dt))

    # Derivative function
    # input time, radius vector, velocity vector, output array w/ velocity vector, acceleration vector
    def deriv(self, t, y):
        r, v = y[:3], y[3:]
        a_ecef = grav.acceleration_g(*self.xyz_to_sphr(self.gcrf_to_ecef(*r, t)))
        a_gcrf = np.array(self.sphr_to_xyz(self.ecef_to_gcrf(*a_ecef, t)))
        return np.concatenate(v, a_gcrf)

    """
    CONVERSION FUNCTIONS
    """
    # test using some (radius, 0, 0)

    # Defining N(phi) in the Geodetic to ECEF conversion
    # Inputs latitude (rad), outputs...something. it's just kinda as a middleman to make things more readable
    # Maybelline Flesher
    def prime_vertical_radius(self, lat):
        var1 = self.A ** 2 * np.cos(lat) ** 2
        var2 = self.B ** 2 * np.sin(lat) ** 2
        return self.A**2 / np.sqrt(var1 + var2)


    # Geodetic to ECEF (Earth Centered, Earth Fixed)
    # Uses latitude (rad), longitude (rad), and altitude (m) as Geodetic coordinates, outputs X, Y, Z (float)
    # N represents ellipsoid radius of curvature in the prime vertical plane
    # Converts Geodetic coordinates to ECEF coordinates
    # Jarrett Usui, Maybelline Flesher
    def geodetic_to_ecef(self, lat, long, alt):
        n_phi = self.prime_vertical_radius(lat)
        x_ecef = (n_phi + alt) * np.cos(long) * np.cos(lat)
        y_ecef = (n_phi + alt) * np.cos(long) * np.sin(lat)
        z_ecef = ((n_phi * (self.B ** 2 / self.A ** 2)) + alt) * np.sin(long)
        return x_ecef, y_ecef, z_ecef


    # ECEF to Geodetic
    # Uses X, Y, Z as ECEF coordinates (float), and outputs latitude (rad), longitude (rad), and altitude (m).
    # Uses an application of Ferrari's Solution, supposedly has a lower error in the latitude and is much more
    #   algebraic in its notation
    # Converts ECEF coordinates to Geodetic coordinates, using Quan & Zheng 2024
    # Jarrett Usui, Maybelline Flesher
    def ecef_to_geodetic(self, x_ecef,y_ecef,z_ecef):
        # Step 1: Prepare the initialized constants and variables
        l = (self.Am * self.ECCm_2 ** 2) ** 2
        m = x_ecef**2 + y_ecef**2 # when talking about w^2, just use m for simplicity's sake
        w_axis = np.sqrt(m)
        n = z_ecef**2
        n_c = (1 - self.ECCm_2) * n
        p = m + n_c - l
        q = 27 * m * n_c * l

        # Step 2: Distinguish the applicable zone
        if p**3 + q >= 0:
            temp_1 = (np.sqrt(p**3 + q) + np.sqrt(q))**(2/3)
            temp_2 = (np.sqrt(p**3 + q) - np.sqrt(q))**(2/3)
            t = p + temp_1 + temp_2
        else:
            temp_1 = np.sqrt(-q/p)
            temp_2 = (np.arccos(np.sqrt(-q/(p**3))))/3
            t = temp_1/(np.cos(temp_2))

        # If variable "t" passes the isfinite check (meaning that it is not an infinite float and it's not NaN)
        if np.isfinite(t):

            # Step 3: Compute the intermediate variables
            u_m = np.sqrt((36 * m * l) + t**2)
            u_n_c = np.sqrt((36 * n_c * l) + t**2)
            v = u_m + u_n_c
            w = (2 * t) + (6 * l) + v
            i = (2 * (t + u_n_c))/(w + np.sqrt(6 * l * (w + v + 6*(m + n_c))))
            s = np.sqrt(i**2 + n)

            # Step 4: Compute Geodetic Coordinates if the following:
            if t > 0 or n > 0:
                lat = m*np.sign(y_ecef) * ((np.pi / 2) - 2 * np.arctan(x_ecef / (w_axis + np.abs(y_ecef))))
                long = 2 * np.arctan(z_ecef / (i + s))
                alt = (w_axis * i) + n + (self.Am * np.sqrt(i ** 2 + n_c))
                return lat, long, alt

            # Step 5: Compute Geodetic Coordinates if the following:
            if t == 0 and n == 0:
                lat = m * np.sign(y_ecef) * ((np.pi / 2) - 2 * np.arctan(x_ecef / (w_axis + np.abs(y_ecef))))
                long = 2 * np.arctan2(y_ecef, x_ecef)
                alt = -1 * np.sqrt(1 - self.ECCm_2)
                return lat, long, alt

        # If the variable "t" fails the isfinite() check:
        # We switch to an iterative method called the Bowring Method, which has the same validity but may
        #   be worse, efficiency-wise. Both methods follow O(n) growth since the variables are constant
        else:
            return self.ecef_to_geodetic_bowring(x_ecef, y_ecef, z_ecef)


    # ECEF to Geodetic (Using Bowring's Iterative Method)
    # Uses X, Y, Z as ECEF (float), as well as default max iterations of 5 and a convergence tolerance of 1e-12,
    #   and outputs latitude (rad), longitude (rad), and altitude (m)
    # Has a higher error of the latitude versus the more algebraic closed-form approach
    # Converts ECEF coordinates to Geodetic coordinates using Bowrings Method
    # Maybelline Flesher
    def ecef_to_geodetic_bowring(self, x_ecef, y_ecef, z_ecef, max_iter=5, tol=1e-12):
        # Step 1: Calculate Longitude
        long = np.arctan2(y_ecef, x_ecef)

        # Step 2: Find the distance from the Z-axis
        r = np.sqrt(x_ecef**2 + y_ecef**2)

        # Step 3: Initial guess for latitude
        lat = np.arctan2(z_ecef, r * (1 - self.ECCm_2))

        # Step 4: Iteratively refine latitude
        for _ in range(max_iter):
            n = self.Am / np.sqrt(1 - self.ECCm_2 * np.sin(lat) ** 2)  # Radius of curvature
            alt = r / np.cos(lat) - n
            lat_new = np.arctan2(z_ecef, r * (1 - self.ECCm_2 * n / (n + alt)))
            if np.abs(lat_new - lat) < tol:
                lat = lat_new
                break
            lat = lat_new

        # Step 5: Final altitude calculation using latitude
        n = self.Am / np.sqrt(1 - self.ECCm_2 * np.sin(lat) ** 2)
        alt = r / np.cos(lat) - n

        return lat, long, alt


    # UTC to Julian Date
    # Inputs as the UTC date/time (float), outputs to a Julian Date (float)
    # Maybelline Flesher
    @staticmethod
    def utc_to_julian(dt: datetime) -> float:
        dt = dt.astimezone(timezone.utc)
        year, month = dt.year, dt.month
        day = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0

        if month <= 2:
            year -= 1
            month += 12

        a = int(year / 100)
        b = 2 - a + int(a / 4)

        julian_date = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        return julian_date

    # The Greenwich Mean Sidereal Time (rad)
    # Input as the Julian Date, and output as the GMST in radians
    @staticmethod
    def gmst_angle(julian_date: float) -> float:
        """Greenwich Mean Sidereal Time (radians)."""
        time = (julian_date - 2451545.0) / 36525.0  # Julian centuries from J2000
        gmst = 280.46061837 + 360.98564736629 * (julian_date - 2451545.0) \
               + 0.000387933 * time**2 - time**3 / 38710000.0
        gmst = np.deg2rad(gmst % 360.0)
        return gmst

    # ECEF to ECI (GCRF)
    # Input as X, Y, Z coordinates (float) (non-inertial frame), and output as X, Y, Z
    #   coordinates (float) (inertial frame)
    # Maybelline Flesher
    def ecef_to_gcrf(self, x_ecef, y_ecef, z_ecef, dt: datetime):
        # Convert datetime to Astropy Time
        t = Time(dt, scale="utc")

        # Represent ECEF vector in ITRS frame (ECEF = ITRS)
        itrs = ITRS(x=x_ecef * u.cm*100, y=y_ecef * u.cm*100, z=z_ecef * u.cm*100, obstime=t)

        # Convert to GCRS (which is aligned to GCRF, Earth-centered)
        gcrs = itrs.transform_to(GCRS(obstime=t))

        # Return as X, Y, Z
        return gcrs.cartesian.x.to(u.cm*100).value, gcrs.cartesian.y.to(u.cm*100).value, gcrs.cartesian.z.to(u.cm*100).value

    # ICRS to ITRS (ECEF)
    # Input as X, Y, Z coordinates (float) (inertial frame), and output as X, Y, Z
    #   coordinates (float) (non-inertial frame)
    # Same as above, useing skyfield instead of astropy for some more efficiency
    # t is datetime in TBD seconds, ts is a skyfield timescale object
    @staticmethod
    def icrs_to_itrs(r_eci, t, ts):
        #
        t_obj = ts.tdb(jd=t/86400) # Convert seconds to days for skyfield
        R = itrs.rotation_at(t_obj) # Find Rotation between ICRS and ITRS at t
        r_itrs = R @ r_eci # Rotate vector into ITRS
        return r_itrs
    
    # ITRS to ICRS 
    # Input as X, Y, Z coordinates (float) (non-inertial frame), and output as X, Y, Z
    #   coordinates (float) (inertial frame)
    # Same as nelow, useing skyfield instead of astropy for some more efficiency
    # t is datetime in JD float, ts is a skyfield timescale object
    @staticmethod
    def itrs_to_icrs(r_ecef, t, ts):
        t_obj = ts.tdb(jd=t/86400) # Convert seconds to days for skyfield
        R = itrs.rotation_at(t_obj) # Find Rotation between ICRS and ITRS at t
        R_T = T(R) # Invert rotation matrix, ITRS to ICRS
        r_icrs = R_T @ r_ecef  # Rotate vector into ICRS
        return r_icrs


    def gcrf_to_ecef(self, x_eci, y_eci, z_eci, dt: datetime):
        # Convert datetime to Astropy Time
        t = Time(dt, scale="utc")

        # Represent ECEF vector in ITRS frame (ECEF = ITRS)
        gcrs = GCRS(x=x_eci * u.cm * 100, y=y_eci * u.cm * 100, z=z_eci * u.cm * 100, obstime=t)

        # Convert to GCRS (which is aligned to GCRF, Earth-centered)
        itrs = gcrs.transform_to(ITRS(obstime=t))

        # Return as X, Y, Z
        return itrs.cartesian.x.to(u.cm*100).value, itrs.cartesian.y.to(u.cm*100).value, itrs.cartesian.z.to(u.cm*100).value


    # Convert position spherical vector to cartesian coordinate vector [r,theta,phi] -> [x,y,z]
    @staticmethod
    def sphr_to_xyz_point(sphr):
        output = np.zeros(3)
        
        #unpack spherical cords
        (r,theta,phi) = sphr
        
        #Calculate z and xy components
        z = r * np.cos(theta)
        xy = r * np.sin(theta)
        
        #Caculate x and y
        x = xy * np.cos(phi)
        y = xy * np.sin(phi)
        
        #Assign to output vector
        output = x,y,z
        
        return output
    
    # Convert Spherical coordinates to Cartesian [xyz] -> [radius, theta, phi]
    @staticmethod
    def xyz_to_sphr(xyz):
        #create sphr vector
        sphr = np.zeros(3)
        
        #unpack xyz
        (x,y,z) = xyz
        
        #Calculate Spherical Coordinates
        r = np.linalg.norm(xyz)
        phi = np.atan2(y,x)
        theta = np.acos(z/r)
        
        sphr = r,theta,phi
        return sphr

    @staticmethod
    def sphr_to_xyz_vec(point, vec):

        r,theta,phi = point
        sp = np.sin(phi)
        cp = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)

        m = np.array([[st * cp, ct * cp, -sp],[st*sp,ct * sp,cp],[ct, -st, 0]])
        return np.dot(m, vec)






if __name__ == "__main__":
    r = np.array([8000,2000,100])

    rx = State.xyz_to_sphr(r)

    ry = State.sphr_to_xyz(rx)


    print(r, rx, ry)