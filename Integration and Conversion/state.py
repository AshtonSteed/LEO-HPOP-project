import numpy as np
import butchertableau as bt
from datetime import datetime, timezone
from astropy.time import Time
from astropy.coordinates import ITRS, GCRS
import astropy.units as u
import Gravity as g


# U = -(G*M_earth)/r) + {a}
#    sum(N_z)(n=2)[(J_n*P_n^0*sin(theta))/(r^(n+1)] + {b}
#    sum(N_t)(n=2)[sum(n)(m=1)[(P_n^m*sin(theta)* {c}
#                              (C_n^m*cos(m*phi)+S_n^m*sin(m*phi)))/(r^(n+1))]] {d}

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

# TODO: Implement other methods to load sat data, helper functions for sph. coordinates. Might be reasonable to make State as a large array rather than little objects, not sure
class State:
    state_list = []
    # R: Position vector [km]
    # V: Velocity Vector [km/s]
    # t: Initial time [julian]
    def __init__(self, r, v, t_0):
        self.state = np.concatenate((r, v))
        self.t = t_0
        self.state_list.append(self.state)
        self.a = 6378.137 #equatorial radius of the earth, km
        self.b = 6356.7523142 #polar radius of the earth, km
        self.am = self.a*1000 #eq radius, m
        self.bm = self.b*1000 #pol radius, m
        self.eccm_2 = (self.am**2-self.bm**2)/(self.am**2)

    # Makes an arrayList of states, we can use this to get the most recent state (such as state_list[-1])


    # RK8
    # TODO: Add even higher order integration, RK8??, might be nice to move acceleration call directly into this update
    # TODO: Change acceleration to be a function of the state, add other perturbations to acceleration()
    def state_update(self, dt):
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

        return self.state_list.append(State(self.state[:3], self.state[3:], self.t + dt))

    # Matlab Functions
    """
        %--------------------------------------------------------------------------
        %
        %  Runge-Kutta 8th Order
        %
        % Reference:
        % Goddard Trajectory Determination System (GTDS): Mathematical Theory,
        % Goddard Space Flight Center, 1989.
        % 
        % Last modified:   2019/04/15   Meysam Mahooti
        % 
        %--------------------------------------------------------------------------
        function [y] = RK8(func, t, y, h )
        
        k_1 = func(t         ,y                                                                           );
        k_2 = func(t+h*(4/27),y+(h*4/27)*k_1                                                              );
        k_3 = func(t+h*(2/9) ,y+  (h/18)*(k_1+3*k_2)                                                      );
        k_4 = func(t+h*(1/3) ,y+  (h/12)*(k_1+3*k_3)                                                      );
        k_5 = func(t+h*(1/2) ,y+   (h/8)*(k_1+3*k_4)                                                      );
        k_6 = func(t+h*(2/3) ,y+  (h/54)*(13*k_1-27*k_3+42*k_4+8*k_5)                                     );
        k_7 = func(t+h*(1/6) ,y+(h/4320)*(389*k_1-54*k_3+966*k_4-824*k_5+243*k_6)                         );
        k_8 = func(t+h       ,y+  (h/20)*(-234*k_1+81*k_3-1164*k_4+656*k_5-122*k_6+800*k_7)               );
        k_9 = func(t+h*(5/6) ,y+ (h/288)*(-127*k_1+18*k_3-678*k_4+456*k_5-9*k_6+576*k_7+4*k_8)            );
        k_10= func(t+h       ,y+(h/820)*(1481*k_1-81*k_3+7104*k_4-3376*k_5+72*k_6-5040*k_7-60*k_8+720*k_9));
        
        y = y + h/840*(41*k_1+27*k_4+272*k_5+27*k_6+216*k_7+216*k_9+41*k_10);
        
        end
        
        
        % constants
        GM    = 1;                      % gravitational coefficient
        e     = 0.1;                    % eccentricity
        Kep   = [1, e ,0 ,0 ,0 ,0]';    % Keplerian elements (a,e,i,Omega,omega,M)
        
        % header
        fprintf( '\nRunge-Kutta 8th Order Integration\n\n' );
        
        % Initial state of satellite (x,y,z,vx,vy,vz)
        y_0 = State(GM, Kep, 0);
        
        % step size
        h = 0.01; % [s]
        
        % initial values
        t_0 = 0; % [s]
        t_end = 3600; % end time [s]
        
        Steps = t_end/h;
        
        tic
        % Integration from t=t_0 to t=t_end
        for i = 1:Steps-h
            y = RK8(@Deriv, t_0, y_0, h);
            y_0 = y;
            t_0 = t_0 + h;
        end
        y_ref = State(GM, Kep, t_end); % Reference solution
        
        fprintf('Accuracy   Digits\n');
        fprintf('%6.2e',norm(y-y_ref));
        fprintf('%9.2f\n',-log10(norm(y-y_ref)));
        toc
        
        
        function yp = Deriv(t, y)

        % State vector components
        r = y(1:3);
        v = y(4:6);
        
        % State vector derivative
        yp = [v;-r/((norm(r))^3)];
    """


    # Derivative function
    # input time, radius vector, velocity vector, output array w/ velocity vector, acceleration vector
    def deriv(self, t, y):
        r = y[:3]
        v = y[3:]
        return np.array(v, -r/(np.linalg.norm(r)**3))

    """
    t (universal time at position) is included for the acceleration, because we have to turn the ECEF coords 
    into GCRF coords for the derivation, and then back into ECEF for actually using the acceleration.
    
    so this means that we need to include, in the deriv function, the conversions
    """


    """
    def acceleration_g(r):
        # TODO: Add Higher order harmonics, make into own file
        # Calculates acceleration due to gravity assuming point mass Earth
        mu = 398600.4418
        r_norm = np.linalg.norm(r)
        return -mu * r / (r_norm ** 3)
    """

    # Takes Initial state and numerically integrates with time step dt n times
    @staticmethod
    def integrate(old_state, dt, t):
        # TODO: Implement a better integration method, Add Drag/ Other perturbations
        n = int(t / dt)
        current_state = old_state
        for i in range(n):
            #a = acceleration_g(current_state.r())
            current_state = current_state.state_update(dt)
        return current_state


    def main(self):
        print("Hello World 2!")
        r = np.array([6878.0, 0, 0])
        v = np.array([0, 7.61268, 0])
        sat1 = State(r, v, 0)
        time = 94.61 * 60
        dt = 1 #seconds
        ###

        print("Start position: ", sat1.r())

        sat_final = self.integrate(sat1, dt, time * 30 )

        print("End Position: ", sat_final.r())

    if __name__ == "__main__":
        main()

    """
    test using some (r, 0, 0)
    """

    # Defining N(phi) in the Geodetic to ECEF conversion
    # Inputs latitude (rad), outputs...something. it's just kinda as a middleman to make things more readable
    # Maybelline Flesher
    def prime_vertical_radius(self, lat):
        var1 = self.a**2 * np.cos(lat)**2
        var2 = self.b**2 * np.sin(lat)**2
        return self.a**2 / np.sqrt(var1 + var2)


    # Geodetic to ECEF (Earth Centered, Earth Fixed)
    # Uses latitude (rad), longitude (rad), and altitude (m) as Geodetic coordinates, outputs X, Y, Z (float)
    # N represents ellipsoid radius of curvature in the prime vertical plane
    # Converts Geodetic coordinates to ECEF coordinates
    # Jarrett Usui, Maybelline Flesher
    def geodetic_to_ecef(self, lat, long, alt):
        n_phi = self.prime_vertical_radius(lat)
        x_ecef = (n_phi + alt) * np.cos(long) * np.cos(lat)
        y_ecef = (n_phi + alt) * np.cos(long) * np.sin(lat)
        z_ecef = ((n_phi * (self.b**2/self.a**2)) + alt)*np.sin(long)
        return x_ecef, y_ecef, z_ecef


    # ECEF to Geodetic
    # Uses X, Y, Z as ECEF coordinates (float), and outputs latitude (rad), longitude (rad), and altitude (m).
    # Uses an application of Ferrari's Solution, supposedly has a lower error in the latitude and is much more
    #   algebraic in its notation
    # Converts ECEF coordinates to Geodetic coordinates, using Quan & Zheng 2024
    # Jarrett Usui, Maybelline Flesher
    def ecef_to_geodetic(self, x_ecef,y_ecef,z_ecef):
        # Step 1: Prepare the initialized constants and variables
        l = (self.am * self.eccm_2**2)**2
        m = x_ecef**2 + y_ecef**2 #when talking about w^2, just use m for simplicity's sake
        w_axis = np.sqrt(m)
        n = z_ecef**2
        n_c = (1-self.eccm_2)*n
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
                alt = (w_axis * i) + n + (self.am * np.sqrt(i**2 + n_c))
                return lat, long, alt

            # Step 5: Compute Geodetic Coordinates if the following:
            if t == 0 and n == 0:
                lat = m * np.sign(y_ecef) * ((np.pi / 2) - 2 * np.arctan(x_ecef / (w_axis + np.abs(y_ecef))))
                long = 2 * np.arctan2(y_ecef, x_ecef)
                alt = -1 * np.sqrt(1 - self.eccm_2)
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
        lat = np.arctan2(z_ecef, r * (1 - self.eccm_2))

        # Step 4: Iteratively refine latitude
        for _ in range(max_iter):
            n = self.am / np.sqrt(1 - self.eccm_2 * np.sin(lat)**2)  # Radius of curvature
            alt = r / np.cos(lat) - n
            lat_new = np.arctan2(z_ecef, r * (1 - self.eccm_2 * n / (n + alt)))
            if np.abs(lat_new - lat) < tol:
                lat = lat_new
                break
            lat = lat_new

        # Step 5: Final altitude calculation using latitude
        n = self.am / np.sqrt(1 - self.eccm_2 * np.sin(lat)**2)
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


    def eci_to_ecef(self, x_eci, y_eci, z_eci, dt: datetime):
        # Convert datetime to Astropy Time
        t = Time(dt, scale="utc")

        # Represent ECEF vector in ITRS frame (ECEF = ITRS)
        gcrs = GCRS(x=x_eci * u.cm * 100, y=y_eci * u.cm * 100, z=z_eci * u.cm * 100, obstime=t)

        # Convert to GCRS (which is aligned to GCRF, Earth-centered)
        itrs = gcrs.transform_to(ITRS(obstime=t))

        # Return as X, Y, Z
        return itrs.cartesian.x.to(u.cm*100).value, itrs.cartesian.y.to(u.cm*100).value, itrs.cartesian.z.to(u.cm*100).value

