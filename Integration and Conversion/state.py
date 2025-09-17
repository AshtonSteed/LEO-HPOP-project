import numpy as np
import butchertableau as bt


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
    # t: Initial time [s]
    def __init__(self, r, v, t):
        self.state = np.concatenate((r, v))
        self.t = t
        self.state_list.append(self.state)
        self.a = 6378.137 #equatorial radius of the earth, km
        self.b = 6356.7523142 #polar radius of the earth, km
        self.am = self.a*1000 #eq radius, m
        self.bm = self.b*1000 #pol radius, m
        self.eccm_2 = (self.am**2-self.bm**2)/(self.am**2)

    # Makes an arrayList of states, we can use this to get the most recent state (such as state_list[-1])

    # Helper Function to return position vector
    def r(self):
        return self.state[:3]

    # Helper Function to return velocity vector
    def v(self):
        return self.state[3:]

    # RK4
    # TODO: Add even higher order integration, RK8??, might be nice to move acceleration call directly into this update
    # TODO: Change acceleration to be a function of the state, add other pertubations to acceleration()
    def state_update(self, dt):
        initial_state = self.last_state()
        x = bt.butcher(8, 0)
        A, B, C = x.radau()
        r = initial_state.r()
        v = initial_state.v()

        k1_r = v
        k1_v = acceleration_g(r)

        k2_r = v + ((4/27) * dt * k1_v)
        k2_v = acceleration_g(r + (4/27 * dt * k1_r))

        k3_r = v + ((2/9) * dt * k2_v)
        k3_v = acceleration_g(r + ((2/9) * dt * k2_r))

        k4_r = v + ((1/3) * dt * k3_v)
        k4_v = acceleration_g(r + ((1/3) * dt * k3_r))

        k5_r = v + ((1/2) * dt * k4_v)
        k5_v = acceleration_g(r + ((1/2) * dt * k4_r))

        k6_r = v + ((2/3) * dt * k5_v)
        k6_v = acceleration_g(r + ((2/3) * dt * k5_r))

        k7_r = v + ((1/6) * dt * k6_v)
        k7_v = acceleration_g(r + ((1/6) * dt * k6_r))

        k8_r = v + ((2/3) * dt * k7_v)
        k8_v = acceleration_g(r + ((2/3) * dt * k7_r))

        k9_r = v + ((5/6) * dt * k8_v)
        k9_v = acceleration_g(r + ((5/6) * dt * k8_r))

        k10_r = v + (dt * k9_v)
        k10_v = acceleration_g(r + (dt * k9_r))


        r_new = r + (dt/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        v_new = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return self.state_list.append(State(r_new, v_new, self.t + dt))

    #TODO: Conversion functions between ECEF and Geodetic Coordinates, (Quan & Zeng, 2024)
    #TODO: Also tack on conversion between ECEF and ECI using epoc of choice (J2000?)

    def acceleration_g(r):
        # TODO: Add Higher order harmonics, make into own file
        # Calculates acceleration due to gravity assuming point mass Earth
        mu = 398600.4418
        r_norm = np.linalg.norm(r)
        return -mu * r / (r_norm ** 3)

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

    # From (You 2000), other various sources

    # Defining N(phi) in the GtECEF conversion
    # Maybelline Flesher
    def prime_vertical_radius(self, lat):
        var1 = self.a**2 * np.cos(lat)**2
        var2 = self.b**2 * np.sin(lat)**2
        return self.a**2 / np.sqrt(var1 + var2)

    # Geodetic to ECEF (Earth Centered, Earth Fixed)
    # Uses latitude (rad), longitude (rad), and altitude (m) as Geodetic coordinates, outputs X, Y, Z (float)
    # N represents ellipsoid radius of curvature in the prime vertical plane
    # Jarrett Usui, Maybelline Flesher
    def geodetic_to_ecef(self, lat, long, alt):
        n_phi = self.prime_vertical_radius(lat)
        x_ecef = (n_phi + alt) * np.cos(long) * np.cos(lat)
        y_ecef = (n_phi + alt) * np.cos(long) * np.sin(lat)
        z_ecef = ((n_phi * (self.b**2/self.a**2)) + alt)*np.sin(long)
        return x_ecef, y_ecef, z_ecef

    # ECEF to Geodetic
    # Uses X, Y, Z as ECEF coordinates (float), and outputs latitude (rad), longitude (rad), and altitude (m).
    # Uses an application of Ferrari's Solution, supposedly has a lower error in the latitude and is much more algebraic in its notation
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
