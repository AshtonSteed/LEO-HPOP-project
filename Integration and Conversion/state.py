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
    def integrate(old_state, dt, t):
        # TODO: Implement a better integration method, Add Drag/ Other perturbations
        n = int(t / dt)
        current_state = old_state
        for i in range(n):
            #a = acceleration_g(current_state.r())
            current_state = current_state.state_update(dt)
        return current_state

    def main():
        print("Hello World 2!")
        r = np.array([6878.0, 0, 0])
        v = np.array([0, 7.61268, 0])
        Sat1 = State(r, v, 0)
        T = 94.61 * 60
        dt = 1 #seconds
        ###

        print("Start position: ", Sat1.r())

        Satfinal = integrate(Sat1, dt, T * 30 )

        print("End Position: ", Satfinal.r())

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
    # Uses latitude, longitude, and altitude as input for calculations, outputs XYZ
    # N represents ellipsoid radius of curvature in the prime vertical plane
    # Function converts Geodetic (latitude, longitude, altitude) coordinates into Earth Centered Earth Fixed (ECEF) coordinates, or x,y,z
    # Jarrett Usui, Maybelline Flesher
    def geodetic_to_ecef(self, lat, long, alt):
        n_phi = self.prime_vertical_radius(lat)
        x_ecef = (n_phi + alt) * np.cos(long) * np.cos(lat)
        y_ecef = (n_phi + alt) * np.cos(long) * np.sin(lat)
        z_ecef = ((n_phi * (self.b**2/self.a**2)) + alt)*np.sin(long)
        return x_ecef, y_ecef, z_ecef

    # ECEF to Geodetic
    # Uses XYZ coordinates as input for calculations, and outputs latitude, longitude, and altitude.
    # Uses an application of Ferrari's solution, supposedly (Zhu 1994) is the most accurate, proposed
    #   by Heikkinen (1984)
    # I can't find Heikkinen's, but (You 2000) presents the math behind it. We will be using a
    #   Non-Iterative Method of the Zeroth Order.
    # Function converts Cartesian ECEF coords (X,Y,Z) into Geodetic (latitude, longitude, altitude)
    # Jarrett Usui, Maybelline Flesher
    def Convert2Geodetic(self, x_ecef,y_ecef,z_ecef):

        latitude = 2*np.arctan(Z/(I+S))
        longitude = np.sign(Y)*(np.pi/2-2*np.arctan(X/(W+abs(Y)))
        altitude = -1*np.sqrt(1-e**2)*sqrt(a**2-m/e**2)
        return (latitude,longitude,altitude)


