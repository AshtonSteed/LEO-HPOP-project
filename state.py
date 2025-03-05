import numpy as np
import pandas as pd
import scipy as sp
import butchertableau as bt

# U = -(G*M_earth)/r) + {a}
#    sum(N_z)(n=2)[(J_n*P_n^0*sin(theta))/(r^(n+1)] + {b}
#    sum(N_t)(n=2)[sum(n)(m=1)[(P_n^m*sin(theta)*(C_n^m*cos(m*phi)+S_n^m*sin(m*phi)))/(r^(n+1))] {c,d}

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

# We need to define functions for each ssection, so that when we combine them all, it's less of a stress on the system,
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

    #k_1 = func(t, y);
    #k_2 = func(t + h * (4 / 27), y + (h * 4 / 27) * k_1);
    #k_3 = func(t + h * (2 / 9), y + (h / 18) * (k_1 + 3 * k_2));
    #k_4 = func(t + h * (1 / 3), y + (h / 12) * (k_1 + 3 * k_3));
    #k_5 = func(t + h * (1 / 2), y + (h / 8) * (k_1 + 3 * k_4));
    #k_6 = func(t + h * (2 / 3), y + (h / 54) * (13 * k_1 - 27 * k_3 + 42 * k_4 + 8 * k_5));
    #k_7 = func(t + h * (1 / 6), y + (h / 4320) * (389 * k_1 - 54 * k_3 + 966 * k_4 - 824 * k_5 + 243 * k_6));
    #k_8 = func(t + h, y + (h / 20) * (-234 * k_1 + 81 * k_3 - 1164 * k_4 + 656 * k_5 - 122 * k_6 + 800 * k_7));
    #k_9 = func(t + h * (5 / 6),
    #           y + (h / 288) * (-127 * k_1 + 18 * k_3 - 678 * k_4 + 456 * k_5 - 9 * k_6 + 576 * k_7 + 4 * k_8));
    #k_10 = func(t + h, y + (h / 820) * (
    #            1481 * k_1 - 81 * k_3 + 7104 * k_4 - 3376 * k_5 + 72 * k_6 - 5040 * k_7 - 60 * k_8 + 720 * k_9));
    #y = y + h / 840 * (41 * k_1 + 27 * k_4 + 272 * k_5 + 27 * k_6 + 216 * k_7 + 216 * k_9 + 41 * k_10);

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
