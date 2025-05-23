import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt




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






# We need to define functions for each section, so that when we combine them all, it's less of a stress on the system,
# and is more readable for readers.

#TODO: Implement other methods to load sat data, helper functions for sph. coordinates. 
class State:
    # R: Position vector [km] ECI
    # V: Velocity Vector [km/s] ECI
    # t: Initial time [s]
    def __init__(self, r, v, t):
        self.state = np.concatenate((r, v))
        self.t = t
    # Helper Function to return position vector
    def r(self):
        return self.state[:3] 
     # Helper Function to return velocity vector
    def v(self):
        return self.state[3:] 
    
    # Simple Euler Method Integration
    # TODO: RK4 or similar method, might be nice to move acceleration call directly into this update
    # Takes an old state and new acceleration vector, returns new updated state after Euler Method step
    def state_update(self, acceleration, dt):
        r = self.r()
        v = self.v()
        
        r_new = r + v*dt
        v_new = v + acceleration*dt
        
        # Not modifying state in place, but returning new state
        # Should make it easier to save state data and plot path over time
        #self.set_r(r_new)
        #self.set_v(v_new)
        
        return State(r_new, v_new, self.t + dt)
    
    #TODO: Conversion functions between ECEF and Geodetic Coordinates, (Quan & Zeng, 2024)
    #TODO: Also tack on conversion between ECEF and ECI using epoc of choice (J2000?)

    
def acceleration_g(state):
    # TODO: Add Higher order harmonics, make into own file
    # Calculates acceleration due to gravity assuming point mass Earth
    mu = 398600.4418
    r = state.r()
    r_norm = np.linalg.norm(r)
    return -mu*r/(r_norm**3)

# Takes Initial state and numerically integrates with time step dt n times
def integrate(old_state, dt, t):
    # TODO: Implement a better integration method, Add Drag/ Other perturbations
    n = int(t/dt)
    for i in range(n):
        a = acceleration_g(old_state)
        new_state = old_state.state_update(a, dt)
        old_state = new_state
    return new_state


def main():
    print("Hello World 2!")
    r = np.array([6878.0, 0, 0]) #500km Circular Orbit
    v = np.array([0, 7.61268, 0])
    Sat1 = State(r, v, 0)
    T = 94.61 * 60 # seconds
    dt = 1 # seconds
    ###
    
    print("Start position: ", Sat1.r())
    
    Satfinal = integrate(Sat1, dt, T*10)
    
    print("End Position: ", Satfinal.r())
    
   
    
    
    
    
    
    
    
    



if __name__ == "__main__":
    main()


