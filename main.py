import numpy as np
import pandas as pd
import scipy as sp 
import matplotlib.pyplot as plt
from Gravity.gravity import Gravity
from Integration_and_Conversion.state import State



# Name of Function: state_update_full
# Arguments / variables used:
#   - t: Current time (unused, but required by solve_ivp signature)
#   - statevec: A numpy array representing the current state vector [rx, ry, rz, vx, vy, vz]
#   This function serves as the derivative function for scipy.integrate.solve_ivp, returning
#   the derivatives of the state vector [vx, vy, vz, ax, ay, az].
def state_update_full(t, statevec, g: Gravity):

    # Initialize output array for derivatives [vx, vy, vz, ax, ay, az]
    output = np.zeros_like(statevec)

    # Create a State object for convinience in coordinate conversion and data pulling
    state = State(statevec, t)

    # Extract position and velocity vectors
    rvec = state.r()
    v = state.v()

    #Calculate Spherical position vector
    #TODO: account for GCRF->ECEF Spherical
    sphr_r = State.xyz_to_sphr(rvec)
    (r,theta,phi) = sphr_r
    #calculate acceleration up to N=M=20
    a_s = g.acceleration_g(r, theta, phi)
    # Convert acceleration to cartesian
    acart = State.sphr_to_xyz_vec(sphr_r,a_s)
    #acart = State.sphr_to_xyz_point(a_s)



    output[:3] = v


    # The last three elements of the output are the acceleration components (dv/dt = a)
    output[3:] = acart

    return output


# Name of Function: state_update_full
# Arguments / variables used:
#   - t: Current time (unused, but required by solve_ivp signature)
#   - statevec: A numpy array representing the current state vector [rx, ry, rz, vx, vy, vz]
#   This function serves as the derivative function for scipy.integrate.solve_ivp, returning
#   the derivatives of the state vector [vx, vy, vz, ax, ay, az].
def state_update(t, statevec, g: Gravity):
    
    # Initialize output array for derivatives [vx, vy, vz, ax, ay, az]
    output = np.zeros_like(statevec)
    
    # Create a State object for convinience in coordinate conversion and data pulling
    state = State(statevec, t)
    
    # Extract position and velocity vectors
    r = state.r()
    v = state.v()

    # Calculate the norm (magnitude) of the position vector
    r_norm = np.linalg.norm(r)
    a = -g.mu * r / (r_norm**3) # Point mass earth acceleration
    output[:3] = v
    # The last three elements of the output are the acceleration components (dv/dt = a)
    output[3:] = a
    
    return output


# Name of Function: integrate
# Arguments / variables used:
#   - initial_state_vector: A numpy array representing the initial state vector [rx, ry, rz, vx, vy, vz].
#   - t_span: A tuple (t0, tf) representing the start and end times for the integration.
#   - dt: The initial step size for the integration.
# What the intention of the function is:
#   This function numerically integrates the equations of motion using scipy.integrate.solve_ivp.
#   It takes an initial state vector and a time span, and returns the results of the integration.
def integrate(initial_state_vector, t_span, dt, g):
    
    # Use scipy.integrate.solve_ivp to perform the numerical integration
    results = sp.integrate.solve_ivp(state_update, t_span, initial_state_vector, 
                                     first_step=dt, rtol=1e-9, atol=1e-9, method='DOP853', args=[g])
    return results


def main():
    print("Starting orbital simulation...")
    
    g = Gravity()
    
    # Initial conditions for a 500km circular orbit
    r_initial = np.array([g.radius + 500, 0, 0])  # Position vector [km]
    v_initial = np.array([0, np.sqrt(g.mu / (g.radius + 500)), 0]) # Velocity vector [km/s]
    t_initial = 0.0                       # Initial time [s]

    # Create a State object to hold initial conditions
    initial_satellite_state = State(np.concatenate((r_initial, v_initial)), t_initial)
    
    # Define simulation parameters
    orbital_period = np.sqrt(4 * np.pi**2 * np.linalg.norm(r_initial)**3 /g.mu) # seconds for one orbit
    print(orbital_period)
    simulation_duration = orbital_period * 10 # Simulate for 30 orbits
    time_step = 1               # Initial step size for the solver [s]
    
    # Define the time span for integration (from t_initial to simulation_duration)
    t_span = (t_initial, simulation_duration)
    
    print("Initial position: ", initial_satellite_state.r())
    print("Initial velocity: ", initial_satellite_state.v())
    
    # Integrate the equations of motion
    # Pass the raw state vector from the State object
    simulation_results = integrate(initial_satellite_state.state, t_span, time_step,g)
    
    print("Simulation completed.")
    print("Final Position: ", simulation_results.y[:3,-1])
    print("Number of saved positions: ", len(simulation_results.y[0]))
    
    # Plot the orbit
    plot_orbit(simulation_results, g.radius)


# Name of Function: plot_orbit
# Arguments / variables used:
#   - results: A scipy.integrate.OdeResult object containing the time and state vectors from the integration.
# What the intention of the function is:
#   This function takes the results of an orbital integration and plots the 3D trajectory of the satellite.
#   It extracts the x, y, and z position components over time and visualizes them using matplotlib.
def plot_orbit(results, r_earth):
    # Extract position vectors (x, y, z) from the integration results
    x = results.y[0, :]
    y = results.y[1, :]
    z = results.y[2, :]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Earth as a sphere
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = r_earth * np.outer(np.cos(u), np.sin(v))
    y_sphere = r_earth * np.outer(np.sin(u), np.sin(v))
    z_sphere = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.3, label='Earth')

    ax.plot(x, y, z, label='Satellite Trajectory')
    ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=50, label='Start Position')
    ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=50, label='End Position')
    
    #Set the aspect of the grid to be similar to data displayed
    sx = max(np.ptp(x), 2*r_earth)
    sy = max(np.ptp(y), 2*r_earth)
    sz = max(np.ptp(z), 2*r_earth)
    ax.set_box_aspect((sx,sy,sz))


    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Z Position (km)')
    ax.set_title('Satellite Orbit Trajectory')
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


