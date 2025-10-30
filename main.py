import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from Gravity.gravity import Gravity
from Integration_and_Conversion.state import State
import skyfield.api as sf





# Name of Function: state_update_full
# Arguments / variables used:
#   - t: Current time (unused, but required by solve_ivp signature)
#   - statevec: A numpy array representing the current state vector [rx, ry, rz, vx, vy, vz]
#   This function serves as the derivative function for scipy.integrate.solve_ivp, returning
#   the derivatives of the state vector [vx, vy, vz, ax, ay, az].
def state_update(t, statevec, g: Gravity, ts: sf.Timescale, n=0):
    
    # Initialize output array for derivatives [vx, vy, vz, ax, ay, az]
    output = np.zeros_like(statevec)
    
    # Create a State object for convinience in coordinate conversion and data pulling
    state = State(statevec, t)
    
    # Extract position and velocity vectors
    r = state.r()
    v = state.v()

    # Convert position to ITRS frame to compute acceleration due to gravity
    xyz_itrs = State.icrs_to_itrs(r, t, ts)
    # Find spherical coordinates from Cartesian
    r_norm,theta,phi = state.xyz_to_sphr(xyz_itrs)
    # Compute gravitational acceleration in spherical coordinates
    a_sphr = g.acceleration_g(r_norm, theta, phi, nmax=n)
    # Convert acceleration back Cartesian ITRS frame
    a_itrs = State.sphr_to_xyz_vec((r_norm,theta,phi),a_sphr)
    #Now convert acceleration to ICRS frame
    a = State.itrs_to_icrs(a_itrs, t, ts)
    
    output[:3] = v
    # The last three elements of the output are the acceleration components (dv/dt = a)
    output[3:] = a
    
    return output


# Name of Function: integrate
# Arguments / variables used:
#   - initial_state_vector: A numpy array representing the initial state vector [rx, ry, rz, vx, vy, vz].
#   - t_span: A tuple (t0, tf) representing the start and end times for the integration.
#   - dt: The initial step size for the integration.
#   - n: Maximum degree for gravity harmonics.
# What the intention of the function is:
#   This function numerically integrates the equations of motion using scipy.integrate.solve_ivp.
#   It takes an initial state vector and a time span, and returns the results of the integration.
def integrate(initial_state_vector, t_span, dt, g, ts, n):

    # Use scipy.integrate.solve_ivp to perform the numerical integration
    results = sp.integrate.solve_ivp(state_update, t_span, initial_state_vector,
                                     first_step=dt, rtol=1e-12, atol=1e-12, method='DOP853', args=[g, ts, n])
    return results


def main():
    print("Starting orbital simulation accuracy comparison...")

    g = Gravity()
    ts = sf.load.timescale()

    # Initial conditions for a 500km circular orbit in ICRS
    r_initial = np.array([g.radius + 500, 0, 0])  # Position vector [km]
    v_initial = np.array([0, 0, np.sqrt(g.mu / (g.radius + 500))]) # Velocity vector [km/s]
    
    #Define start and end times in UTC
    t_start_utc = ts.utc(2022, 1, 1, 0, 0, 0)
    t_end_utc = ts.utc(2022, 1, 5, 0, 0)
    #Convert to Julian Date "Physics ready" time
    t_start_db = t_start_utc.tdb * 86400  # Convert days to seconds
    t_end_db = t_end_utc.tdb * 86400      # Convert days to seconds
    
    #print(f"Simulation start time (TDB): {t_start_db}")
    #print(f"Simulation end time (TDB): {t_end_db}")
    
    # Create a State object to hold initial conditions
    initial_satellite_state = State(np.concatenate((r_initial, v_initial)), t_start_db)

    # Define simulation parameters
    orbital_period = np.sqrt(4 * np.pi**2 * np.linalg.norm(r_initial)**3 / g.mu) # seconds for one orbit
    print(f"Orbital period: {orbital_period} seconds")

    time_step = 1               # Initial step size for the solver [s]

    # Define the time span for integration (from t_initial to simulation_duration)
    t_span = (t_start_db, t_end_db)

    print("Initial position: ", initial_satellite_state.r())
    print("Initial velocity: ", initial_satellite_state.v())

    # List of N-M values to test (assuming N=M for simplicity, as per gravity model)
    n_values = [4, 8]  # Different maximum degrees

    results = {}

    for n in n_values:
        print(f"\nRunning simulation with N=M={n}...")
        start_time = time.time()

        # Integrate the equations of motion
        simulation_results = integrate(initial_satellite_state.state, t_span, time_step, g, ts, n)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Store final position for later error calculation
        final_position = simulation_results.y[:3, -1]
        #lot_orbit(simulation_results, g.radius)

        # Log results
        results[n] = {
            'time': elapsed_time,
            'final_position': final_position,
            'num_steps': len(simulation_results.t)
        }

        print(f"Simulation with N=M={n} completed in {elapsed_time:.2f} seconds.")
        print(f"Number of integration steps: {len(simulation_results.t)}")

    # Compute position errors relative to the highest accuracy simulation
    reference_n = max(n_values)
    reference_position = results[reference_n]['final_position']
    for n in n_values:
        position_error = np.linalg.norm(results[n]['final_position'] - reference_position)
        results[n]['position_error'] = position_error
        print(f"Position error relative to N=M={reference_n}: {position_error:.6f} km")

    # Print summary
    print("\nSummary of results:")
    print("N=M | Time (s) | Position Error (km) | Steps")
    print("-" * 40)
    for n, data in results.items():
        print(f"{n:3d} | {data['time']:8.2f} | {data['position_error']:17.6f} | {data['num_steps']:5d}")
        

    # Optionally plot the orbit for the highest accuracy (highest N)
    # plot_orbit(simulation_results, g.radius)


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


