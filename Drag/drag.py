#TODO: A drag class that uses NRLMSIS to calculate drag from date/10.7 and location
from pymsis import utils
from scipy.interpolate import (interp1d, CubicSpline)
import numpy as np
class Drag:
    # USSA-76 Constants (Class Attributes)
    # Parameters for the 1976 US Standard Atmosphere model, used for basic or preliminary density calculations.
    #KM
    _ALTS = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 
                      150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], dtype=float)
    
    #KG/M^3
    _RHOS = np.array([1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5, 1.846e-5, 
                      3.416e-6, 5.604e-7, 9.708e-8, 2.222e-8, 8.136e-9, 3.831e-9, 2.076e-9, 5.194e-10, 
                      2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12, 1.184e-12, 5.215e-13, 
                      1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15], dtype=float)
    
    _SCALES = np.array([7.31, 6.427, 6.546, 7.36, 7.714, 7.274, 6.69, 5.793, 5.382, 5.877, 7.263, 
                        9.473, 12.636, 16.156, 22.523, 29.74, 37.943, 46.336, 53.628, 59.906, 65.508, 
                        70.813, 76.377, 86.804, 103.628, 115.149, 116.023, 268.0], dtype=float)

    solarmag = None
    dates = None
    def __init__(self):
        self._init_ussa_1976_spline()
    
    def _init_ussa_1976_spline(self):
        """
        Creates a cubic spline of ln(rho) vs altitude.
        We spline the LOG of density because density varies by orders of magnitude,
        which causes standard splines to oscillate wildly.
        """
        # Calculate Log-Density for the anchor points
        log_rhos = np.log(self._RHOS)
        
        # Create spline: h_km -> ln(rho)
        # bc_type='natural' ensures 2nd derivative is zero at ends (smooth vacuum transition)
        self.log_rho_spline = CubicSpline(self._ALTS, log_rhos, bc_type='natural', extrapolate=False)

    def get_ussa_density(self, h_km):
        # h_km: altitude in kilometers
        """
        Calculates density using cubic spline interpolation.
        Preferred for HPOP/Integrators due to C2 continuity.
        """
        # Get log density
        log_rho = self.log_rho_spline(h_km)
        # Convert back to density: rho = exp(ln(rho)) and convert to kg/km^3
        rho = np.exp(log_rho) * 1e9  # 1 m^3 = 1e-9 km^3
        # Handle vacuum/out-of-bounds (spline returns nan for extrapolation if strictly set)
        return np.nan_to_num(rho, nan=0.0)
    
    def load_f107_ap(self, start_date, end_date):
        step = np.timedelta64(3,'h') # Ap Index updated every 3 hours, F10.7 daily. use either 1 day or 3 hour step size
        # Generate date array from start to end date with specified step
        dates = np.arange(start_date, end_date, step, dtype='datetime64[s]')
        self.solarmag = np.column_stack(utils.get_f107_ap(dates))
        self.dates = dates.astype('float64') # Convert to float64 for interpolation
        
        # Create 1D interpolator for solarmag data, more efficient than doing this every single interpolation
        self.solarmag_interp = interp1d(self.dates, self.solarmag, axis=0, fill_value="extrapolate")
        #print(self.solarmag)
        # NOTE: Datetime MUST be in consistent units for conversion
    
    # Find the intpolated index for a given time
    def interpolate_index(self, date):
        time = date.astype('float64')
        solarmag = self.solarmag_interp(time)
        f107, f107a, *ap = solarmag
        print(f"Interpolated Indices at {date}: F10.7={f107}, F10.7a={f107a}, Ap={ap}")
        
        
    def drag_acceleration(self, pos_xyz, vel_xyz, date, high_fidelity=False,Cd=2.2, A=1.0e-6, m=1.0, r_earth=6378.1363):
        # pos_xyz = (x, y, z) in km
        # vel_xyz = (vx, vy, vz) in km/s
        # date = something??
        
        # Returns acceleration in km/s^2 due to drag
        
        
        # Placeholder for density calculation
        # In low precision runs, use basic_density
        # In high precision, use interpolated time model
        if not high_fidelity:
            altitude_km = np.linalg.norm(pos_xyz) - r_earth
            density = self.get_ussa_density(altitude_km)  # Earth radius in km
        else:
            # Do the high fidelity density using pymsis array
            pass
            
        
        # Earths Angular Velocity in rad/s
        omega_earth = 7.2921150e-5
        omega_vec = np.array([0, 0, omega_earth])
        
        #ramming velocity of satellite, assuming atmosphere rotates with Earth
        v_atm = np.cross(omega_vec, pos_xyz)
        v_rel = vel_xyz - v_atm
        
        # Use parameters and density to approximate drag acceleration
        a_drag = -0.5 * Cd * A / m * density * np.linalg.norm(v_rel) * v_rel
        
        return a_drag
        
    
        
if __name__ == "__main__":
    # Example usage of pymsis to get atmospheric data
    # Define parameters
    year = 2020
    day_of_year = 100
    seconds_in_day = 36000  # 10:00 AM
    lat = 45.0  # degrees
    lon = -75.0  # degrees
    alt_km = 300.0  # altitude in kilometers
    
    
    d = Drag()
    d.load_f107_ap(np.datetime64('2020-10-27T15:00:00'), np.datetime64('2020-10-28T15:00:00'))
    d.interpolate_index(np.datetime64('2020-10-27T20:30:00'))
    
    # Get atmospheric data using pymsis