#TODO: A drag class that uses NRLMSIS to calculate drag from date/10.7 and location
from pymsis import utils
import numpy as np
class Drag:

    solarmag = None
    dates = None
    def __init__(self):
        pass
    
    def load_f107_ap(self, start_date, end_date):
        step = np.timedelta64(3,'h') # Ap Index updated every 3 hours, F10.7 daily. use either 1 day or 3 hour step size
        # Generate date array from start to end date with specified step
        dates = np.arange(start_date, end_date, step, dtype='datetime64[s]')
        self.solarmag = np.column_stack(utils.get_f107_ap(dates))
        self.dates = dates.astype('float64') # Convert to float64 for interpolation
        print(self.solarmag)
        # NOTE: Datetime MUST be in consistent units for conversion
    
    def interpolate_index(self, date):
        time = date.astype('float64')
        solarmag = np.interp(time, self.dates, self.solarmag)
        f107, f107a, *ap = solarmag
        print(f"Interpolated Indices at {date}: F10.7={f107}, F10.7a={f107a}, Ap={ap}")
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