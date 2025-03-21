import pandas as pd
import numpy as np
import os

class Gravity:
    def __init__(self):
        self.harmonics = self.load_egm_to_numpy()
        
    def load_egm_to_numpy(self,filename="EGM2008NM1000.csv"):
        """
        Loads EGM data from a whitespace-separated CSV file into a NumPy array.

        Args:
            filename (str): The name of the CSV file (without the 'data/' prefix).

        Returns:
            numpy.ndarray: A structured NumPy array with column names 'n', 'm', 'C', 'S', 'Cerr', 'Serr'.
                        Returns None if the file is not found.

        """

        filepath = os.path.join("data", filename)  # Construct full path, os independent
        # Custom converter to handle werid 'D' notation, otherwise numpy gets mad
        def d_to_e(s):
            try:
                return float(s.replace('D', 'E'))
            except ValueError:
                return np.nan  # Handle cases where conversion fails

        try:
            # Define data types coefficients
            dtype = 'float64'

            # Load the data using genfromtxt, handling whitespace
            data = np.genfromtxt(
            filepath,
            dtype=dtype,
            delimiter=None,       # Use None for whitespace
            skip_header=0,          # No header rows
            converters={
                2: d_to_e,  # Apply to column C (index 2)
                3: d_to_e,  # Apply to column S (index 3)
                4: d_to_e,  # Apply to column Cerr (index 4)
                5: d_to_e   # Apply to column Serr (index 5)
            },
            usecols=(2, 3), # only return C and S, errors and indecies arent needed
            encoding='utf-8' # Explicitly set encoding, helpful for unexpected characters

            )

            return data

        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None  
    def coefficients(self, n,m):
        '''
        Helper function to return [Cnm, Snm] for a given n, m
        Probably more performant to index directly into harmonics for our summation, think before using
        '''
        assert m<=n and m>= 0 and n>= 2 #make sure that index is valid, doesnt check for indecies above max
    
        index = (n * (n + 1)) // 2 - 3 + m # equation for row of n, m
        
        return self.harmonics[index]  
    
if __name__ == '__main__':
    ag = Gravity()
    print(ag.coefficients(483, 18))
    
