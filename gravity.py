import pandas as pd
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt

class Gravity:
    def __init__(self):
        self.harmonics = self.load_egm_to_numpy()
        self.mu = 398600.4415 #km^3 s^-2, From Pavlis et al. 2008
        self.r = 6378.1363 #km, Reference radius from EGM2008
        
    def load_egm_to_numpy(self,filename="EGM2008NM1000.csv"):
        """
        Loads EGM data from a whitespace-separated CSV file into a NumPy array.

        Args:
            filename (str): The name of the CSV file (without the 'data/' prefix).

        Returns:
            numpy.ndarray: A structured NumPy array with columns [Cnm, Snm] with 64bit floats

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
    
    def potential(self, r, theta, phi, nmax=200, mmax=200, relerror=0E-2):
        """
        Calculate the gravitational potential at a given point in Earth-fixed coordinates

        Parameters:
        r (float): Distance from the center of mass (CoM) in kilometers.
        theta (float): North polar angle in radians (0 to pi).
        phi (float): Angle from the prime meridian in radians (0 to 2*pi).
        nmax (int): Maximum order of terms to consider in the summation.
        mmax (int): Maximum degree of terms to consider in the summation.
        relerror (float): Maximum relative error acceptable. If 0, only nmax and mmax are used.

        Returns:
        tuple: A tuple containing:
            - Potential energy at the point in km^2/s^2 or MJ/kg.
            - Maximum order of terms considered.
            - Estimated percentage error of the approximation.
        """
        
        # Normalize the distance with respect to the reference radius
        rnorm = self.r / r
        
        # Calculate all normalized spherical Legendre polynomials up to nmax and mmax
        legendre = sp.special.sph_legendre_p_all(nmax, mmax, theta)[0]
        # legendre[n, m] represents P(cos(theta)), where n is the order and m is the degree
        
        # Initialize previous and current approximations of the potential
        oldun = 0  # Previous approximation up to the last n
        newu = 1   # Start normalized potential at 1
        
        # Initialize indices for the summation
        n = 2  # Starting order
        m = 0  # Starting degree
        i = 0  # Index for accessing harmonics
        
        # Iterate until the relative error is within the acceptable limit or n reaches nmax
        while not (m == 0 and abs(newu - oldun) <= relerror * newu) and n <= nmax:
            # Update the previous approximation when m is 0
            if m == 0:
                oldun = newu
            
            # Add the next term to the potential
            newu += rnorm**n * legendre[n, m] * (self.harmonics[i, 0] * np.cos(m * phi) + self.harmonics[i, 1] * np.sin(m * phi))
            
            # Increment the index for harmonics
            i += 1
            
            # Determine whether to increment m or n
            m_increment = (m < n) and m < mmax  # Boolean flag to control incrementing m or n
            m = (m + m_increment) * (1 - (not m_increment))  # Increment m or reset to 0
            n += (not m_increment)  # Increment n only if m reaches n or mmax
        
        # Return the potential energy, maximum order, and estimated percentage error
        return (-self.mu / r * newu, n - 1, 100 * abs(newu - oldun) / newu)
    def ellipsoid_distance(self, lat_rad, h, ao=6378.1370, f=1/298.257223563):
        """
        Helper function to converts geodetic point coordinates (latitude, longitude, height) to
        geocentric radius of point on WGS84 ellipsoid. 

        Args:
            lat_deg: Geodetic latitude in radians.
            h: Height above the ellipsoid in kilometers.
            ao: Equatorial Radius of the ellipsoid (default: WGS84)
            f: Flattening of the ellipsoid (default: WGS84).

        Returns:
            The distance r from the centroid
        """
       
        a = ao + h
        b = a * (1 - f)
        c = np.cos(lat_rad)
        s= np.sin(lat_rad)
        
        r = np.sqrt(((a**2*c)**2+(b**2*s)**2)/((a*c)**2 + (b*s)**2))

        return r
    def plot_potential(self, altitude,nmax=10, num_points=200): # Increased num_points for better resolution
        """
        Plot the gravitational potential anomaly with respect to J2 on a map at a specified altitude.
        """
        latitude = np.linspace(-np.pi/2, np.pi/2, num_points)
        longitude = np.linspace(-np.pi, np.pi, num_points)
        lon_grid, lat_grid = np.meshgrid(longitude, latitude)

        # Convert latitude to colatitude (theta)
        theta_grid = np.pi/2 - lat_grid  # theta = pi/2 - latitude
        phi_grid = lon_grid  # longitude = phi

        r = self.r + altitude

        # Calculate potential on the grid
        potential_grid = np.zeros_like(lat_grid)
        potential_rid = np.zeros_like(lat_grid)
        for i in range(num_points):
            for j in range(num_points):
                potential_grid[i, j] = self.potential(r, theta_grid[i, j], phi_grid[i, j], nmax=nmax)[0]
                potential_rid[i,j] =self.potential(r, theta_grid[i, j], phi_grid[i, j], nmax=2)[0]
                #potential_rid[i,j] =self.potential(r, theta_grid[i, j], phi_grid[i, j], nmax=0)[0]


        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'aitoff'})

        # Create a filled mesh plot
        im = ax.pcolormesh(lon_grid, lat_grid, potential_grid-potential_rid, cmap='viridis') # Adjust levels as needed

        # Add a colorbar
        fig.colorbar(im, ax=ax, label='Gravitational Potential Anomaly J{}-J2 (km^2/s^2)'.format(nmax))

        # Daytona Beach coordinates
        lat_daytona = np.radians(29.218103)
        lon_daytona = np.radians(-81.031723)

        # Plot Daytona Beach point
        ax.scatter(lon_daytona, lat_daytona, color='red', s=100, label='Daytona Beach')

        ax.set_xlabel('Longitude (rad)')
        ax.set_ylabel('Latitude (rad)')
        ax.set_title('Gravitational Potential Anomaly Map at {} km Altitude'.format(altitude))
        ax.grid(True)
        
        plt.show()
    #TODO make a function that returns a vector [ar, atheta, aphi] in Fixed Geocentric coordinates.  

    
if __name__ == '__main__':
    ag = Gravity()
    #print(ag.coefficients(207, 1))
    #print(ag.potential(7000.1363, 100, 100, relerror=0))
    #print(ag.potential(10000.1363, 100, 100, relerror=0.0001E-2))
    
    r = ag.r
    ref = -ag.mu / r
    theta_daytona = np.radians(90-29.218103)
    phi_daytona = np.radians(-81.031723)
    
    theta = np.linspace(0,np.pi, 100)
    
    
    ag.plot_potential(50, nmax=10)
    

