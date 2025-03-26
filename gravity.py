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
    
    def potential(self, r, theta, phi, nmax=200,mmax=200,relerror=0E-2):
        '''
        Function to find the potential at a given (r, Theta, Phi) in Earth-fixed coordinates
        Inputs:
        (r, theta, phi) -> Distance from CoM [km] , Noth Polar Angle [0-pi], Angle from Prime Meridian [0-2pi]
        nmax, mmax -> max order of terms considered
        error -> Max relative error acceptable. If 0, use only nmax, mmax
        Outputs:
        (Potential energy at point [km^2 s^-2 or MJ/kg], max_order, number of terms)
        '''
        
        rnorm=self.r/r #normalize r with respect to reference radius
        
        legendre = sp.special.sph_legendre_p_all(nmax,mmax,theta)[0] # Calculate all normalized spherical legendre polynomials up to nmax, mmax
        # ^ looks like P(cos(theta)), indexed as legendre[n,m]
        # output of sph_legendre_p_all is soemthing like [i, n+1, 2 *m+1]
        # i specifies derivative order, see scipy docs, but the [0] only returns the normal value
        
        
        oldun = 0 # Previous approximation up to last n
        newu = 1  #start normalized potential at 1
        
        n = 2 #starting values for N, M
        m=0

        i = 0 #simple counter variable for indexing 
        
        
       
        # while iterative error is greater than accepted and n is below limits
        while not(m==0 and abs(newu - oldun) <= relerror * newu) and n<=nmax:
            #print(abs(newu-oldu)/newu)
            if m==0:
                oldun = newu # set old approxiation to current best up to (n,m=n)
            
            #Add next term, (R/r)^n * P(cos(theta))* (C cos(mphi) + S sin(mphi))
            newu += rnorm**n * legendre[n,m] * (self.harmonics[i, 0] * np.cos(m * phi) + self.harmonics[i, 1] * np.sin(m*phi))

            #Index terms appropriately
            i+=1
            
            #sneaky bool code, if statements are slow
            m_increment = (m<n) and m<mmax # bool flag controlling whether to increment m or n, 0 or 1
            m = (m+m_increment) * (1- (not m_increment)) # add 1 to m or set it to 0
            n += (not m_increment) # Increment n only if n=m or m=mmax
            
            
        
        return (-self.mu / r * newu, n-1, 100 * abs(newu - oldun)/newu) # return (value, max order , Estimated % error)
            
    
if __name__ == '__main__':
    ag = Gravity()
    #print(ag.coefficients(207, 1))
    #print(ag.potential(7000.1363, 100, 100, relerror=0))
    #print(ag.potential(10000.1363, 100, 100, relerror=0.0001E-2))
    
    r = ag.r
    ref = -ag.mu / r
    
    
    theta = np.linspace(0,np.pi, 100)
    phi = np.radians(-81.031723)
    vec_func = np.vectorize(ag.potential)
    ref1 = vec_func(r, theta, phi, nmax=150)[0]
    ref2 = vec_func(r, theta, phi+np.pi, nmax=150)[0]
    
    u1 = vec_func(r, theta, phi, nmax=2)[0]
    u2 = vec_func(r, theta, phi+np.pi, nmax=2)[0]
    
    u3 = vec_func(r, theta, phi, nmax=4)[0]
    u4 = vec_func(r, theta, phi+np.pi, nmax=4)[0]
    
    u5 = vec_func(r, theta, phi, nmax=10)[0]
    u6 = vec_func(r, theta, phi+np.pi, nmax=10)[0]
    
    u7 = vec_func(r, theta, phi, nmax=50)[0]
    u8 = vec_func(r, theta, phi+np.pi, nmax=50)[0]
    
    
    
   
    # Create the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})  # Create a polar subplot
    
    plt.plot(np.radians(90-29.218103), 0, marker='o', markersize=10, color='black', label='Daytona Beach')  # Add the marker
    
    ax.plot(theta, np.zeros_like(theta),color='grey',linestyle='-', label='Reference [n=m=150]')  # Plot theta vs. potential
    ax.plot(-theta, np.zeros_like(theta), color='grey',linestyle='-')  # Plot theta vs. potential
    
    ax.plot(theta, np.arcsinh(ref-ref1),color='green', label='Point Mass')  # Plot theta vs. potential
    ax.plot(-theta, np.arcsinh(ref-ref2), color='green')  # Plot theta vs. potential
    
    ax.plot(theta, np.arcsinh(u1-ref1),color='blue', label='J2')  # Plot theta vs. potential
    ax.plot(-theta, np.arcsinh(u2-ref2), color='blue')  # Plot theta vs. potential
    
    
    
    ax.plot(theta, np.arcsinh(u3-ref1),color='red', label='J4')  # Plot theta vs. potential
    ax.plot(-theta, np.arcsinh(u4-ref2), color='red')  # Plot theta vs. potential
    
    ax.plot(theta, np.arcsinh(u5-ref1),color='orange', label='N=10')  # Plot theta vs. potential
    ax.plot(-theta, np.arcsinh(u6-ref2), color='orange')  # Plot theta vs. potential
    
    ax.plot(theta, np.arcsinh(u7-ref1),color='magenta', label='N=50')  # Plot theta vs. potential
    ax.plot(-theta, np.arcsinh(u8-ref2), color='magenta')  # Plot theta vs. potential
    
    
    
    
    ax.set_theta_direction(-1)  # Ensure theta increases counterclockwise
    ax.set_theta_offset(np.pi / 2)  # Start theta at the top (North Pole)
    ax.set_title(f"asin(Geopotential error) vs. Colatitude (r={r}, phi={np.degrees(phi):.2f} deg)")
    ax.set_xlabel("Colatitude (radians)")
    ax.set_ylabel("asin(Geopotential error)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.75)
    plt.show()
    
