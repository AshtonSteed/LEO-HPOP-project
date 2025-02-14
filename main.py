import numpy as np
import pandas as pd
import scipy as sp

print("Hello World 2!")


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
leo_data = pd.read_csv('~~.csv', ',')


# We need to declare all the variables needed in the equations - by applying those variables to parts of
# the numpy array:






# We need to define functions for each section, so that when we combine them all, it's less of a stress on the system,
# and is more readable for readers.







