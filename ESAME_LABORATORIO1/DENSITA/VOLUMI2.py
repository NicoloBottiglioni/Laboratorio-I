import numpy as np

#parallelepipedo

h=41.84/1000
sigma_h= 0.02/1000
l=10.00/1000
sigma_l=0.02/1000
V= h*l**2
sigma_V= V*np.sqrt((sigma_h/h)**2 + ((2*sigma_l)/l)**2)

print(V, sigma_V)