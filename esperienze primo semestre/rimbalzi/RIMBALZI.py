import wave


import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit



t=np.array([2.99147, 3.82479, 4.51646, 5.09436, 5.58020, 5.99475, 6.34125, 6.63485])
sigma_t= 0.004

dt=np.diff(t)

n= np.arange(len(dt)) + 1

h= 9.81*(dt**2.0)/8.0 #metri
dh=2.0 * np.sqrt(2.0) * h * sigma_t / dt

def expo(n, h0, gamma):
    return h0 * gamma**n


plt.figure('Altezza dei rimbalzi')
plt.errorbar(n, h, dh, fmt='.', color='red')
popt, pcov = curve_fit(expo, n, h, sigma=dh)
h0_hat, gamma_hat = popt
sigma_h0, sigma_gamma = np.sqrt(pcov.diagonal())

print(h0_hat, sigma_h0, gamma_hat, sigma_gamma)

x=np.linspace(0.0, 8.0, 100)
plt.plot(x, expo(x, h0_hat, gamma_hat))
plt.yscale('log')
plt.grid(which='both', ls='dashed', color='grey')
plt.xlabel('Rimbalzo')
plt.ylabel('altezza massima [m]')

#residui
plt.figure('Residui')
res = h-expo(n, h0_hat, gamma_hat)
plt.errorbar(n, res, dh, fmt='.', color='red')
plt.axhline(0, color="black")
plt.grid(which= 'both', ls='dashed', color='grey')
plt.xlabel('Rimbalzo')
plt.ylabel('Residui')
plt.show()
#1.192 0.015  0.705 0.003