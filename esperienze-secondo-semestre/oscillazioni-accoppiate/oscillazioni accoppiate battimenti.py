import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

t1, x1 = np.loadtxt(r"./vesperienze-secondo-semestre/dati/battimenti1.txt", usecols=(0,1),  unpack=True)
dt1=np.full(t1.shape, 0.5)
dx1=np.full(x1.shape, 0.5)

t2, x2 = np.loadtxt(r"./vesperienze-secondo-semestre/dati/battimenti1.txt", usecols=(2,3),  unpack=True)
dt2=np.full(t2.shape, 0.5)
dx2=np.full(x2.shape, 0.5)


def f(t, A, omega1, omega2, T, phi1, phi2, k):
    return A*e^(-t/T)*np.cos(omega*t + phi1)*np.cos(omega2*t + phi2) + K 
 


plt.figure('Battimenti')
plt.errorbar(t1, x1, dx1, dt1, fmt='.')
plt.errorbar(t2, x2, dx2, dt2, fmt='.')
plt.show()
'''popt, pcov = curve_fit(f, t, x, sigma = dx, p0=)
p0=()


tt = np.linspace(0., 200., 100)
plt.plot(xx, f(tt, m_hat, q_hat))


plt.xlabel('tempo [s]')
plt.ylabel('ampiezza [u.a.]')
plt.grid(which='both', ls='dashed', color='black') '''