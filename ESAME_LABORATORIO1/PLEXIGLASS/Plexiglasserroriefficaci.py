import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#dati sperimentali 
x=np.array([]) #quadretti
y=np.array([]) #quadretti
dx= np.full(x.shape, ) #quadretti
dy= np.full(y.shape, ) #quadretii

#modello e fit con gli errori efficaci
def line(x, m, q):
    return m*x +q

popt, pcov= curve_fit(line, x, y, sigma=dy, absolute_sigma= True)
for i in range(4):
    sigma_eff = np.sqrt(dy**2.0 + (popt[0] * dx)**2.0)
    popt, pcov = curve_fit(line, x, y, sigma=sigma_eff, absolute_sigma=True)
    chisq = (((y - line(x, *popt)) / sigma_eff)**2.0).sum()
    print(f'Step {i}...')
    print(popt, np.sqrt(pcov.diagonal()))
    print(f'Chisquare = {chisq:.2f}')


#Grafico e grafico dei residui
fig=plt.figure('poterediottrico_errori efficaci', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(x, y, dy, dx, fmt='.', label='punti sperimentali', color='midnightblue')
ax1.plot(x, line(x, *popt), label='Modello di best-fit', color='deepskyblue')
ax1.set_ylabel('1/q [m^(-1)]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

res= (y - line(x, *popt)) / dy
ax2.errorbar(x, res, dy, fmt='.', color='midnightblue')
ax2.plot(x, np.full(x.shape, 0.0), color='deepskyblue')
ax2.set_xlabel('1/p [m^(-1)]')
ax2.set_ylabel('Residu normalizzati')
ax2.grid(color='lightgray', ls='dashed')

plt.ylim(-1.8, 1.8)
fig.align_ylabels((ax1, ax2))
plt.show()
