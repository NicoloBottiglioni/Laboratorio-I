import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#import dei dati sperimentali
x, y = np.loadtxt(r"C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\rifrazione_plexiglass_e_focale_lente_divergente_\Dati_rifrazione_plexiglass.txt", unpack=True)
dx= np.full(x.shape, 1) #quadretti
dy= np.full(y.shape, 1) #quadretii

#Modello lineare di fit
def line(x, n, q):
    return n*x +q

popt, pcov= curve_fit(line, x, y, sigma= dy)


n_hat, q_hat = popt
dn, dq = np.sqrt(pcov.diagonal())

#l'indice di rifrazione del plexiglass è il reciproco del parametro restituito dal fit
na_hat= 1/n_hat
dna= dn/((n_hat)**2)


print('INDICE DI RIFRAZIONE:', na_hat, dna, 'INTERCETTA:', q_hat, dq)
err_rel_n = dna/na_hat
print('ERRORE RELATIVO INDICE RIFRAZIONE:', err_rel_n)


#residui normalizzati
res= (y - line(x, n_hat, q_hat))/dy



#chisquared e suo valore atteso
chisq = np.sum((((y - line(x, *popt)) / dy)**2))
print(f'Chi quadro = {chisq :.1f}')
X= np.sqrt(16)
print('Chisq atteso', 8, '+/-', X)

#controlliamo se l'erroe sulla x è trascurabile
ss= n_hat*dx 
print(ss)





fig=plt.figure('Rifrazione Plexiglass', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))


ax1.errorbar(x, y, dy, dx, fmt='.', label='punti sperimentali')
#xgrid = np.linspace(0.0, 10.0, 100)
ax1.plot(x, line(x, *popt), label='Modello di best-fit')
# Setup the axes, grids and legend.
ax1.set_ylabel('Distanza asse - raggio incidente [quadretti]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
# And now the residual plot, on the bottom panel.
ax2.errorbar(x, res, dy, fmt='.')
# This will draw a horizontal line at y=0, which is the equivalent of the best-fit
# model in the residual representation.
ax2.plot(x, np.full(x.shape, 0.0))
# Setup the axes, grids and legend.
ax2.set_xlabel('Distanza asse - raggio rifratto [quadretti]')
ax2.set_ylabel('Residui [quadretti.]')
ax2.grid(color='lightgray', ls='dashed')

# The final touch to main canvas :-)
plt.ylim(-2.1, 3.0)
fig.align_ylabels((ax1, ax2))
plt.show()