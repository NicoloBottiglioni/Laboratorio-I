import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#import dei dati
x, y =np.loadtxt(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\oscillazioni-accoppiate\DATI_oscilalzioni_accoppiate\oscillazionismorzate2_MarcoNico.txt', usecols=(2, 3), unpack=True)

#scartiamo i primi dati, in cui il pendolo era fermo
x=x[22:1669]
y=y[22:1669]
dx=np.full(x.shape, 0.001) #secondi
dy=np.full(y.shape, 1) #unità arbitrarie

#modello
def f(t, a0, tau, w, fi, k):
    return a0*np.exp(-t/tau)*np.cos(w*t + fi) + k

#pguess
p=[134, 35., 4.63, 0., 479]

#fit dei dati
popt, pcov= curve_fit(f, x, y, p0=p, sigma=dy)
a0_hat, tau_hat, w_hat, fi_hat, k_hat = popt
da0, dtau, dw, dfi, dk = np.sqrt(pcov.diagonal())

#print parametri di best fit
print('Ampiezza di oscillazione', a0_hat, '\pm', da0)
print('tempo di decadimento', tau_hat, '\pm', dtau)
print('pulsazione', w_hat, '\pm', dw)
print('fase', fi_hat, '\pm', dfi)
print('Costante di traslzione', k_hat, '\pm', dk)

#Calcolo del periodo
T_hat = (2*np.pi)/w_hat
dT= (2*np.pi*dw)/(w_hat)**2 
print('periodo', T_hat, '\pm', dT)


#residui normalizzati e chisq
res= (y - f(x, *popt))/dy
X=np.sqrt(2*1642) 
chisq= np.sum((((y - f(x, *popt))/dy)**2))
print(f'Chi quadro = {chisq :.1f}')
print('Chisq atteso', 1642, '+/-', X)

#plot
fig = plt.figure('Pendolo_singolo_smorzato')
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(x[::3], y[::3], dy[::3], dx[::3], fmt='.', label='Dati', color='midnightblue')
ax1.plot(x, f(x, *popt), label='Modello di best-fit', color='deepskyblue')
ax1.set_ylabel('ampiezza [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
ax2.errorbar(x[::3], res[::3], dy[::3], fmt='.', color='midnightblue')
ax2.plot(x, np.full(x.shape, 0.0), color='deepskyblue')
ax2.set_xlabel('tempo [secondi]')
ax2.set_ylabel('Residui normalizzati [a. u.]')
ax2.grid(color='lightgray', ls='dashed')
plt.ylim(-12, 14)
plt.xlim(26, 28.5)
fig.align_ylabels((ax1, ax2))
plt.show()