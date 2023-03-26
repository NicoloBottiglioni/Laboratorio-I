import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.optimize import curve_fit

#Import dei dati. Sulle ascisse i tempi e sulle ordinate le posizioni 
xA, yA = np.loadtxt(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\oscillazioni-accoppiate\DATI_oscilalzioni_accoppiate\oscillazioni_battimenti2_MarcoNico.txt', usecols=(0, 1), unpack=True)

#scarto di alcuni dati
xA = xA[36:1494]
yA = yA[36:1494]
#incertezze
dxA=np.full(xA.shape, 0.001) #secondi
dyA= np.full(yA.shape, 1)#unità arbitrarie


#Modello matematico dei battimenti e fit
def battimenti1(t, a0, T, w1, f1, w2, f2, k):
    return a0*np.exp(-t/T)*np.cos(w1*t + f1)*np.cos(w2*t + f2) + k  

p=[77, 50, 0.09, 1.57, 4.33, 0., 414]
popt, pcov= curve_fit(battimenti1, xA, yA, sigma= dyA, p0=p, maxfev= 100000)
a0_hat, T_hat, w1_hat, f1_hat, w2_hat, f2_hat, k_hat = popt
da0, dT, dw1, df1, dw2, df2, dk= np.sqrt(pcov.diagonal())

#chiquadro
X=np.sqrt(2*1451)
chisq= np.sum((((yA - battimenti1(xA, *popt))/dyA)**2))
print(f'Chi quadro = {chisq :.1f}')
print('Chisq atteso', 1451, '+/-', X)

#stampa dei paramentri stimati dal fit
print('ampiezza delle oscillazioni', a0_hat, da0)
print('tempo di decadimento', T_hat, dT)
print('pulasione modulante', w1_hat, dw1)
print('fase della modulazione', f1_hat, df1)
print('pulazione portante', w2_hat, dw2)
print('fase portante', f2_hat, df2)
print('costante di traslazione', k_hat, dk)

#plot del grafico
fig=plt.figure('battimenti prime colonna')
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))

#dimezziamo gli array
xA=xA[::2]
yA=yA[::2]
dxA=dxA[::2]
dyA=dyA[::2]


ax1.errorbar(xA, yA, dyA, dxA, fmt='.', label='Dati', color='red')

#xgrid = np.linspace(min(xA), max(xA), 1490)
ax1.plot(xA, battimenti1(xA, *popt), label='Modello di best-fit, ', color='blue')

ax1.set_ylabel('Ampiezza [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

res= yA- battimenti1(xA, *popt)
ax2.errorbar(xA, res, dyA, fmt='.', color='red')

ax2.plot(xA, np.full(xA.shape, 0.0), color='blue')

ax2.set_xlabel('tempo [s]')
ax2.set_ylabel('Residui [a. u.]')
ax2.grid(color='lightgray', ls='dashed')


plt.ylim(-6, 6)
fig.align_ylabels((ax1, ax2))

plt.show()
