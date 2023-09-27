import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''NOTE:
--
--
--
--
'''

#Dati sperimentali
#R=
l=np.array([0.6810, 0.6400, 0.603, 0.555, 0.511, 0.468]) #metri      #Aggiungerci il raggio della sfera 
dl= np.full(l.shape, 0.001) # risoluzione strumento in metri diviso radice di dodici

T= np.array([16.794, 16.256, 15.696, 15.094, 14.470, 13.740]) # misura dei 10 periodi per ogni lunghezza
t= T/10  # misura del singolo periodo per ongi lunghezza
dt= 0.001  #Risoluzione strumentale diviso per il numero di periodi misurati 

#modello
def periodo(l, theta):
    theta= 0.145  #in radianti
    return 2*np.pi*np.sqrt(l/9.81)*(1+ theta**2/16 + 11*theta**4/3072 + 173*theta**6/737280 + 22931*theta**8/1321205760)


#Verifica dei termini che posso trascurare
theta= 0.145
print(theta**2/16, 11*theta**4/3072)

#Chisq
chisq= np.sum(((t - periodo(l, theta))/dt)**2)
print('CHisq atteso', len(l), '\pm', np.sqrt(2*len(l)))
print('Chisq misurato', chisq)



#Grafico
fig=plt.figure('periodo pendolo semplice', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(l, t, dt, dl, fmt='.', label='punti sperimentali', color='midnightblue')
ax1.plot(l, periodo(l, theta), label='Modello di best-fit', color='deepskyblue')
ax1.set_ylabel('T [s]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

res= (t - periodo(l, theta)) / dt  #residui normalizzati
ax2.errorbar(l, res, dt, fmt='.', color='midnightblue')
ax2.plot(l, np.full(l.shape, 0.0), color='deepskyblue')
ax2.set_xlabel('l [m]')
ax2.set_ylabel('Residui normalizzati')
ax2.grid(color='lightgray', ls='dashed')

#plt.ylim(-1.8, 1.8)
fig.align_ylabels((ax1, ax2))
plt.show()