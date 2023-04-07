import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#import dei dati
x, y =np.loadtxt(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\oscillazioni-accoppiate\DATI_oscilalzioni_accoppiate\oscillazionilibere1_MarcoNico.txt', usecols=(2, 3), unpack=True)

#scartiamo i primi dati, in cui il pendolo era fermo
x=x[20:1893]
y=y[20:1893]
dx=np.full(x.shape, 0.001) #secondi
dy=np.full(y.shape, 1) #unità arbitrarie

#modello
def f(t, a0, tau, w, fi, k):
    return a0*np.exp(-t/tau)*np.cos(w*t + fi) + k

#pguess
p=[200, 37, 4.43, 0., 476]

#fit dei dati
popt, pcov= curve_fit(f, x, y, p0=p, sigma=dy, maxfev=1000000000)
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

#Periodo ideale
R=3.45/100 #raggio del cilindro, ovvero il diametro diviso due. Misura portata in metri
dR= 0.05/100 #errore:risoluzione del metro a nastro diivisa per due. Misura in metri
L= 48.0/100 #lunghezza dell'asta rigida omogenea. Misura in metri
dL= 0.1/100 #errore sulla lunghezza dell'asta in metri
l=L+R #distanza del pendolo, cioè dal punto di sospensione al centro di massa del cilindro. Misura in metri
dl= np.sqrt(dR**2 + dL**2)

I=((R**2)/2) + l**2 #Momento d'inerzia del cilindro. kg m^2
dI=np.sqrt((R*dR)**2 + (2*l*dl)**2)

W=np.sqrt((9.81*l)/I)
dW=np.sqrt((9.81/I*l)*dl**2 +(9.81*l/I**3)*dI**2)/2

T1_hat = (2*np.pi)/W
dT1= (2*np.pi*dW)/(W)**2

#periodo del pendolo matematico
T2_hat= 2*np.pi*np.sqrt(l/9.81)


#print dei periodi 
print('periodo', T_hat, '\pm', dT)
print('Pulsazione ideale', W, '\pm', dW)
print('Periodo ideale', T1_hat, '\pm', dT1)
print('Periodo pendolo matematico', T2_hat)
print('distanza l centro di massa - pto di sospensione', l, '\pm', dl)
print('momento d Inerzia per unità di massa', I, '\pm', dI)

#residui normalizzati e chisq
res= (y - f(x, *popt))/dy
X=np.sqrt(2*1869) 
chisq= np.sum((((y - f(x, *popt))/dy)**2))
print(f'Chi quadro = {chisq :.1f}')
print('Chisq atteso', 1869, '+/-', X)

#plot
fig = plt.figure('Pendolo_singolo_non_smorzato')
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(x[::3], y[::3], dy[::3], dx[::3], fmt='.', label='Dati', color='forestgreen')
ax1.plot(x, f(x, *popt), label='Modello di best-fit', color='lime')
ax1.set_ylabel('ampiezza [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
ax2.errorbar(x[::3], res[::3], dy[::3], fmt='.', color='forestgreen')
ax2.plot(x, np.full(x.shape, 0.0), color='lime')
ax2.set_xlabel('tempo [secondi]')
ax2.set_ylabel('Residui normalizzati [a. u.]')
ax2.grid(color='lightgray', ls='dashed')
plt.ylim(-18, 18)
fig.align_ylabels((ax1, ax2))
plt.show()
