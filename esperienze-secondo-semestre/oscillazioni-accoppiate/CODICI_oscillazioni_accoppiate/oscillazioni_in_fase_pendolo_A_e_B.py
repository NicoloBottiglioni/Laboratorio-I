import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#dati pendolo in fase A e pendolo in fase B
xA, yA, xB, yB =np.loadtxt(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\oscillazioni-accoppiate\DATI_oscilalzioni_accoppiate\oscillazionimolla_fase_marcoNico.txt', unpack=True)

#scartiamo i primi dati, dato che i pendoli sono stati tenuti fermi dopo aver avviato il programma di acquisizione, e anche
#gi ultimi dati, dato che uno dei due pendoli si era fermato èrima dell'altro per qualche arcano motivo.
xA=xA[20:1561]
yA=yA[20:1561]
xB=xB[20:1561]
yB=yB[20:1561]
dxA= np.full(xA.shape, 0.001)#secondi
dyA=np.full(yA.shape, 1)#ua
dxB= np.full(xB.shape, 0.001)#secondi
dyB=np.full(yB.shape, 1)#ua

#modello di fit
def phases(t, a0, T, w, fi, k):
    return a0*np.exp(-t/T)*np.cos(w*t + fi) + k


#Fit pednolo A
pA=[120, 40, 4.485, 0., 410]

poptA, pcovA= curve_fit(phases, xA, yA, p0=pA, sigma=dyA)
a0_hatA, T_hatA, w_hatA, fi_hatA, k_hatA = poptA
da0A, dTA, dwA, dfiA, dkA = np.sqrt(pcovA.diagonal())

#Parametri di best-fit
print('ampiezza delle oscillazioni A',  a0_hatA, '\pm', da0A)
print('Tempo di decadimento A', T_hatA, '\pm', dTA)
print('Pulsazione della fase A', w_hatA, '\pm', dwA)
print('Sfasamento A', fi_hatA, '\pm', dfiA)
print('Costante di traslazione A', k_hatA, '\pm', dkA)


#residui normalizzati pendolo A
resA = (yA - phases(xA, *poptA))/dyA

#creazione plot e grafico dei residui pendolo A
figA = plt.figure('Oscillazioni_in_fase_pendolo_A')
ax1, ax2 = figA.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(xA[::3], yA[::3], dyA[::3], dxA[::3], fmt='.', label='Dati', color='red')


ax1.plot(xA, phases(xA, *poptA), label='Modello di best-fit', color='dodgerblue')
ax1.set_ylabel('ampiezza [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
ax2.errorbar(xA[::3], resA[::3], dyA[::3], fmt='.', color='red')
ax2.plot(xA, np.full(xA.shape, 0.0), color='dodgerblue')
ax2.set_xlabel('tempo [secondi]')
ax2.set_ylabel('Residui normalizzati [a. u.]')
ax2.grid(color='lightgray', ls='dashed')
plt.xlim(0.0, 80)
figA.align_ylabels((ax1, ax2))

#Chisquared A
XA=np.sqrt(2*1536) 
chisqA= np.sum((((yA - phases(xA, *poptA))/dyA)**2))
print(f'Chi quadroA = {chisqA :.1f}')
print('Chisq attesoA', 1536, '+/-', XA)


#Fit pendolo B

pB=[120, 40, 4.485, 0., 410]

poptB, pcovB= curve_fit(phases, xB, yB, p0=pB, sigma=dyB)
a0_hatB, T_hatB, w_hatB, fi_hatB, k_hatB = poptB
da0B, dTB, dwB, dfiB, dkB = np.sqrt(pcovB.diagonal())

#Parametri di best-fit B
print('ampiezza delle oscillazioni B',  a0_hatB, '\pm', da0B)
print('Tempo di decadimento B', T_hatB, '\pm', dTB)
print('Pulsazione della fase B', w_hatB, '\pm', dwB)
print('Sfasamento B', fi_hatB, '\pm', dfiB)
print('Costante di traslazione B', k_hatB, '\pm', dkB)


#residui normalizzati
resB = (yB - phases(xB, *poptB))/dyB


#creazione plot e grafico dei residui
figB = plt.figure('Oscillazioni_in_fase_pendolo_B')
ax1, ax2 = figB.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(xB[::3], yB[::3], dyB[::3], dxB[::3], fmt='.', label='Dati', color='darkorange')


ax1.plot(xB, phases(xB, *poptB), label='Modello di best-fit', color='dodgerblue')
ax1.set_ylabel('ampiezza [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
ax2.errorbar(xB[::3], resB[::3], dyB[::3], fmt='.', color='darkorange')
ax2.plot(xB, np.full(xB.shape, 0.0), color='dodgerblue')
ax2.set_xlabel('tempo [secondi]')
ax2.set_ylabel('Residui normalizzati [a. u.]')
ax2.grid(color='lightgray', ls='dashed')
plt.xlim(0.0, 80)
figB.align_ylabels((ax1, ax2))

#Chisquared
XB=np.sqrt(2*1536) 
chisqB= np.sum((((yB - phases(xB, *poptB))/dyB)**2))
print(f'Chi quadroB = {chisqB :.1f}')
print('Chisq attesoB', 1536, '+/-', XB)

plt.show()

