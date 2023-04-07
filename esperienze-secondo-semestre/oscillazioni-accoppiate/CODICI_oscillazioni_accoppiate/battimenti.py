import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Import dei dati del pednolo A e del pendolo B
xA, yA, xB, yB = np.loadtxt(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\oscillazioni-accoppiate\DATI_oscilalzioni_accoppiate\oscillazioni_battimenti2_MarcoNico.txt', unpack=True)

#scartiamo i primi dati, dal momento che i pendoli sono stati tenuti fermi qualche istante prima di essere lasciati liberi di oscillare 
xA=xA[44:1494] 
yA=yA[44:1494]
xB=xB[18:1494] 
yB=yB[18:1494]
dxA=np.full(xA.shape, 0.001) #secondi
dyA= np.full(yA.shape, 1)#unità arbitrarie
dxB=np.full(xB.shape, 0.001) #secondi
dyB=np.full(yB.shape, 1)#unità arbitrarie

#Modello matematico dei battimenti 
def battimenti(t, a0, T, w1, f1, w2, f2, k):
    return 2*a0*np.exp(-t/T)*np.cos(w1*t + f1)*np.cos(w2*t + f2) + k  

#Fit pendolo A
pA=[77.5, 50, 0.09, 1.5, 4.33, 3., 413]
poptA, pcovA= curve_fit(battimenti, xA, yA, sigma= dyA, p0=pA, maxfev=1000000000)
a0_hatA, T_hatA, w1_hatA, f1_hatA, w2_hatA, f2_hatA, k_hatA = poptA
da0A, dTA, dw1A, df1A, dw2A, df2A, dkA = np.sqrt(pcovA.diagonal())

#chiquadro A
XA=np.sqrt(2*1443)
chisqA= np.sum((((yA - battimenti(xA, *poptA))/dyA)**2))
print(f'Chi quadro A = {chisqA :.1f}')
print('Chisq atteso A', 1443, '\pm', XA)

#stampa dei paramentri stimati dal fit A
print('ampiezza delle oscillazioni A', a0_hatA, '\pm', da0A)
print('tempo di decadimento A', T_hatA, '\pm', dTA)
print('pulasione modulante A', w1_hatA, '\pm', dw1A)
print('fase della modulazione A', f1_hatA, '\pm', df1A)
print('pulazione portante A', w2_hatA, '\pm', dw2A)
print('fase portante A', f2_hatA, '\pm', df2A)
print('costante di traslazione A', k_hatA, '\pm', dkA)

#residui normalizzati A
resA= (yA- battimenti(xA, *poptA))/dyA
#plot del grafico A
figA=plt.figure('battimenti pendolo A')
ax1, ax2 = figA.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(xA[::3], yA[::3], dyA[::3], dxA[::3], fmt='.', label='Dati', color='orangered')
ax1.plot(xA, battimenti(xA, *poptA), label='Modello di best-fit, ', color='darkblue')
ax1.set_ylabel('Ampiezza [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
ax2.errorbar(xA[::3], resA[::3], dyA[::3], fmt='.', color='orangered')
ax2.plot(xA, np.full(xA.shape, 0.0), color='darkblue')
ax2.set_xlabel('tempo [s]')
ax2.set_ylabel('Residui normalizzati [a. u.]')
ax2.grid(color='lightgray', ls='dashed')
plt.ylim(-12, 12)
figA.align_ylabels((ax1, ax2))


#fit pendolo B
pB=[107.5, 50, 0.1, 0., 4.48, 3.1, 526.5]
poptB, pcovB= curve_fit(battimenti, xB, yB, sigma=dyB, p0=pB, maxfev=1000000000)
a0_hatB, T_hatB, w1_hatB, f1_hatB, w2_hatB, f2_hatB, k_hatB = poptB
da0B, dTB, dw1B, df1B, dw2B, df2B, dkB, = np.sqrt(pcovB.diagonal())

#print dei parametri di best-fit pednolo B
print('ampiezza delle oscillazioni B', a0_hatB, '\pm', da0B)
print('tempo di decadimento B', T_hatB, '\pm', dTB)
print('pulasione modulante B', w1_hatB, '\pm', dw1B)
print('fase della modulazione B', f1_hatB, '\pm', df1B)
print('pulazione portante B', w2_hatB, '\pm', dw2B)
print('fase portante B', f2_hatB, '\pm', df2B)
print('costante di traslazione B', k_hatB, '\pm', dkB)

#residui normalizzati pendolo B
resB = (yB - battimenti(xB, *poptB))/dyB

#grafico pendolo B
figB=plt.figure('battimenti pendolo B')
ax1, ax2 = figB.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(xB[::3], yB[::3], dyB[::3], dxB[::3], fmt='.', label='Dati', color='gold')
ax1.plot(xB, battimenti(xB, *poptB), label='Modello di best-fit, ', color='darkblue')
ax1.set_ylabel('Ampiezza [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
ax2.errorbar(xB[::3], resB[::3], dyB[::3], fmt='.', color='gold')
ax2.plot(xB, np.full(xB.shape, 0.0), color='darkblue')
ax2.set_xlabel('tempo [s]')
ax2.set_ylabel('Residui normalizzati [a. u.]')
ax2.grid(color='lightgray', ls='dashed')
plt.ylim(-15, 15)
figB.align_ylabels((ax1, ax2))

#chusq pendolo B
XB= np.sqrt(2*1469)
chisqB= np.sum((((yB - battimenti(xB, *poptB))/dyB)**2))
print(f'Chisq B = {chisqB:.1f}')
print('Chisq atteso B', 1469, '\pm', XB)

#plt.show()