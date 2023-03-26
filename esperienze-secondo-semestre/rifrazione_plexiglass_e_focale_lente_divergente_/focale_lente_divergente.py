import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


#Dati sperimentali. p e q sono i valori che effettivamente abbiano misurato. a noi interessano invece i reciproci, 
#che chiameremo inv_p e inv_q. Inoltre, convertiamo tutto in metri 
p = np.array([-6.48, -8.57, -10.08, -11.51, -12.77, -13.34, -14.23, -14.67, -15.08, -15.30])/100  #metri
q = np.array([9.66, 17.21, 18.13, 25.81, 33.59, 40.55, 44.48, 52.29, 58.00, 60.24])/100 #metri
sigma_p = np.full(p.shape, 0.50)/100 #metri
sigma_q = np.full(q.shape, 1.00)/100 #metri


inv_p= 1/p
inv_q= 1/q

sigma_inv_q= sigma_p/(p**2)
sigma_inv_p= sigma_q/(q**2)

x=-inv_p
y=inv_q
dx=sigma_inv_p
dy= sigma_inv_q
#-----------------------------------------------------------------------------------------------------------------------------
#MODELLO DI FIT LINEARE
def line(x, m, Q):
    
    return m * x + Q
    

#QUESTO è IL FIT VERO E PROPRIO.
# TRAMITE I SEGUENTI COMANDI SI POSSONO DETERMINARE I PARAMENTRI OTTIMANLI PER CUI IL MODELLO SI ADATTA AL MEGLIO CON I DATI SPERIMENTALI.
#DOPODICHE' ANDREMO A PRENDERE PLOTTARE CON LA RETTA OTTIMALE, OVVERO QUELLA AVENTE I PARAMETRI DETERMINATI DAL FIT
#  IL PRIMO ARGOMETNO DELLA FUNZIONE CURVE_FIT E' IL MODELLO CON IL QUALE SI VOLGIONO FITTARE I DATI 
#SPERIMENTALI, IL SECONDO E IL TERZO ARGOMENTO SONO L'ASCISSA X E L'ORDINATA Y DEI PUNTI SPERIMENTALI.
#POPT CONTIENE I VALORI CENTRALI DEI PARAMENTRI DEL FIT. SE SI VUOLE USARE QUESTI ULTIMI NELL'ARGOMENTO DI UNA FUNZIONE, SI PUO' 
#SCRIVERE *POPT OPPURE SCRIVERE PER ESTESO TUTTI I VALORI CENTRALI. SE SONO NECESSARI TUTTI I VALORI, OVVIAMENTE E' PIU' COMODO POPT
#ALTRRIMENTI SE SI VOLGIONO RICHIAMARE SOLO ALCUNI VASLORI CENTRALI, E' MEGLIO SCRIVERLI A MANO 
popt, pcov = curve_fit(line, x, y, sigma=dy)
m_hat, Q_hat = popt

#SULLA DIAGONALE DELLA MATRICE DI COVARIANZA CI SONO GLI ERRORI DEI PARAMETRI DEL FIT. DETERMINIAMOLI GRAZIE AL SEGUENTE COMANDO
sigma_m, sigma_Q = np.sqrt(pcov.diagonal())

#STAMPA DI CIO' CHE CI INTERESSA 
print(m_hat, sigma_m, 'potere diottrico 1/f:', Q_hat, sigma_Q)

#Procediamo alla costruzione del nostro grafico
fig=plt.figure('focale di una divergente', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
res= y - line(x, m_hat, Q_hat)

ax1.errorbar(x, y, dy, dx, fmt='.', label='punti sperimentali', color='blue')

ax1.plot(x, line(x, *popt), label='Modello di best-fit', color='orange')
# Setup the axes, grids and legend.
ax1.set_ylabel('1/q [m^-1]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
# And now the residual plot, on the bottom panel.
ax2.errorbar(x, res, dy, fmt='.', color='blue')
# This will draw a horizontal line at y=0, which is the equivalent of the best-fit
# model in the residual representation.
ax2.plot(x, np.full(x.shape, 0.0), color='orange')
# Setup the axes, grids and legend.
ax2.set_xlabel('1/p [m^-1]')
ax2.set_ylabel('Residui [m^-1]')
ax2.grid(color='lightgray', ls='dashed')

# The final touch to main canvas :-)
plt.ylim(-2.0, 2.0)
fig.align_ylabels((ax1, ax2))

#test del chi squared
chisq = np.sum((((y - line(x, *popt)) / dy)**2))
print(f'Chi quadro = {chisq :.1f}')
X= np.sqrt(16)
print('Chisq atteso', 8, '+/-', X)
plt.show()