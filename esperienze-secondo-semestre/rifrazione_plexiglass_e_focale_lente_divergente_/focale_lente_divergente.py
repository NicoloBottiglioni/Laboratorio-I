import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


#Dati sperimentali. p e q sono i valori che effettivamente abbiano misurato. a noi interessano invece i reciproci, 
#che chiameremo inv_p e inv_q
p = np.array([-10.1, -12.1, -7.2, -5.3, -4.9, -8.2, -6.6, -9.0, -11.4, -5.6 ])  #cm
q = np.array([19.6, 30.5, 17.05, 9.9, 7.55, 18.55, 12.75, 22.1, 29.25, 10.5 ])
sigma_p = np.full(p.shape, 0.5) #cm
sigma_q = np.full(q.shape, 2.09)


#2.0, 2.8, 3.35, 2.0, 1.35, 1.65, 3.05, 1.9, 2.25, 0.6, 2.1////////scartata: p-6.0  q 7.8,
#------------------------------------------------------------------------------------------------------------------------------
inv_p= 1/p
inv_q= 1/q

sigma_inv_q= sigma_p/(p**2)
sigma_inv_p= sigma_q/(q**2)
#------------------------------------------------------------------------------------------------------------------------------
'''for i in range(10):
    inv_p.append(1/p)

for i in range(10):
    inv_q.append(1/q)

for i in range(10):
    sp = sigma_p / ((p([i])) **2.0)
    sigma_inv_p.append(sp)

for i in range(10):
    sq = sigma_q / ((p([i])) **2.0)
    sigma_inv_q.append(sq)'''

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

#STAMPA DI CIO' CHE CI INTERESSA: RICORDARE CHE NOI VOGLIAMO LA DISTANZA FOCALE F, quindi poi domani bisogna modificare questo punto. 
print(m_hat, sigma_m, 'potere diottrico 1/f:', Q_hat, sigma_Q)


fig=plt.figure('Rifrazione Plexiglass', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
res= y - line(x, m_hat, Q_hat)

ax1.errorbar(x, y, dy, dx, fmt='.', label='punti sperimentali')
#xgrid = np.linspace(0.0, 10.0, 100)
ax1.plot(x, line(x, *popt), label='Modello di best-fit')
# Setup the axes, grids and legend.
ax1.set_ylabel('y [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
# And now the residual plot, on the bottom panel.
ax2.errorbar(x, res, dy, fmt='.')
# This will draw a horizontal line at y=0, which is the equivalent of the best-fit
# model in the residual representation.
ax2.plot(x, np.full(x.shape, 0.0))
# Setup the axes, grids and legend.
ax2.set_xlabel('x [a. u.]')
ax2.set_ylabel('Residuals [a. u.]')
ax2.grid(color='lightgray', ls='dashed')

# The final touch to main canvas :-)
plt.ylim(-0.06, 0.06)
fig.align_ylabels((ax1, ax2))

chisq = np.sum((((y - line(x, *popt)) / dy)**2))
print(f'Chi quadro = {chisq :.1f}')
X= np.sqrt(16)
print('Chisq atteso', 8, '+/-', X)
plt.show()