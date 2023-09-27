import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''NOTE:
-- fare grafico V(m), perché in tal modo le incertezze su x sono trascurabili
--V e m sono distribuite uniformemente
--m è distribuita uniformemente nell'intervallo pari alla risoluzione della bilancia di precisione. Per quanto riguarda V, esso è 
il prodotto di variabili uniformi anchesse nell'intervallo pari alla risoluzione dello strumento utilizzato per misurarle. 
-- Le altezze misurarle con un calibro, perché i solidi sono troppo alti per essere fatti con il palmer. Con il palmer misurare 
altre grandezze, facendo attenzione. 
--Verificare che gli struenti siano calibrati: quando i calibri sono chiusi verificare che le tacche corrispondenti allo zero coincidano
e fare la stessa con il palmer, nel quale devono coincidere lo zero del cilindro mobile e la scala del cilindro fisso
-- fare attenzione alla letturea del palmer
--errore di zero della bilancia di precisione 
--Chisq: la variabile y non è distribuita gaussianamente, essendo essa il prodotto si variabili uniformi indipendenti. In generale, V
non è nemmeno distribuito uniformemente, dunque il Chisq non è distribuito come un chisq. Nonostante ciò ha comunque media pari al 
numero di gradi di libertà ma riguardo la sua varianza non sappiamo dire altro.
--L'intercetta stimata deve essere compatibile con lo zero e il coefficiente angolare è l'inverso della densità del materiale
-- CONVERTIRE E MISURE IN KG E IN METRI
---volume prisma esagonale     V=6*a**2*h*cotan(np.pi/6) sigma_V= V*np.sqrt(((2*sigma_a)/a)**2 + (sigma_h/h)**2)
--- volume cilindro  V= h*np.pi*d**2/4   sigma_V= V*np.sqrt(((2*sigma_d)/d)**2 + (sigma_h/h)**2)
--volume parallelepipedo   V= h*l*p   sigma_V= V*np.sqrt((sigma_h/h)**2 + (sigma_l/l)**2 + (sigma_p/p)**2)
-- volume prisma quadrato V= h*l**2   sigma_V= V*np.sqrt(((2*sigma_l)/l)**2 + (sigma_h/h)**2)
--volume sfera         V=np.pi*d**3/6   sigma_V=3*V*sigma_d/d'''



#ARRAY DEI DATI, CIOE' VOLUMI E MASSE 
x= np.array([22.240, 10.492, 24.722, 28.821, 16.532])/1000 #masse in g
dx=np.array([0.001])/(1000*np.sqrt(12)) #risoluzione strumentale diviso radice di 12
y=np.array([26.8, 1215.0, 2568.3, 10662.5, 6074.3 ])/(10**(-9)) #volumi in millimetri cubi
dy= np.array([0.5, 1.4, 3.0, 8.4, 7.0])/(10**(-9)) #errori dei vari volumi 

#modello di fit e fit
def line(x, m, q):
    return m*x + q

popt, pcov= curve_fit(line, x, y, sigma=dy, absolute_sigma=True)
m_hat, q_hat = popt 
dm, dq= np.sqrt(pcov.diagonal())
chisq=np.sum((((y - line(x, *popt))/dy)**2))

#densità
rho= 1/m_hat
sigma_rho= dm/(m_hat**2)

#print dei valori
print('densità =', rho, '\pm', sigma_rho, '----', 'intercetta=', q_hat, '\pm', dq)
print('CHISQ=', chisq)

#residui normalizzati
res= (y- line(x, *popt))/dy

#grafico
fig=plt.figure('Focale metodo odr', figsize=(10., 6.), dpi=100)

ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(x, y, dy, dx, fmt='.', label='punti sperimentali', color='midnightblue')
ax1.plot(x, line(x, *popt), label='Modello di best-fit', color='deepskyblue')
ax1.set_ylabel('volume [m^3]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(x, res, dy, fmt='.', color='midnightblue')
#np.linspace
ax2.plot(x, np.full(x.shape, 0.0), color='deepskyblue')
ax2.set_xlabel('massa [kg]')
ax2.set_ylabel('Residui [m^(-1)]')
ax2.grid(color='lightgray', ls='dashed')


#plt.ylim(-1.8, 1.8)
fig.align_ylabels((ax1, ax2))
plt.show()