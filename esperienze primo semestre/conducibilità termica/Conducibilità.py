import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#definire misure su ascisse e ordinate con i rispettivi errori
'''
x = np.array([])
sigma_x = np.full(x.shape, 0.1)
T = np.array([])
sigma_T = np.full(T.shape, 0.2)
'''

x, T=np.loadtxt(r'C:\Users\nicob\OneDrive\Desktop\dati_alluminio.txt',unpack=True)
sigma_x = np.full(x.shape, 0.1)
sigma_T = np.full(T.shape, 0.2)

#definire il modello di fit
def line(x, m, q):
    return m * x + q

#plotttare le misure con i rispettivi errori
plt.figure('Grafico Posizione-Temperatura Alluminio')
plt.errorbar(x, T, sigma_T, sigma_x, fmt='o')

# Fit vero e proprio

popt, pcov = curve_fit(line, x, T, sigma = sigma_T)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())

#stampa del coefficiente angolare e dell'intercetta
print('coef angolare',m_hat, sigma_m)
print('intercetta', q_hat, sigma_q)


'''V_hat=10.2
sigma_V=0.1
A_hat=1.61
sigma_A=0.01
S_hat=25.00
U_hat= 0.0125*0.0125*np.pi
sigma_S=0.05
L_hat= (V_hat * A_hat) / (2 * U_hat * m_hat )
print('Valore centrale conducibilità', L_hat)'''

# Grafico del modello di best fit.
x = np.linspace(0., 40., 100)
plt.plot(x, line(x, m_hat, q_hat))
# Formattazione del grafico.

plt.xlabel('Posizione [cm]')
plt.ylabel('Temperatura [$^\\circ$C]')
plt.grid(which='both', ls='dashed', color='black')
#plt.savefig()

#mostrare la figura
plt.show()