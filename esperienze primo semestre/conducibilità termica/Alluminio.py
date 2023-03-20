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
sigma_T = np.array([0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3])

#definire il modello di fit
def line(x, m, q):
    return m * x + q

#plotttare le misure con i rispettivi errori
plt.figure('Grafico Posizione-Temperatura Alluminio')
plt.errorbar(x, T, sigma_T, sigma_x, fmt='.')

# Fit vero e proprio

popt, pcov = curve_fit(line, x, T, sigma = sigma_T)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
#DIre nella relazione che è stato considerato solo l'errore sulla y e che la funzione curve_fit fa il fit secondo i minimi quadrati!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#stampa del coefficiente angolare e dell'intercetta
print('coef angolare alluminio',m_hat, sigma_m) #gradi/cm
print('intercetta', q_hat, sigma_q)

#valore centrale conducibilità alluminio e errore conducibilità
V=10.2 #V
eV=0.1

A=1.61 #A
eA=0.01

D=25.00 #mm
eD=0.05

#portiamo tutto in m
m=m_hat*100
em=sigma_m*100
D=D/1000
eD=eD/1000
#Calcolo la sezione
S=np.pi*D**2/4
eS=np.pi*D*eD/2


L_hat= -(V * A) / (2 * S * m)
eL=np.sqrt((A*eV/(S*m))**2 +(V*eA/(S*m))**2 + (V*A*eS/(m*S**2))**2 + (V*A*em/(S*m**2))**2)/2


print('Valore centrale conducibilità', L_hat, '+/-',eL)
#Il valore centrale della conducibilità viene più grande . Rifletterne il motivo. Strumenti e dissipazione termica aria


# Grafico del modello di best fit.
xx = np.linspace(0., 40., 100)
plt.plot(xx, line(xx, m_hat, q_hat))

#Formattazione del grafico.
plt.xlabel('Posizione [cm]')
plt.ylabel('Temperatura [$^\\circ$C]')
plt.grid(which='both', ls='dashed', color='black')

#Grafico dei residui
res = T-line(x, m_hat, q_hat)
plt.errorbar(x, res, sigma_T, fmt='o')
plt.axhline(0,color="black")
plt.grid(which= 'both', ls='dashed', color='gray')
plt.xlabel('× [cm]')
plt.ylabel('Residui Alluminio')
plt.ylim(-1.0, 1.0)

#mostrare la figura
plt.show()