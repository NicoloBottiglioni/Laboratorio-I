import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#definire misure su ascisse e ordinate con i rispettivi errori
x, T=np.loadtxt(r'C:\Users\nicob\OneDrive\Desktop\dati_rame.txt', unpack= True)
sigma_x=np.full(x.shape, 0.1)
sigma_T=np.array([0.2, 0.1, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3])


#definire il modello di fit
def line(x, m, q):
    return m * x + q

#plotttare le misure con i rispettivi errori
plt.figure('Grafico Posizione-Temperatura Rame')
plt.errorbar(x, T, sigma_T, sigma_x, fmt='.')


#fit vero e proprio
popt, pcov = curve_fit(line, x, T, sigma=sigma_T)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
#Dire nella relazione che è stato solo considerato solo l'errore sulla y!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#stampa del coefficiente angolare e dell'intercetta

print('Coefficiente angolare rame', m_hat, sigma_m)
print('Intercetta', q_hat, sigma_q)

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

S=np.pi*D**2/4
eS=np.pi*D*eD/2


L_hat= -(V * A) / (2 * S * m)
eL=np.sqrt((A*eV/(S*m))**2 +(V*eA/(S*m))**2 + (V*A*eS/(m*S**2))**2 + (V*A*em/(S*m**2))**2)/2

print('Valore centrale conducibilità', L_hat, '+/-',eL)


#Grafico modello di best fit. Qua ho messo 'xx' perchè così non ho problemi col grafico dei resiudi
xx =np.linspace(0.,40.,100)
plt.plot(xx,line(xx, m_hat, q_hat))

#formattazione del grafico
plt.xlabel('Posizione [cm]')
plt.ylabel('Temperatura [$^\\circ$C]')
plt.grid(which='both', ls='dashed', color='black')

#Grafico dei residui
res = T-line(x, m_hat, q_hat)
plt.errorbar(x, res, sigma_T, fmt='o')
plt.axhline(0,color="black")
plt.grid(which= 'both', ls='dashed', color='black')
plt.xlabel('× [cm]')
plt.ylabel('Residui Rame ')
plt.ylim(-1.0, 1.0)



#mostrare la figura
plt.show()


