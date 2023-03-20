from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


'''MISURE DELLA MASSA M E DIAMETRO D DELLE SFERE DI ACCIAIO DALLA MINORE ALLA MAGGIORE'''
M=np.array([8.406, 11.914, 19.131, 25.045, 45.054])  #GRAMMI - misure scalate di 0.006 g perchè errore di zero
sigma_M=np.full(M.shape, 0.004) #GRAMMI- i valori oscillavano di 0.004 grammi
D=  np.array([12.82, 14.40, 16.85, 18.47, 22.45]) #MILLIMETRI
sigma_D= np.full(D.shape, 0.05) #MILLIMETRI- risoluzione del calibro ventesimale

#CONVERSIONE IN KILOGRAMMI E METRI DELLE SFERE DI ACCIAIO
M=M/1000.0
sigma_M=sigma_M/1000.0
D=D/1000.0
sigma_D=sigma_D/1000.0

#CALCOLO DEL RAGGIO E DEL VOLUME DELLE SFERE
R=D/2.0
sigma_R=sigma_D/2.0 #Metri
V= (4.0*np.pi*R**3.0)/3.0 #Metricubi
sigma_V=3.0*V*sigma_D/D



'''DEFINIZIONE DEL MODELLO DI FIT LINEARE'''
def line(x, m, q):
    return m*x + q



''''SCATTER PLOT ACCIAIO - VALORI DENSITA' ACCIAIO - GRAFICO DI BEST FIT ACCIAIO- RESIDUI ACCIAIO'''
plt.figure('Grafico Volume-massa_acciaio')
plt.errorbar(M, V, sigma_V, sigma_M, fmt='.', color='red') #primo valore asse x, poi y, poi errore y, poi errore x
popt, pcov = curve_fit(line, M, V)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
xx = np.linspace(0., 0.05, 10)
plt.plot(xx, line(xx, m_hat, q_hat), color='green')
plt.xlabel('Massa [Kg]')
plt.ylabel('Volume[m$^3$]')
plt.grid(which='both', ls='dashed', color='grey')

#definizione della densità e dell'intercetta
ro=1.0/m_hat
sigma_ro=((1.0/m_hat)**2)*sigma_m
int=q_hat
sigma_int=sigma_q

print(f'Densità acciaio {ro}+/-{sigma_ro}----Intercetta{int}+/-{sigma_int}')



#RESIDUI
plt.figure('Residui_acciaio')
res = V-line(M, m_hat, q_hat)
plt.errorbar(M, res, sigma_V, fmt='o', color='red')
plt.axhline(0, color="black")
plt.grid(which= 'both', ls='dashed', color='grey')
plt.xlabel('massa [kg]')
plt.ylabel('Residui acciaio')


#LEGGE DI POTENZA PER LE SFERE
def power_law(x, norm, index):
    return norm * (x**index)



#SCATTER PLOT ACCIAIO- GRAFICO BILOGARITMICO ACCIAIO
plt.figure('Grafico raggio_massa_acciaio')
plt.errorbar(M, R, sigma_R, sigma_M, fmt='.', color='red')
popt, pcov = curve_fit(power_law, M, R)
norm_hat, index_hat = popt
sigma_norm, sigma_index = np.sqrt(pcov.diagonal())
x = np.linspace(0.008, 0.05, 100)
plt.plot(x, power_law(x, norm_hat, index_hat), color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('massa [kg]')
plt.ylabel('raggio [m]')
plt.grid(which='both', ls='dashed', color='grey')

#confronto indice e norma
I=index_hat
sigma_I=sigma_index

k=np.cbrt(3.0/(np.pi*4.0*ro))
sigma_k=np.sqrt(((1.0/(ro*np.cbrt(36.0*np.pi*ro)))**2.0)*sigma_ro**2.0)


print(f'Indice legge di potenza{I} {sigma_I} ----- Norma legge di potenza {norm_hat} {sigma_norm} ---Valore atteso della norma {k} {sigma_k}')

#RESIDUI 2
plt.figure('Residui_acciaio_2')
res = R-power_law(M, norm_hat, index_hat)
plt.errorbar(M, res, sigma_R, fmt='o', color='red')
plt.axhline(0, color="black")
plt.grid(which= 'both', ls='dashed', color='grey')
plt.xlabel('massa [kg]')
plt.ylabel('Residui acciaio2')

plt.show()