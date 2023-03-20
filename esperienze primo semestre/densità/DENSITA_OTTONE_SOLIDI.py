import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


#CILINDRO PICCOLO - misure già in kg e m - valori di diametro, altezza, massa
d=9.82/1000.0
sigma_d=0.02/1000.0

h=16.13/1000.0
sigma_h=0.01/1000.0

m=10.461/1000.0 #valore già scalato di 0.006 perchè errore di zero
sigma_m=0.004/1000.0


'''volume cilindro piccolo'''
v=(h*np.pi*d**2.0)/4.0
sigma_v=np.sqrt(((np.pi*d*h/2.0)**2.0)*sigma_d**2.0 + (((np.pi*d**2.0)/4.0)**2.0)*sigma_h**2.0)




#CILINDRO GRANDE - misure gia in kg e m - valori di diametro, altezza, massa

D=9.86/1000.0
sigma_D=0.01/1000.0

H=37.55/1000.0
sigma_H=0.01/1000.0

M=24.573/1000.0 #valore già scalato di 0.006 perchè errore di zero
sigma_M=0.004/1000.0

'''volume cilindro grande'''
V=(H*np.pi*D**2.0)/4.0
sigma_V=np.sqrt(((np.pi*D*H/2.0)**2.0)*sigma_D**2.0 + ((np.pi*D**2.0/4.0)**2.0)*sigma_H**2.0)




#PARALLELEPIPEDO - misure in kg e m - valori di altezza, lunghezza, profondità, massa
a=41.40/1000.0
sigma_a= 0.02/1000.0

l=9.91/1000.0
sigma_l=0.02/1000.0

p=9.890/1000.0
sigma_p=0.02/1000.0

mp=34.690/1000.0
sigma_mp=0.004/1000.0

'''volume parallelepipedo'''
vp=a*l*p
sigma_vp=np.sqrt((l*p*sigma_a)**2.0 + (a*p*sigma_l)**2.0 + (a*l*sigma_p)**2.0)

#PRISMA - valori in kg e m - valori di apotema e altezza e massa
T=9.89/2000.0
sigma_T=0.02/2000.0

A=22.60/1000.0
sigma_A=0.01/1000.0

MP=16.393/1000.0
sigma_MP=0.004/1000.0

'''volume prisma'''
VP=6.0*A*np.tan(np.radians(30))*T**2.0
sigma_VP=np.sqrt(((6.0*np.tan(np.radians(30))*T**2.0)**2.0)*sigma_A**2.0 + ((12.0*A*T*np.tan(np.radians(30)))**2.0)*sigma_T**2.0)


#array volumi e masse
VOL=np.array([v, V, vp, VP])
sigma_VOL=np.array([sigma_v, sigma_V, sigma_vp, sigma_VP])
MAS=np.array([m, M, mp, MP])
sigma_MAS=np.array([sigma_m, sigma_M, sigma_mp, sigma_MP])


#modello di fit e fit
def line(x, m, q):
    return m*x + q

plt.figure('Grafico Volume_massa_ottone')
plt.errorbar(MAS, VOL, sigma_VOL, sigma_MAS, fmt='.', color='purple')
popt, pcov=curve_fit(line, MAS, VOL)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
xx = np.linspace(0., 0.05, 10)
plt.plot(xx, line(xx, m_hat, q_hat), color='orange')
plt.xlabel('Massa [Kg]')
plt.ylabel('Volume[m$^3$]')
plt.grid(which='both', ls='dashed', color='black')

#Definizione densità
ro=1.0/m_hat
sigma_ro=((1.0/m_hat)**2)*sigma_m
int=q_hat
sigma_int=sigma_q

print(f'Denstià ottone{ro} {sigma_ro}----Intercetta {int} {sigma_int} ')

#RESIDUI
plt.figure('Residui_ottone')
res = VOL-line(MAS, m_hat, q_hat)
plt.errorbar(MAS, res, sigma_VOL, fmt='.', color='purple')
plt.axhline(0, color="black")
plt.grid(which= 'both', ls='dashed', color='black')
plt.xlabel('massa [kg]')
plt.ylabel('Residui acciaio')
