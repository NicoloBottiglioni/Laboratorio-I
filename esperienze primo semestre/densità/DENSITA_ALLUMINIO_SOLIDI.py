from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt

#CILINDRO PICCOLO - misure in kg e m
d=5.98/1000.0
sigma_d=0.05/1000.0

h=19.48/1000.0
sigma_h=0.05/1000.0

m=1.484/1000.0
sigma_m=0.004/1000.0

'''volume cilindro piccolo'''
v=(h*np.pi*d**2.0)/4.0
sigma_v=np.sqrt(((np.pi*d*h/2.0)**2.0)*sigma_d**2.0 + (((np.pi*d**2.0)/4.0)**2.0)*sigma_h**2.0)



#CILINDRO GRANDE - misure in kg e m
D=11.95/1000.0
sigma_D=0.02/1000.0

H=19.31/1000.0
sigma_H=0.01/1000.0

M=5.848/1000.0
sigma_M=0.004/1000.0


'''volume cilindro grande'''
V=(H*np.pi*D**2.0)/4.0
sigma_V=np.sqrt(((np.pi*D*H/2.0)**2.0)*sigma_D**2.0 + ((np.pi*D**2.0/4.0)**2.0)*sigma_H**2.0)



#PARALLELEPIPEDO QUADRATO - misure in kg e m- misure di lunghezza, profondità, altezza
l=10.07/1000.0
sigma_l=0.02/1000.0

p=10.09/1000.0
sigma_p=0.02/1000.0

a=17.74/1000.0
sigma_a=0.01/1000.0

mp=4.887/1000.0
sigma_mp=0.004/1000.0

'''volume parallelepipedo quadrato'''
vp=a*l*p
sigma_vp=np.sqrt((l*p*sigma_a)**2.0 + (a*p*sigma_l)**2.0 + (a*l*sigma_p)**2.0)



#PARALLELEPIPEDO- misure in kg e m - misure di lunghezza, profondità, larghezza
L=17.68/1000.0
sigma_L=0.02/1000.0

P=20.19/1000.0
sigma_P=0.02/1000.0

A=8.28/1000.0
sigma_A=0.02/1000.0

MP=8.002 /1000.0
sigma_MP=0.004/1000.0

'''volume parallelepipedo'''
VP=A*L*P
sigma_VP=np.sqrt((L*P*sigma_A)**2.0 + (A*P*sigma_L)**2.0 + (A*L*sigma_P)**2.0)



#array volumi e masse
VOL=np.array([v, V, vp, VP])
sigma_VOL=np.array([sigma_v, sigma_V, sigma_vp, sigma_VP])
MAS=np.array([m, M, mp, MP])
sigma_MAS=np.array([sigma_m, sigma_M, sigma_mp, sigma_MP])


#modello di fit e fit
def line(x, m, q):
    return m*x + q

plt.figure('Grafico Volume_massa_alluminio')
plt.errorbar(MAS, VOL, sigma_VOL, sigma_MAS, fmt='.', color='blue')
popt, pcov=curve_fit(line, MAS, VOL)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
xx = np.linspace(0., 0.01, 10)
plt.plot(xx, line(xx, m_hat, q_hat), color='yellow')
plt.xlabel('Massa [Kg]')
plt.ylabel('Volume[m$^3$]')
plt.grid(which='both', ls='dashed', color='grey')

#Definizione densità
ro=1.0/m_hat
sigma_ro=((1.0/m_hat)**2)*sigma_m
int=q_hat
sigma_int=sigma_q

print(f'Denstià ottone{ro} {sigma_ro}----Intercetta {int} {sigma_int}')


#RESIDUI
plt.figure('Residui_alluminio')
res = VOL-line(MAS, m_hat, q_hat)
plt.errorbar(MAS, res, sigma_VOL, fmt='.', color='blue')
plt.axhline(0, color="black")
plt.grid(which= 'both', ls='dashed', color='grey')
plt.xlabel('Massa [kg]')
plt.ylabel('Residui alluminio')
