import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


#CILINDRO PICCOLO - misure già in kg e m - valori di diametro, altezza, massa
d=9.82/1000.0
sigma_d=0.02/1000.0

h=16.13/1000.0
sigma_h=0.01/1000.0

m=10.461/1000.0 #valore già scalato di 0.006 perchè errore di zero
sigma_m=0.001/1000.0


'''volume cilindro piccolo'''
v=(h*np.pi*d**2.0)/4.0
sigma_v=np.sqrt(((np.pi*d*h/2.0)**2.0)*sigma_d**2.0 + (((np.pi*d**2.0)/4.0)**2.0)*sigma_h**2.0)

#Cilindro lungo
D3=0.6/1000.0
sigmaD3=0.02/1000

H3=94.3/100
sigma_H3=0.02/1000

M3=22.240/1000
sigmaM3=0.001/1000

V3=(h*np.pi*D3**2.0)/4.0
sigma_V3=np.sqrt(((np.pi*D3*H3/2.0)**2.0)*sigmaD3**2.0 + (((np.pi*D3**2.0)/4.0)**2.0)*sigma_H3**2.0)


#CILINDRO GRANDE - misure gia in kg e m - valori di diametro, altezza, massa

D=9.86/1000.0
sigma_D=0.01/1000.0

H=37.55/1000.0
sigma_H=0.01/1000.0

M=24.573/1000.0 #valore già scalato di 0.006 perchè errore di zero
sigma_M=0.001/1000.0



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
sigma_mp=0.001/1000.0

'''volume parallelepipedo'''
vp=a*l*p
sigma_vp=np.sqrt((l*p*sigma_a)**2.0 + (a*p*sigma_l)**2.0 + (a*l*sigma_p)**2.0)

#PRISMA MENO GROSSO - valori in kg e m - valori di apotema e altezza e massa
T=9.89/2000.0
sigma_T=0.02/2000.0

A=22.60/1000.0
sigma_A=0.01/1000.0

MP=16.393/1000.0
sigma_MP=0.004/1000.0

'''volume prisma'''
VP=6.0*A*np.tan(np.radians(30))*T**2.0
sigma_VP=np.sqrt(((6.0*np.tan(np.radians(30))*T**2.0)**2.0)*sigma_A**2.0 + ((12.0*A*T*np.tan(np.radians(30)))**2.0)*sigma_T**2.0)

#PRISMA GROSSISSIMO
M4=28.821/1000
sigmaM4=0.001/1000

U=15.00/2000
sigmaU=0.02/2000

A2=18.24/1000
sigmaA2= 0.01/1000

V4=6.0*A2*np.tan(np.radians(30))*U**2.0
sigma_V4=np.sqrt(((6.0*np.tan(np.radians(30))*U**2.0)**2.0)*sigmaA2**2.0 + ((12.0*A2*U*np.tan(np.radians(30)))**2.0)*sigmaA2**2.0)


#array volumi e masse
VOL=np.array([v, V, VP, V4])
sigma_VOL=np.array([sigma_v, sigma_V, sigma_VP, sigma_V4])
MAS=np.array([m, M, MP, M4])
sigma_MAS=np.array([sigma_m, sigma_M, sigma_MP, sigma_V4])
print(VOL)
print(MAS)

#modello di fit e fit
def line(x, m, q):
    return m*x + q

popt, pcov=curve_fit(line, MAS, VOL, sigma= sigma_VOL)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())

#Definizione densità
ro=1.0/m_hat
sigma_ro=((1.0/m_hat)**2)*sigma_m
int=q_hat
sigma_int=sigma_q

print(f'Denstià ottone{ro} {sigma_ro}----Intercetta {int} {sigma_int} ')

#residui normalizzati
res= (VOL- line(MAS, *popt))/sigma_VOL
chisq=np.sum((((VOL - line(MAS, *popt))/sigma_VOL)**2))
print(f'chisq stimato = {chisq}')


#grafico
fig=plt.figure('densità', figsize=(10., 6.), dpi=100)

ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(MAS, VOL, sigma_VOL, sigma_MAS, fmt='.', label='punti sperimentali', color='midnightblue')
ax1.plot(MAS, line(MAS, *popt), label='Modello di best-fit', color='deepskyblue')
ax1.set_ylabel('volume [m^3]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(MAS, res, sigma_VOL, fmt='.', color='midnightblue')
#np.linspace
ax2.plot(MAS, np.full(MAS.shape, 0.0), color='deepskyblue')
ax2.set_xlabel('massa [kg]')
ax2.set_ylabel('Residui normalizzati]')
ax2.grid(color='lightgray', ls='dashed')


#plt.ylim(-1.8, 1.8)
fig.align_ylabels((ax1, ax2))
plt.show()