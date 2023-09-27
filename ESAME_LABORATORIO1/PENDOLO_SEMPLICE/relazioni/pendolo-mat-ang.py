import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit 
from scipy import odr 
l = 25
sigma_l = 0.2
g = 9.81
#import dei dati in caso di file esterno: 
t1, t2, t3, t4 = np.genfromtxt(fname ='', usecols=(0, 1, 2,3,4), skip_header=0,skip_footer=0, unpack=True )
n = len(t1) #numero di misure per periodo 
m = 10 #numero di periodi misurati per ogni misura 
t1_mu = np.mean(t1)/m
t2_mu = np.mean(t2)/m
t3_mu = np.mean(t3)/m
t4_mu = np.mean(t4)/m
sigma_t1 =  np.std( t1 ,ddof = 1)/np.sqrt(n)/m
sigma_t2 =  np.std( t2 ,ddof = 1)/np.sqrt(n)/m
sigma_t3 =  np.std( t3 ,ddof = 1)/np.sqrt(n)/m
sigma_t4 =  np.std( t4 ,ddof = 1)/np.sqrt(n)/m
sin_teta = np.array([1, 2, 3, 4, 5, 6, 7 ,8])
teta = np.asin(sin_teta) 
sigma_sin = np.full(sin_teta.shape, 1)
sigma_teta = (1/np.sqrt(1-sin_teta**2))*sigma_sin
t= np.array([t1_mu, t2_mu, t3_mu,t4_mu])
sigma_t = np.array([sigma_t1, sigma_t2, sigma_t3,sigma_t4])
# definisco una variabile y = t/sqrt(l) e vado a definire una nuova funzione da fittare 
y = t/np.sqrt(l)
sigma_y = y*np.sqrt((sigma_t/t)**2 + (1/2*l/sigma_l)**2) 

#in base a sigma y e teta grande verifico fino  a che termine devo sviluppare taylor 
# voglio che sigma_y >> del primo termine dello sviluppo che ignoro

def period_model(teta, g ):
    return 2*np.pi/np.sqrt(g)*(1 + 1/16*teta**2 + 11/3072*teta**4 + 173/737280*teta**6)

#disegno del grafico 
plt.figure('pendolo mat')
plt.subplot(2,1,1)
plt.errorbar(teta, y, sigma_t , sigma_y,  fmt='.',color='crimson', ecolor='crimson')
x = np.linspace(5, 35, num=5000)
plt.plot (x, period_model(x,g),color='orangered')
plt.xlim (4, 36)

plt.ylabel('periodo/sqrt(l) [s*m^-1/2]')
plt.grid(which='both', ls='dashed', color='gray')

#residui 

plt.subplot(2,1,2)
residui= y - period_model(teta,g)
plt.errorbar(teta, residui, sigma_y,  fmt='.', color='crimson',  ms = 4)
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('teta[rad]')
plt.ylabel('Residuals')
plt.show()

#calcolo del chi quadro 
#teorico 
chi_teorico = n
sigma_chi = np.sqrt(2*n)

#effettivo
chisq= np.sum(((residui)/sigma_y)**2)

print(f' X_teo = {chi_teorico} +- {sigma_chi}')

print(f'chisq={chisq}')
#se il chiquadro è ragionevole il gioco è fatto ma probabilmente non lo sarà dato che gli errori sulla x non saranno trascurabili rispetto quelli sulla y 
#non sono in grado di fare un fit in ODR quindi l'unica altra alternativa è quella di usare gli errori efficaci 

sigma_y_eff = np.sqrt(sigma_y**2 + ((2*np.pi/np.sqrt(g)*( 1/8*teta + 11*4/3072*teta**3 + 173*6/737280*teta**5))*sigma_teta)**2 )

#ricalcolo il chi  con gli errori efficaci. 
chisq= np.sum((residui/sigma_y_eff)**2)

#alternativa: stima di  l con ODR
#in caso si voglia stimare la validità del fit tramite stima del parametro l posso farlo in odr 


def modello(pars, teta):
    return 2.0 * np.pi * np.sqrt(pars[0]/g)*(1 + 1/16*teta**2 + 11/3072*teta**4 + 173/737280*teta**6)
model = odr.Model(modello)      # ovviamente è prima necessario definire un modello (ricordarsi anche di importare la libreria di scipy per odr)
data = odr.RealData(teta, t, sx = sigma_teta, sy = sigma_t)
alg = odr.ODR(data, model, beta0 = (1.0, 1.0))
out = alg.run()
a_hat = out.beta
sigma_a = np.sqrt(out.cov_beta.diagonal())
chi = out.sum_square

parameters = np.array([a_hat])    #Questo è necessario se si vuole rappresentare poi il grafico 
l_hat = parameters[0]
sigma_l = sigma_a[0]

#grafico col fit in ODR

plt.figure('pendolo mat2')
plt.subplot(2,1,1)
plt.errorbar(teta, y, sigma_t , sigma_y,  fmt='.',color='crimson', ecolor='crimson')
x = np.linspace(5, 35, num=5000)
plt.plot (x, period_model(x,g),color='orangered')
plt.xlim (4, 36)

plt.ylabel('periodo[s]')
plt.grid(which='both', ls='dashed', color='gray')

#residui 

resid=np.sqrt(out.delta**2 + out.eps**2)
for i in range(0, 4):   # 4 è l'ipotetico numero di angoli  diversi misurato
    if t[i] < 2*np.pi/np.sqrt(l_hat/g)*(1 + 1/16*teta[i]**2 + 11/3072*teta[i]**4 + 173/737280*teta[i]**6):
        resid[i] = -resid[i]


plt.subplot(2,1,2)
plt.errorbar(teta, resid, sigma_t, fmt='.', color='royalblue', ecolor='royalblue')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('teta[rad]')
plt.ylabel('Residui')
plt.show()