import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit 
from scipy import odr 
l = 1 
sigma_l = 0.001/np.sqrt(12)
g = 9.81
#import dei dati in caso di file esterno: 
t1, t2, t3, t4 = np.genfromtxt(fname ='', usecols=(0, 1, 2,3,4), skip_header=0,skip_footer=0, unpack=True )
p = len(t1) #numero di misure per periodo 
m = 10 #numero di periodi misurati per ogni misura 
t1_mu = np.mean(t1)/m
t2_mu = np.mean(t2)/m
t3_mu = np.mean(t3)/m
t4_mu = np.mean(t4)/m
sigma_t1 =  np.std( t1 ,ddof = 1)/np.sqrt(p)/m  
sigma_t2 =  np.std( t2 ,ddof = 1)/np.sqrt(p)/m
sigma_t3 =  np.std( t3 ,ddof = 1)/np.sqrt(p)/m
sigma_t4 =  np.std( t4 ,ddof = 1)/np.sqrt(p)/m
d1 = np.array([])
d2 = np.array([])
sigma_d1 = np.full(d1.shape, 0.001)
sigma_d2 = np.full(d2.shape, 0.001)
tan_teta = d1/d2
sigma_tan = tan_teta*np.sqrt((sigma_d1/d1)**2 + (sigma_d2/d2)**2)
teta = np.atan(tan_teta) 
sigma_teta = (1/(1-tan_teta**2))*sigma_tan
t= np.array([t1_mu, t2_mu, t3_mu,t4_mu])
sigma_t = np.array([sigma_t1, sigma_t2, sigma_t3,sigma_t4])
n= len(t)

'''#test coi dati di marco 
n= 10
t = np.array([2013,2023,2023,2031,2032,2032,2040,2041,2045,2048])
sigma_t =np.array([5,2,3,6,4,3,2,1,3,5])
t= t/1000
sigma_t = sigma_t/1000
teta = np.array([73,112,150,187,224,261,285,331,377,420])
teta = teta/1000
sigma_teta = np.full(teta.shape,0.003)'''
# definisco una variabile y = t/sqrt(l) e vado a definire una nuova funzione da fittare 
y = t/np.sqrt(l)
sigma_y = y*np.sqrt((sigma_t/t)**2 + (1/2*sigma_l/l)**2) 

#in base a sigma y e teta grande verifico fino  a che termine devo sviluppare taylor 
# voglio che sigma_y >> del primo termine dello sviluppo che ignoro

def period_model(teta):
    return 2*np.pi/np.sqrt(g)*(1 + 1/16*teta**2 + 11/3072*teta**4 + 173/737280*teta**6)

#disegno del grafico 
plt.figure('pendolo mat')
plt.subplot(2,1,1)
plt.errorbar(teta, y, sigma_y,   fmt='.',color='crimson', ecolor='crimson')
x = np.linspace(0, 20, num=5000)
plt.plot (x, period_model(x),color='orangered')
plt.xlim(0, 0.6)
plt.ylabel('periodo/sqrt(l) [s*m^-1/2]')
plt.grid(which='both', ls='dashed', color='gray')

#residui 

plt.subplot(2,1,2)
residui= y - period_model(teta)
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
chisq_eff= np.sum((residui/sigma_y_eff)**2)
print(f'chisq_eff={chisq_eff}')

#viene un chi quadro da fare spaventosamente schifo quindi dovrei assumere che sia sbagliato il modello 

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
def function(z,j):
    return 2*np.pi*np.sqrt(j/9.81)*(1 + 1/16*z**2 + 11/3072*z**4 + 173/737280*z**6)
plt.figure('pendolo mat2')
plt.subplot(2,1,1)
plt.errorbar(teta, t, sigma_t ,  fmt='.',color='crimson', ecolor='crimson')
zz=np.linspace(0, 5)
plt.plot (zz, function(zz,1), color= 'crimson')

plt.ylabel('periodo[s]')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlim(0,0.6)
#residui 

resid=np.sqrt(out.delta**2 + out.eps**2)
for i in range(0, 10):   # 4 è l'ipotetico numero di angoli  diversi misurato
    if t[i] < function(teta[i],1):
        resid[i] = -resid[i]


plt.subplot(2,1,2)
plt.errorbar(teta, resid, sigma_t, fmt='.', color='royalblue', ecolor='royalblue')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('teta[rad]')
plt.ylabel('Residui')
plt.show()
print(l_hat)
print(sigma_l)
print(chi)
#il chisquared fa meno schifo ma ancora incompatibile tutta via viene un valore fittato per l compatibile con quello misurato 

