import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

'''NOTE
---Misurare R e aggiungerlo a l. Prima di fare questo provare a non aggiungerlo e a vedere come viene il fit, semmai aggiungerlo come parametro e infine sommare R a l
---alla fine controllare il numero di gradi di libertà!!!!!!!!!!!!!!!!!!!!!!!!!
---per trascurare prendo l, THETA maggiore e l'errore sul tempo minore.
-- Controllare i residui
--Prima di misurare i periodi, vedere per quale angolo il pendolo oscilla effettivamente 10 volte senza variazioni significative'''





#deviazioni standard per i tempi
t1=np.array([]) #tempo di 10 periodi per la prima ampiezza
t2=np.array([])
t3=np.array([])
t4=np.array([])
t5=np.array([])
t6=np.array([])
n=len(t1) #numero di volte in cui si misura il tempo di 10 periodi per una ampiezza
m= 10 #numero di periodi misurati in una singola volta

t1_mu = np.mean(t1/m) # t1/m è il tempo di un singolo periodo. Con questo comando h fatto la media di n misure diverse
t2_mu = np.mean(t2/m)
t3_mu = np.mean(t3/m)
t4_mu = np.mean(t4/m)
t5_mu = np.mean(t5/m)
t6_mu = np.mean(t6/m)

sigma_t1 =  np.std( t1/m ,ddof = 1)/np.sqrt(n)
sigma_t2 =  np.std( t2/m ,ddof = 1)/np.sqrt(n)
sigma_t3 =  np.std( t3/m ,ddof = 1)/np.sqrt(n)
sigma_t4 =  np.std( t4/m ,ddof = 1)/np.sqrt(n)
sigma_t5 =  np.std( t5/m ,ddof = 1)/np.sqrt(n)
sigma_t6 =  np.std( t6/m ,ddof = 1)/np.sqrt(n)


#angoli - per ogni ampiezza diversa, misuro i cateti e ricavo così gli angoli
K1=np.array([])/1000 #m; divido perché misuro in carta millimetrata e converto: è il cateto orizzontale
K2=np.array([]) /1000 #m; è il cateto verticale
sigma_K1= /(1000*np.sqrt(12)) # deviazione della distribuzione uniforme
sigma_K2= /1000
D= K1/K2
sigma_D=D*np.sqrt((sigma_K1/K1)**2 + (sigma_K2/K2)**2)
THETA= np.arctan(D)
sigma_THETA= sigma_D/(1+D**2)
print(f'Valore fissato dell angolo= {THETA} +/- {sigma_THETA}')

#dati
theta=np.array([]) # radianti
dtheta=np.full()

t=np.array([t1_mu, t2_mu, t2_mu, t3_mu, t4_mu, t5_mu, t6_mu])
dt=np.array([sigma_t1, sigma_t2, sigma_t2, sigma_t3, sigma_t4, sigma_t5, sigma_t6])
print(f'errori sul tempo {dt}') #per verificare che termini dello sviluppo trascurare

'''
#Misura di L+R
L=
dL =
R=
dR=
print(f'{L+R} +/- {np.sqrt(dL**2 + dR**2)}')
'''

#modello e fit 
#Modello 
def period(pars, theta):
    return 2*np.pi*np.sqrt((pars[0])/9.81)*(1+ theta**2/16 + 11*theta**4/3072 + 173*theta**6/737280 + 22931*theta**8/1321205760)
#Assumiamo noto il valore 9.81


#Fit odr
model = odr.Model(period)
data = odr.RealData(theta, t, sx=dtheta, sy=dt)
alg = odr.ODR(data, model, beta0=(0.145 , 0))
out = alg.run()
l_hat = out.beta
sigma_l = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square

#print('Raggio', R_hat, sigma_R)
print('Angolo stimato', l_hat, sigma_l)
print(f'Chisquare = {chisq:.1f}')
print(f'Chisq atteso={len(theta)-1} +/- {np.sqrt(2*len(theta)-2)}') #Probabilmente il chisq non è così, perche le misure su x e y non sono entrambe gaussiane. Probabilemte, la deviazione std del chiquadro è diversa ma il valore centrale è giusto.

#residui (metodo francesco)
resid=np.sqrt(out.delta**2 + out.eps**2)
for i in range(0, 4):   # 4 è l'ipotetico numero di angoli  diversi misurato
    if t[i] < 2*np.pi/np.sqrt(l_hat/9.81)*(1 + 1*theta[i]**2//16 + 11*theta[i]**4/3072 + 173*theta[i]**6/737280 + 22931*theta[i]**8/1321205760):
        resid[i] = -resid[i]

'''
#residui(mio metodo)
#residui - Controllare i valori perché me li mette a cazzo di cane
res=np.sqrt(out.delta**2 + out.eps**2)
print(f'residui {res}')
res1= np.array([])'''

#grafico
fig=plt.figure('fit_pendolo_semplice', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))


ax1.errorbar(theta, t, dt, dtheta, fmt='.', label='punti sperimentali', color='darkslateblue')
#linspace
ax1.plot(theta, period(out.beta, theta), label='Modello di best-fit', color='lightsteelblue')
ax1.set_ylabel('T [s]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(theta, res, dt, fmt='.',  color='darkslateblue')
ax2.plot(theta, np.full(theta.shape, 0.0), color='lightsteelblue')
ax2.set_xlabel('l [m]')
ax2.set_ylabel('Residui [s]')
ax2.grid(color='lightgray', ls='dashed')
#plt.ylim(-4.0, 4.0)
fig.align_ylabels((ax1, ax2))
plt.show()