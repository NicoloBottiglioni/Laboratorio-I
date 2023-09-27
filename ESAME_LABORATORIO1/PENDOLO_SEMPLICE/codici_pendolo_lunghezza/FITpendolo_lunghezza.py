import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

'''NOTE
---Misurare R e aggiungerlo a l. Prima di fare questo provare a non aggiungerlo e a vedere come viene il fit, semmai aggiungerlo come parametro e infine sommare R a l
---Misurare con la più grande precisione possibile l'angolo inizale
---alla fine controllare il numero di gradi di libertà!!!!!!!!!!!!!!!!!!!!!!!!!
---per trascurare prendo theta, l maggiore e l'errore sul tempo minore.
--Prima di misurare i periodi, vedere per quale angolo il pendolo oscilla effettivamente 10 volte senza variazioni significative'''



'''
#deviazioni standard per i tempi
t1=np.array([]) #tempo di 10 periodi per la prima lunghezza
t2=np.array([])
t3=np.array([])
t4=np.array([])
t5=np.array([])
t6=np.array([])
n=len(t1) #numero di volte in cui si misura il tempo di 10 periodi per una lunghezza
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
'''


#valore fissato di theta (l'angolo è fissato, quindi è sempre lo stesso. Per questo possiamo misurare theta con solo una misura di cateti)
K1=   /1000 #m; divido perché misuro in carta millimetrata e converto: è il cateto orizzontale
K2=  /1000 #m; è il cateto verticale
sigma_K1= /(1000*np.sqrt(12)) # deviazione della distribuzione uniforme
sigma_K2= /1000
D= K1/K2
sigma_D=D*np.sqrt((sigma_K1/K1)**2 + (sigma_K2/K2)**2)
THETA= np.arctan(D)
sigma_THETA= sigma_D/(1+D**2)
print(f'Valore fissato dell angolo= {THETA} +/- {sigma_THETA}')



#Dati
#R - ricordarsi di modificare anche dl 
#dR
#L=np.array([]) misura lunghezza pendolo trascurando raggio della sfera
#dL=np.full(L.shape, 0.001/np.sqrt(12))
#l=L+R
#dl=np.sqrt(dR**2 + dL**2)
l=np.array([0.6810, 0.6400, 0.603, 0.555, 0.511, 0.468]) # lunghezza pednolo in Metri
dl= np.full(l.shape, 0.001/np.sqrt(12)) #errore statistico in metri (risoluzione metro a nastro / sqrt(12)).
'''t=np.array([t1_mu, t2_mu, t2_mu, t3_mu, t4_mu, t5_mu, t6_mu])
dt=np.array([sigma_t1, sigma_t2, sigma_t2, sigma_t3, sigma_t4, sigma_t5, sigma_t6])
print(f'errori sul tempo {dt}') #per verificare che termini dello sviluppo trascurare'''



T=np.array([16.794, 16.256, 15.696, 15.094, 14.470, 13.740]) #misura di 10 periodi per ogni lungheza in modo da spalmare l'errore su questi ultimi
t=T/10 #Misura di un periodo per ogni lunghezza
dt= np.full(t.shape, 0.001) #risoluzione strumentale del cronometro diviso 10


#Modello 
def period(pars, l):
    return 2*np.pi*np.sqrt((l)/9.81)*(1+ pars[0]**2/16 + 11*pars[0]**4/3072 + 173*pars[0]**6/737280 + 22931*pars[0]**8/1321205760)
#Assumiamo noto il valore 9.81


#Fit odr
model = odr.Model(period)
data = odr.RealData(l, t, sx=dl, sy=dt)
alg = odr.ODR(data, model, beta0=(0.145 , 0))
out = alg.run()
theta_hat = out.beta
sigma_theta = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square

#print('Raggio', R_hat, sigma_R)
print('Angolo stimato', theta_hat, sigma_theta)
print(f'Chisquare = {chisq:.1f}')
print(f'Chisq atteso={len(l)-1} +/- {np.sqrt(2*len(l)-2)}') #Probabilmente il chisq non è così, perche le misure su x e y non sono entrambe gaussiane. Probabilemte, la deviazione std del chiquadro è diversa ma il valore centrale è giusto.



#Verifica dei termini dello sviluppo che posso trascurare. Verifichaimo sia con il valore misurato di theta sia con quello stimato dal fit
print(f'Termini dello sviluppo con theta misurato: {THETA**2/16} --- {11*THETA**4/3072}--- {173*THETA**6/737280}--- {22931*THETA**8/1321205760}')
print(f'Termini dello sviluppo con theta stimato: {theta_hat**2/16} --- {11*theta_hat**4/3072}--- {173*theta_hat**6/737280}--- {22931*theta_hat**8/1321205760}')

#residui - Controllare i valori perché me li mette a cazzo di cane
res=np.sqrt(out.delta**2 + out.eps**2)

for i in range(0, 10):   # 4 è l'ipotetico numero di angoli  diversi misurato
    if t[i] < period(theta_hat[0], l):
        res[i] = -res[i]

#Grafico
fig=plt.figure('fit_pendolo_semplice', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))


ax1.errorbar(l, t, dt, dl, fmt='.', label='punti sperimentali', color='darkslateblue')
#linspace
ax1.plot(l, period(out.beta, l), label='Modello di best-fit', color='lightsteelblue')
ax1.set_ylabel('T [s]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(l, res, dt, fmt='.',  color='darkslateblue')
ax2.plot(l, np.full(l.shape, 0.0), color='lightsteelblue')
ax2.set_xlabel('l [m]')
ax2.set_ylabel('Residui [s]')
ax2.grid(color='lightgray', ls='dashed')
#plt.ylim(-4.0, 4.0)
fig.align_ylabels((ax1, ax2))
plt.show()