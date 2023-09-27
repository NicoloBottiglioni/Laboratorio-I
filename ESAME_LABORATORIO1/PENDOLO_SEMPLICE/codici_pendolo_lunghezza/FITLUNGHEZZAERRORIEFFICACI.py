import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#Dati
#R
l=np.array([0.6810, 0.6400, 0.603, 0.555, 0.511, 0.468]) # lunghezza pednolo in Metri
dl= np.full(l.shape, 0.001/np.sqrt(12)) #errore statistico in metri (risoluzione metro a nastro / sqrt(12)).

T=np.array([16.794, 16.256, 15.696, 15.094, 14.470, 13.740]) #misura di 10 periodi per ogni lungheza in modo da spalmare l'errore su questi ultimi
t=T/10 #Misura di un periodo per ogni lunghezza
dt= np.full(t.shape, 0.001) #risoluzione strumentale del cronometro diviso 10

#Modello 
def period(l, theta):
    return 2*np.pi*np.sqrt((l)/9.81)*(1+ theta**2/16 + 11*theta**4/3072 + 173*theta**6/737280 + 22931*theta**8/1321205760)
#Assumiamo noto il valore 9.81

#fit errori efficaci
#modello e fit con gli errori efficaci

popt, pcov= curve_fit(period, l, t, sigma=dt, absolute_sigma= True)
for i in range(5):
    sigma_eff = np.sqrt(dt**2.0 + ((1+ popt[0]**2/16 + 11*popt[0]**4/3072 + 173*popt[0]**6/737280 + 22931*popt[0]**8/1321205760)*np.pi * dl/(np.sqrt(9.81*(l))))**2.0)
    popt, pcov = curve_fit(period, l, t, sigma=sigma_eff, absolute_sigma=True)
    chisq = (((t - period(l, *popt)) / sigma_eff)**2.0).sum()
    print(f'Step {i}...')
    print(popt, np.sqrt(pcov.diagonal()))
    print(f'Chisquare = {chisq:.2f}')



    #grafico