'''Calcolo della media e della deviazione standard campione per quattro serie di misure
distinte—che potrebbero rappresentare, ad esempio, misure ripetute del
periodo di un pendolo per 4 valori
diversi della lunghezza del pendolo
stesso. Organizzando i dati in modo
opportuno, ovverosia come una matrice 4 x 8 in cui ciascuna riga corrisponde ad un valore specifico di lunghezza, si possono calcolare le medie
e le deviazioni standard riga per riga
utilizzando l'argomento axis=1 delle
funzioni statistiche di numpy'''

import numpy as np

T = [[1.53, 1.46, 1.44, 1.63, 1.38, 1.49, 1.50, 1.16],
     [1.88, 1.75, 1.76, 1.82, 1.66, 1.92, 1.62, 1.79],
     [2.16, 2.10, 1.90, 2.00, 2.12, 2.08, 1.99, 2.01],
     [2.32, 2.15, 2.27, 2.35, 2.17, 2.24, 2.29, 2.18]]

mean = np.mean(T, axis=1)
stdev = np.std(T, ddof=1, axis=1)

print(mean)
print(stdev)


#--------------------------------------------------------------------------------------------------------------------------------
'''Calcolo della media
e della deviazione standard campione utilizzando le funzioni di numpy
descritte in appendice F. Si noti l'uso dell'argomento ddof=1 per ottenere l'estimatore imparziale sn-1,
in assenza del quale la funzione
restituirebbe sn.
'''
import numpy as np

x = [2.53, 2.58, 2.28, 2.41, 2.61, 2.87, 2.40, 2.38, 2.34]

mean = np.mean(x)
stdev = np.std(x, ddof=1)

print(mean, stdev)

#--------------------------------------------------------------------------------------------------------------------

#per ottenre la edviazione std della media basta che divido quella del campione per radice di n

#-------------------------------------------------------------------------------------------------------------------
'''Frammento di codice
per il calcolo della funzione cumulativa di
una distribuzione di Gauss in forma standard, a partire dalla funzione erf() del
modulo math di Python. (Per completezza, il modulo scipy.special offre un'implementazione alternativa della funzione
degli errori che può operare direttamente
su array di numpy.) Si confrontino i valori forniti in output dal programma con
quelli riportati nella (206), oppure nella
tabella in appendice A.1.'''

import numpy as np

def Phi(z):
    """ Gaussian cumulative function.
    """
    return 0.5 + 0.5 * np.math.erf(z / np.sqrt(2.0))

def integrate_gauss(x1, x2, mu=0.0, sigma=1.0):
    """Integrate a generic gaussian between x1 and x2.
    """
    z1 = (x1 - mu) / sigma
    z2 = (x2 - mu) / sigma
    return Phi(z2) - Phi(z1)

print(integrate_gauss(-1.0, 1.0))
print(integrate_gauss(22.0, 24.0, 20.0, 4.0))

#__________--------------------------------------------------------------------------------------

'''. Frammento di codice per il calcolo della media pesata
delle misure dell'indice di rifrazione
dell'acqua dell'esempio 8.3. La funzione prende in ingresso le liste dei
valori delle misure e degli errori associati e restituisce la media pesata e
l'incertezza associata. Il risultato è illustrato graficamente nella figura 8.2.
Notiamo, per completezza, che l'incertezza sigma_q sulla media pesata è più
piccola del più piccolo tra gli errori
di misura.
'''
import numpy as np
def mediapesata(y, sigma_y):
    w=1/sigma_y**2 #peso
    q_hat= np.avarage(y, weights=w)
    sigma_q= np.sqrt(1/np.sum(w))
    return q_hat, sigma_q

n=np.array([])
sigma_n= np.array([])
q_hat, sigma_q= mediapesata(n, sigma_n)
print(f'q = {q_hat:.4f} +/- {sigma_q:.4f}')


#----------------------------------------------------------

#modello e fit con gli errori efficaci
def line(x, m, q):
    return m*x +q

popt, pcov= curve_fit(line, x, y, sigma=dy, absolute_sigma= True)
for i in range(4):
    sigma_eff = np.sqrt(dy**2.0 + (popt[0] * dx)**2.0) #NOTARE  al posto di popt ci andrebbe la derivata. qui abbiamo messo popt perchè il modello è lineare, la sua derivata è il coefficiente angolare il quale a sua volta è l'elemento 0 dell'array popt. Quindi, qui, abbiamo tecnicamente messo la derivata.
    popt, pcov = curve_fit(line, x, y, sigma=sigma_eff, absolute_sigma=True)
    chisq = (((y - line(x, *popt)) / sigma_eff)**2.0).sum()
    print(f'Step {i}...')
    print(popt, np.sqrt(pcov.diagonal()))
    print(f'Chisquare = {chisq:.2f}')