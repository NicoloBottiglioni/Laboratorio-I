import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#dati sperimentali in metri 
p = np.array([-6.48, -8.57, -10.08, -11.51, -12.77, -13.34, -14.23, -14.67, -15.08, -15.30])/100  #metri
q = np.array([9.66, 17.21, 18.13, 25.81, 33.59, 40.55, 44.48, 52.29, 58.00, 60.24])/100 #metri
sigma_p = np.full(p.shape, 0.5 )/100 #metri 
sigma_q = np.full(q.shape, 1.00 )/100 #metri

#definisco le grandezze che userò per il fit
x= -1/p #diottrie
y= 1/q #diottrie 

sigma_x= sigma_p/(p**2) #diottrie
sigma_y= sigma_q/(q**2) #diottrie

#modello di fit
def line(x, m, Q):
    return m*x + Q

#fit dei minimi quadrati con errori efficaci
popt, pcov= curve_fit(line, x, y, sigma=sigma_y, absolute_sigma= True)
for i in range(4):
    sigma_eff = np.sqrt(sigma_y**2.0 + (popt[0] * sigma_x)**2.0)
    popt, pcov = curve_fit(line, x, y, sigma=sigma_eff, absolute_sigma=True)
    chisq = (((y - line(x, *popt)) / sigma_eff)**2.0).sum()
    print(f'Step {i}...')
    print(popt, np.sqrt(pcov.diagonal()))
    print(f'Chisquare = {chisq:.2f}')

#Grafico e grafico dei residui
fig=plt.figure('poterediottrico_errori efficaci', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(x, y, sigma_y, sigma_x, fmt='.', label='punti sperimentali', color='midnightblue')
ax1.plot(x, line(x, *popt), label='Modello di best-fit', color='deepskyblue')
ax1.set_ylabel('1/q [m^(-1)]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

res= (y - line(x, *popt)) / sigma_y
ax2.errorbar(x, res, sigma_y, fmt='.', color='midnightblue')
ax2.plot(x, np.full(x.shape, 0.0), color='deepskyblue')
ax2.set_xlabel('1/p [m^(-1)]')
ax2.set_ylabel('Residu normalizzati')
ax2.grid(color='lightgray', ls='dashed')

plt.ylim(-1.8, 1.8)
fig.align_ylabels((ax1, ax2))
plt.show()


'''NOTE ESPERIENZA:
1)Metto la convergente sul banco ottico e cerco di posizionarla in modo tale da avere la sorgente, la quale ha una forma 
triangolare, posizionata sul fuoco della convergnete. In tale modo, essendo p=f, si ha q=inf. Se la convergente ha potere 
diottrico pari a 10, il fuoco si troverà  a 10 centimetri.
2) Metto la divergente tra lol schermo e la convergente. a distanza divergente-schermo è -p. Si prende con il srgno meno perchè 
la sorgente della lente divergente è virtuale e si trova dallo stesso lato dell'immagine. In particolare, la sorgente virtuale 
della divergete è l'immagine della convergente.
3)Allontano lo schermo fino a che l'immagine non è a fuoco. La nuova distanza schermo- divergente è q.
4)L'intercetta stimata dal fit è il mio potere diottrico, mentre la pendenza dovrebbe essere pari ad 1
5)Meglio scegliere dal set una divergente con potere -5 e una convergente con potere 10/12
6) Le variabili non sono distribuite gaussianamente. Quindi C ha media 0 e varianza 1. C^2 avrà media 1 e varianza ignota. 
Segue che il chi avrà media pari ai gradi di libertà ma varianza ingnota. Inoltre, dato che Chi non è somma di quadrati di 
variabili gaussiane, si ha che non è distribuito come un chi quadro. Non sappiamo quindi quale sia la distribuzione di chisq
e ciò non ci permette di valutare la bontà di un fit con il valore numerico del chisq.
Per il teorema centrale del limite sappiamo però che se il numero di misure è molto grande, il chisq sarà distribuito 
gaussianamente. Non avendo però misure, non è questo il nostro caso.
7) L' incertezza su p deriva dal fatto che la lente è all'interno di una ghiera. Quella su P deriva dal fatto che l'immagine
sullo schrmo è a fuoco in un intervallo e non in un punto preciso.
8)Anche se uso ODR, il chisq non ha comunque senso perchè non ho variabili gaussiane.
10)Quando si trova la posizione dello schermo per la quale l'immagine della convergente è a fuoco, segnarsi tale posizione 
con un appunto a matita'''
