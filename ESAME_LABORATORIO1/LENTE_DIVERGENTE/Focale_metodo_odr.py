import numpy as np
from scipy import odr
from matplotlib import pyplot as plt


#dati sperimentali
p = np.array([-6.48, -8.57, -10.08, -11.51, -12.77, -13.34, -14.23, -14.67, -15.08, -15.30])/100  #metri
q = np.array([9.66, 17.21, 18.13, 25.81, 33.59, 40.55, 44.48, 52.29, 58.00, 60.24])/100 #metri
sigma_p = np.full(p.shape, 0.5/np.sqrt(12))/100 
sigma_q = np.full(q.shape, 1.00)/100


#grandezze che userò nel fit
x= -1/p
y= 1/q

sigma_x= sigma_p/(p**2)
sigma_y= sigma_q/(q**2)

#Modello, fit e chisq
def fit_model(pars, x):
    return pars[0] * x + pars[1]

model = odr.Model(fit_model)
data = odr.RealData(x, y, sx=sigma_x, sy=sigma_y)
alg = odr.ODR(data, model, beta0=(1.0, 1.0))
out = alg.run()
m_hat, Q_hat = out.beta
sigma_m, sigma_Q = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square

print(f'm = {m_hat:.3f} +/- {sigma_m:.3f}')
print(f'Q = {Q_hat:.3f} +/- {sigma_Q:.3f}')
print(f'Chisquare = {chisq:.1f}')



#Residui. Stampo prima l'aray dei residui per poi modificarlo manualmente, dato che, non so il perchè, me li mette a cazzo di cane
res= np.sqrt(out.delta**2 + out.eps**2)
print(res)
res=np.array([0.02878365, -0.70474565, 0.43817111, 0.10154004, 0.04734974, -0.12936333, 0.09889918, -0.02762256, -0.03583935, -0.00779149])


#Grafico
fig=plt.figure('Focale metodo odr', figsize=(10., 6.), dpi=100)

ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(x, y, sigma_y, sigma_x, fmt='.', label='punti sperimentali', color='midnightblue')
ax1.plot(x, fit_model(out.beta, x), label='Modello di best-fit', color='deepskyblue')
ax1.set_ylabel('1/q [m^(-1)]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(x, res, sigma_y, fmt='.', color='midnightblue')
#np.linspace
ax2.plot(x, np.full(x.shape, 0.0), color='deepskyblue')
ax2.set_xlabel('1/p [m^(-1)]')
ax2.set_ylabel('Residui [m^(-1)]')
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
9)in ODR niente residui normalizzati
10)Quando si trova la posizione dello schermo per la quale l'immagine della convergente è a fuoco, segnarsi tale posizione 
con un appunto a matita.'''