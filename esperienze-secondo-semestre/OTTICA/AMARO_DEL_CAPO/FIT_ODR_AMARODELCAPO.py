import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

#array con le misure  in METRI
q= np.array([8.8, 10.9, 13.5, 17.1, 20.4, 23.8, 33.7])/100
p= np.array([6.6, 6.2, 5.9, 5.4, 5.2, 5.1, 4.9])/100
q=q + 0.011 #ho aggiubto la distanza delllo stronzo dallo stronzo

dq= np.full(q.shape, 0.2)/100
dp=np.full(p.shape, 0.2)/100

y= 1/q
x=1/p
dy= dq/(q**2)
dx= dp/(p**2)

#modello di fit
def line(pars, x):
    # Note the independent variable is the last argument.
    return pars[0] * x + pars[1]

model = odr.Model(line)
data = odr.RealData(x, y, sx=dx, sy=dy)
alg = odr.ODR(data, model, beta0=(1.0, 1.0))
out = alg.run()
m_hat, Q_hat = out.beta
sigma_m, sigma_Q = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square
# Print the fit output.
print(f'm = {m_hat:.3f} \pm {sigma_m:.3f}')
print(f'Q = {Q_hat:.3f} \pm {sigma_Q:.3f}')
print(f'Chisquare = {chisq:.1f}')

#Chiquadroatteso
print('Chi atteso', 5, '\pm', np.sqrt(10))

#residui
res= np.sqrt(out.delta**2 + out.eps**2)
print(res)
res=np.array(0.18523529, -0.13545801, -0.42386945, 0.12575791, 0.20760552, 0.11011799, 0.05631031)

fig=plt.figure('amarodelcapo', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))


ax1.errorbar(x, y, dy, dx, fmt='.', label='punti sperimentali')

ax1.plot(x, line(out.beta, x), label='Modello di best-fit')

ax1.set_ylabel('1/q [m]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(x, res, dy, fmt='.')


ax2.plot(x, np.full(x.shape, 0.0))

ax2.set_xlabel('1/p [m]')
ax2.set_ylabel('Residuai [m]')
ax2.grid(color='lightgray', ls='dashed')


plt.ylim(-1, 1)
plt.xlim(14.63, 21.34)
fig.align_ylabels((ax1, ax2))


#indice di rifrazione amaro del capo. convertimo tutto in cm
f= 1/Q_hat*100 #potere diottrico in cm
df= (sigma_Q/Q_hat**2)*100
R=2.4
dR= 0.1 #cm
n= R/(2*f) + 1
dn= (n-1)*np.sqrt((dR/R)**2 + (df/f)**2)/2

print('indcie amaro del capo', n, '\pm', dn)
plt.show()
 
