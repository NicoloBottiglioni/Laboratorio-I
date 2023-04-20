import numpy as np
from scipy import odr
from matplotlib import pyplot as plt

def fit_model(pars, x):
    # Note the independent variable is the last argument.
    return pars[0] * x + pars[1]

p = np.array([-6.48, -8.57, -10.08, -11.51, -12.77, -13.34, -14.23, -14.67, -15.08, -15.30])/100  #metri
q = np.array([9.66, 17.21, 18.13, 25.81, 33.59, 40.55, 44.48, 52.29, 58.00, 60.24])/100 #metri
sigma_p = np.full(p.shape, 0.5)/100 #cm
sigma_q = np.full(q.shape, 1.00)/100


#------------------------------------------------------------------------------------------------------------------------------
inv_p= 1/p
inv_q= 1/q

sigma_inv_q= sigma_p/(p**2)
sigma_inv_p= sigma_q/(q**2)
#--------------------------------
x=-inv_p
y=inv_q
dx=sigma_inv_p
dy= sigma_inv_q
#----------------------
model = odr.Model(fit_model)
data = odr.RealData(x, y, sx=dx, sy=dy)
alg = odr.ODR(data, model, beta0=(1.0, 1.0))
out = alg.run()
m_hat, Q_hat = out.beta
sigma_m, sigma_Q = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square
# Print the fit output.
print(f'm = {m_hat:.3f} +/- {sigma_m:.3f}')
print(f'Q = {Q_hat:.3f} +/- {sigma_Q:.3f}')
print(f'Chisquare = {chisq:.1f}')


res= np.sqrt(out.delta**2 + out.eps**2)
print(res)
res=np.array([0.02878365, -0.70474565, 0.43817111, 0.10154004, 0.04734974, -0.12936333, 0.09889918, -0.02762256, -0.03583935, -0.00779149])

fig=plt.figure('Focale metodo odr', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))


ax1.errorbar(x, y, dy, dx, fmt='.', label='punti sperimentali', color='darkslateblue')

ax1.plot(x, fit_model(out.beta, x), label='Modello di best-fit', color='lightsteelblue')

ax1.set_ylabel('1/q [m]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(x, res, dy, fmt='.', color='darkslateblue')


ax2.plot(x, np.full(x.shape, 0.0), color='lightsteelblue')

ax2.set_xlabel('1/p [m]')
ax2.set_ylabel('Residuai [m]')
ax2.grid(color='lightgray', ls='dashed')


plt.ylim(-1.8, 1.8)
fig.align_ylabels((ax1, ax2))
plt.show()