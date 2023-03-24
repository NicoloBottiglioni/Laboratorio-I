import numpy as np
from scipy import odr
from matplotlib import pyplot as plt

def fit_model(pars, x):
    # Note the independent variable is the last argument.
    return pars[0] * x + pars[1]

p = np.array([-10.1, -12.1, -7.2, -5.3, -4.9, -8.2, -6.6, -9.0, -11.4, -6.0, -5.6 ])  #cm
q = np.array([19.6, 30.5, 17.05, 9.9, 7.55, 18.55, 12.75, 22.1, 29.25, 7.8, 10.5 ])
sigma_p = np.full(p.shape, 0.5) #cm
sigma_q = np.full(q.shape, 2.09)

#2.0, 2.8, 3.35, 2.0, 1.35, 1.65, 3.05, 1.9, 2.25, 0.6, 2.1
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


'''res= y - fit_model()
fig=plt.figure('Rifrazione Plexiglass', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))


ax1.errorbar(x, y, dy, fmt='.', label='punti sperimentali')
#xgrid = np.linspace(0.0, 10.0, 100)
ax1.plot(x, fit_model(), label='Modello di best-fit')
# Setup the axes, grids and legend.
ax1.set_ylabel('y [a. u.]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
# And now the residual plot, on the bottom panel.
ax2.errorbar(x, res, dy, fmt='.')
# This will draw a horizontal line at y=0, which is the equivalent of the best-fit
# model in the residual representation.
ax2.plot(x, np.full(x.shape, 0.0))
# Setup the axes, grids and legend.
ax2.set_xlabel('x [a. u.]')
ax2.set_ylabel('Residuals [a. u.]')
ax2.grid(color='lightgray', ls='dashed')

# The final touch to main canvas :-)
plt.ylim(-4.0, 4.0)
fig.align_ylabels((ax1, ax2))
plt.show()'''