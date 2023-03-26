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