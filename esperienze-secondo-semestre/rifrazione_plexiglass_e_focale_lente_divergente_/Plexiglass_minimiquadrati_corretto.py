import numpy as np
from scipy.optimize import curve_fit

def fit_model(x, m, q):
    return m * x + q

x, y = np.loadtxt(r"C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\rifrazione_plexiglass_e_focale_lente_divergente_\Dati_rifrazione_plexiglass.txt", unpack=True)
dx= np.full(x.shape, 1) #quadretti
dy= np.full(y.shape, 1) #quadretii
# Run a first least-square fit ignoring the errors on x.
popt, pcov = curve_fit(fit_model, x, y, sigma=dy)
# Iteratively update the errors and refit.
sigma_eff = np.sqrt(dy**2.0 + (popt[0] * dx)**2.0)
popt, pcov = curve_fit(fit_model, x, y, sigma=sigma_eff)
chisq = (((y - fit_model(x, *popt)) / sigma_eff)**2.0).sum()
 # Print the fit output at each step.
print(popt, np.sqrt(pcov.diagonal()))
print(f'Chisquare = {chisq:.2f}')

n_hat, q_hat = popt
dn, dq = np.sqrt(pcov.diagonal())
na_hat= 1/n_hat
dna= dn/((n_hat)**2)


print('INDICE DI RIFRAZIONE:', na_hat, dna, 'INTERCETTA:', q_hat, dq)
