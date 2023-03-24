import numpy as np
from scipy import odr

def fit_model(pars, x):
    # Note the independent variable is the last argument.
    return pars[0] * x + pars[1]

# Read the data from file.
x, y = np.loadtxt(r"C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\rifrazione_plexiglass_e_focale_lente_divergente_\Dati_rifrazione_plexiglass.txt", unpack=True)
dx= np.full(x.shape, 1) #quadretti
dy= np.full(y.shape, 1) #quadretii
# Run the actual ODR.
model = odr.Model(fit_model)
data = odr.RealData(x, y, sx=dx, sy=dy)
alg = odr.ODR(data, model, beta0=(1.0, 1.0))
out = alg.run()
m_hat, q_hat = out.beta
sigma_m, sigma_q = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square
# Print the fit output.
print(f'm = {m_hat:.3f} +/- {sigma_m:.3f}')
print(f'q = {q_hat:.3f} +/- {sigma_q:.3f}')
print(f'Chisquare = {chisq:.1f}')

na_hat= 1/m_hat
dna= sigma_m/((m_hat)**2)


print('INDICE DI RIFRAZIONE:', na_hat, dna, 'INTERCETTA:', q_hat, sigma_q)
err_rel_n = dna/na_hat
print('errore relativo', err_rel_n)