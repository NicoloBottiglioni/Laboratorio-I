import numpy as np

def weighted_average(w, sigma_w):
    """Implementation of the weighted average using numpy.
    """
    c = 1.0 / sigma_w**2.0
    w_hat = np.average(w, weights=c)
    sigma_w = np.sqrt(1.0 / np.sum(c))
    return w_hat, sigma_w


#media pesata per l'oscillazione in fase 
n = np.array([4.449839295742414, 4.450234187105906])
sigma_n = np.array([0.00037481588265089277, 0.0003096729676114258])
wf_hat, sigma_wf = weighted_average(n, sigma_n)
print(f'pulsazione della fase = {wf_hat:.4f} \pm {sigma_wf:.4f}')

#media pesata per l'oscillazione in controfase
n2 = np.array([4.622411044558177, 4.623184878206334])
sigma_n2 = np.array([0.00020694280088758355, 0.0001918953172394255])
wc_hat, sigma_wc = weighted_average(n2, sigma_n2)
print(f'pulsazione della controfase={wc_hat:.4f} \pm {sigma_wc:.4f}')

#media pesata per la pulsazione modulante
n3 = np.array([0.08772466555604169, 0.08073571765306227])
sigma_n3 = np.array([0.0001689927075853065, 0.00038329123021060054])
wb_hat, sigma_wb = weighted_average(n3, sigma_n3)
print(f'pulsazione modulante={wb_hat:.4f} \pm {sigma_wb:.4f}')

#media pesata per la pulsazione portante
n4 = np.array([4.537879287184062, 4.536955364647757])
sigma_n4 = np.array([0.00022442446124850448, 0.0002598918961176012])
wp_hat, sigma_wp = weighted_average(n4, sigma_n4)
print(f'pulsazione portante ={wp_hat:.4f} \pm {sigma_wp:.4f}')


