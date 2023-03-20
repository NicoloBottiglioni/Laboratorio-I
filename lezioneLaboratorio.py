import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


x= np.array([1., 2., 3., 4., 5., 6., 7])
y= np.array([2.12, 4.02, 5.78, 8.22, 10.23, 11.88, 13.76])
dy=np.full(y.shape, 0.25)


def f(x, m):
    return m*x

def chisq(x, y, dy, m):
   return(((y - f(x, m) / dy)**2.).sum())

plt.figure('Dati')
plt.errorbar(x, y, dy, fmt='.')
plt.xlabel('x [u.a.]')
plt.ylabel('y [u. a.]')

for m in (1., 1.5, 2., 2.5, 3.):
    chi2= chisq(x, y, dy, m)
    plt.plot(x, f(x, m), label=f'm={m}, $\\chi^2$ = {chi2:.2f}')
    print(chisq(x, y, dy, m))
plt.legend()

plt.figure('Chi quadro')
m= np.linspace(1., 3., 100)
chi2 = np.array([chisq(x, y, dy, m_) for m_ in m])
plt.plot(m, chi2)
plt.xlabel('m')
plt.ylabel('$\\chi^2 (m)$')


plt.show()

-------------------------------------------------------------------------------------
def f(x, omega):
    return np.sin(omega*x)

omega0= 2.2
n =10
x= np.linspace(0., 10., n)
y= f(x, omega0)
dy=np.full(y.shape, 0.05)
y += np.random.normal(0., dy)

xgrid =  np.linspace(x.min(), x.max(), wigfuehviwhfuw))

popt, pcov = curve_fit(f, x, y, sigma=dy)

