import  numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
g = 9.81
l = [68.1,64,60.3,55.5,51.1,46.8]
sigma_l = np.full(6,1)/(100*np.sqrt(12))
T = [[16.92,16.84,16.75,16.73,16.73],
[16.19,16.29,16.23,16.27,16.30],
[15.64,15.62,15.69,15.79,15.74],
[15.03,15.07,15.17,15.07,15.13],
[14.39,14.44,14.57,14.46,14.49],
[13.8,13.67,13.78,13.84,13.61]]

T = np.matrix(T)
T = T/10
sigma_T = T.std(1)/np.sqrt(5)
T = np.mean(T,axis = 1)
l = np.array(l)
l = l/100
print(T)
print(sigma_T)
sigma_T = np.array([0.0033538,0.00182428,0.00280856,0.00221991,0.00266833,0.00384708])
print(T)
T = np.array([1.6794,1.6256,1.5696,1.5094,1.447 ,1.374 ])
def Periodo(l,theta):
    return 2*np.pi*(np.sqrt(l/g))*(1 + theta/16)


fig = plt.figure("T - Lunghezza")
fig.add_axes((0.1,0.35,0.8,0.6))
plt.title("Grafico Periodo - Lunghezza")

popt,pcov = curve_fit(Periodo,l,T,sigma = sigma_T,absolute_sigma= True, p0 = (0.14))
theta_hat=  popt
sigma_theta = np.sqrt(pcov.diagonal())
x = np.linspace(0,0.75,1000)
plt.plot(x,Periodo(x,theta_hat))
print("bella")
print(theta_hat, sigma_theta)

theta_hat = np.full(6,theta_hat)
sigma_T = np.sqrt( sigma_T**2 + (sigma_l**2)*(np.pi*(1 + theta_hat**2/16)/(g*np.sqrt(l/g)))**2)

popt,pcov = curve_fit(Periodo,l,T,sigma = sigma_T,absolute_sigma= True, p0 = (0.14))
theta_hat=  popt
sigma_theta = np.sqrt(pcov.diagonal())
x = np.linspace(0,0.75,1000)
plt.plot(x,Periodo(x,0.145))

plt.errorbar(l,T,sigma_T,sigma_l,fmt = '.')
plt.grid(which = 'both', ls= 'dashed', color = 'gray')
plt.xlabel('Lunghezza [m]')
plt.ylabel('Periodo [s]')
plt.xlim()

res = T - Periodo(l,theta_hat)
fig.add_axes((0.1, 0.1, 0.8, 0.18))
plt.errorbar(l,res,sigma_T,fmt='.')
plt.plot(x,np.zeros(1000), ls = 'dashed')
plt.grid(which = 'both', ls= 'dashed', color = 'gray')
plt.xlabel('Lunghezza [m]')
plt.ylabel('Residui [s]')

plt.show()

