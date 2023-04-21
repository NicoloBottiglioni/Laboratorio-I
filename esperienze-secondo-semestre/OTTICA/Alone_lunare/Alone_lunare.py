import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


#apertura dell'immagine
img= img.imread(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\OTTICA\Alone_lunare\Lunar_halo.jpg')
plt.imshow(img)

#import dei dati sperimentali con le relative incertezze
x, y = np.loadtxt(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\OTTICA\Alone_lunare\DATI_ALONE2.txt', unpack=True)

dx= np.full(x.shape, 4)
dy= np.full(y.shape, 4)

sigma= 4

#fit circolare per la stima dei parametri: raggio della circonferenza, coordiante del centro della circonferenza
def fit_circle(x, y, sigma):
    """Fit a series of data points to a circle.
    """
    n = len(x)
    # Refer coordinates to the mean values of x and y.
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    # Calculate all the necessary sums.
    s_u = np.sum(u)
    s_uu = np.sum(u**2.0)
    s_uuu = np.sum(u**3.0)
    s_v = np.sum(v)
    s_vv = np.sum(v**2.0)
    s_vvv = np.sum(v**3.0)
    s_uv = np.sum(u * v)
    s_uuv = np.sum(u * u * v)
    s_uvv = np.sum(u * v * v)
    D = 2.0 * (s_uu * s_vv - s_uv**2.0)
    # Calculate the best-fit values.
    u_c = (s_vv * (s_uuu + s_uvv) - s_uv * (s_vvv + s_uuv)) / D
    v_c = (s_uu * (s_vvv + s_uuv) - s_uv * (s_uuu + s_uvv)) / D
    x_c = u_c + x_m
    y_c = v_c + y_m
    r = np.sqrt(u_c**2.0 + v_c**2.0 + (s_uu + s_vv) / n)
    # Calculate the errors---mind this is only rigorously valid
    # if the data points are equi-spaced on the circumference.
    sigma_xy = sigma * np.sqrt(2.0 / n)
    sigma_r = sigma * np.sqrt(1.0 / n)
    return  x_c, y_c, r, sigma_xy, sigma_r

x_c, y_c, r, sigma_xy, sigma_r = fit_circle(x, y, sigma)

print(f'x_c = {x_c:.3f} +/- {sigma_xy:.3f}')
print(f'y_c = {y_c:.3f} +/- {sigma_xy:.3f}')
print(f'r = {r:.3f} +/- {sigma_r:.3f}')

#creazione della figura
fig = plt.figure('Alone_lunare')
plt.errorbar(x, y, dx, dy, fmt='.', color='tomato')
plt.xlabel('X [px]')
plt.ylabel('Y [px]')

plt.xlim([400, 840])
plt.ylim([150, 590])


theta= np.linspace(0, 2*np.pi, 360) 

a= x_c + r*np.cos(theta)
b= y_c + r*np.sin(theta)
plt.plot(a, b, color='chartreuse')
plt.grid()
plt.gca().set_aspect('equal')

#------------------------------------------------------------------------------------------------------------------------------------------
#residui in coordinate polari
res= r - np.sqrt((x_c - x)**2 + (y_c - y)**2)

PHI= np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

if (x_c < x and y_c < y):
    PHI=  np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

elif x_c > x and y_c < y:
    PHI=  np.pi/2 + np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

elif x_c > x and y_c > y:
    PHI=  np.pi + np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

else: 
    PHI=  3*np.pi/2 + np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

fig2= plt.figure('Residui_alone_lunare')
plt.errorbar(PHI, res, sigma, fmt='.', color='tomato')
plt.plot(PHI, np.full(PHI.shape, 0.0), color='chartreuse')
plt.grid(color='lightgray', ls='dashed')
plt.xlabel('Angolo campionamenti-orizzontale [$\circ$]')
plt.ylabel('Residui [pixel]')
#----------------------------------------------------------------------------------------------------------------
#coordinate delle due stelle in pixel
'''Posizione di Spica: 606 - 529   \pm 3'''
''' Posizione di Regulus: 701 - 61 \pm 3'''
'''Posizione di Arturo 325 - 445'''
'''Distanza angolare arturo spica:32.8'''
'''distanza angolare Arturo regulus: 59.7'''
'''Distanza angolare Regolo - Spica: 54.1'''

x1=325
y1=445
x2=701
y2=61
dx1=2
dx2=2
dy1=2
dy2=2

#distanza in pixel tra le due stelle
X= x1-x2
Y= y1-y2
dX=np.sqrt(dx1**2 + dx2**2)
dY=np.sqrt(dy1**2 + dy2**2)

d = np.sqrt((X)**2 + (Y)**2)
sigma_d=np.sqrt(((X*dX)**2 + (Y*dY)**2)/(X**2 + Y**2))

#distanza angolare tra le due stelle in gradi
a=59.7

#fattore di conversione
f= a/d #degrees/px
sigma_f=(a*sigma_d)/d**2

#raggio angolare in gradi
ra= r*f
sigma_ra= ra*np.sqrt((sigma_r/r)**2 + (sigma_f/f)**2)

#conversione raggio angolare da gradi a radianti
ra2= np.deg2rad(ra)
sigma_ra2= np.deg2rad(sigma_ra)

#indice di rifrazione del ghiaccio 
n = np.sin((ra2+ (np.pi/3))/2)/np.sin(np.pi/6)
sigma_n=(np.cos((ra2 + np.pi/3)/2)*sigma_ra2)

#print delle misure 
print('indice di rifrazione ghiaccio', n, '/pm', sigma_n)
print('Distanza stelle in pixel', d, '/pm', sigma_d)
print('Distanza angolare in gradi', ra, '/pm', sigma_ra)

plt.show()
