import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

#apertura dell'immagine
img= img.imread(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\OTTICA\Arcobaleno\Double_rainbow.jpg')
#plt.imshow(img)


#dati arcobaleno esterno
x, y = np.loadtxt(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\OTTICA\Arcobaleno\ROSSO_ESTERNO.txt', unpack=True)
dx= np.full(x.shape, 4)
dy=np.full(y.shape, 4)
sigma=4
#dati arcobaleno interno
x1, y1 =np.loadtxt(r'C:\Users\ACER\OneDrive\Desktop\Laboratorio I\esperienze-secondo-semestre\OTTICA\Arcobaleno\ROSSO_INTERNO.txt', unpack=True)
dx1=np.full(x1.shape, 4)
dy1=np.full(y1.shape, 4)
sigma1= 4

#fit circolare 
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

#fit esterno
x_c, y_c, r, sigma_xy, sigma_r = fit_circle(x, -y, sigma)

print(f'x_c = {x_c:.3f} \pm {sigma_xy:.3f}')
print(f'y_c = {y_c:.3f} \pm {sigma_xy:.3f}')
print(f'r = {r:.3f} \pm {sigma_r:.3f}')

#fit interno
x_c1, y_c1, r1, sigma_xy1, sigma_r1 = fit_circle(x1, -y1, sigma1)

print(f'x_c1 = {x_c1:.3f} \pm {sigma_xy1:.3f}')
print(f'y_c1 = {y_c1:.3f} \pm {sigma_xy1:.3f}')
print(f'r1= {r1:.3f} \pm {sigma_r1:.3f}')

#creazione delle figure
fig = plt.figure('Arcobaleno_esterno')
plt.errorbar(x, -y, dx, dy, fmt='.', color='black')
plt.xlabel('X [px]')
plt.ylabel('Y [px]')
theta= np.linspace(0, 2*np.pi, 360) 
A= x_c + r*np.cos(theta)
B= y_c + r*np.sin(theta)
plt.plot(A, B, color='limegreen')
plt.grid()
plt.gca().set_aspect('equal')


fig1 = plt.figure('Arcobaleno_interno')
plt.errorbar(x1, -y1, dx1, dy1, fmt='.', color='black')
plt.xlabel('X [px]')
plt.ylabel('Y [px]')
A1= x_c1 + r1*np.cos(theta)
B1= y_c1 + r1*np.sin(theta)
plt.plot(A1, B1, color='red')
plt.grid()
plt.gca().set_aspect('equal')

#plt.show()

#rapporto tra i raggi 
R=r/r1
dR=R*np.sqrt((sigma_r/r)**2 + (sigma_r1/r1)**2)


'''
#bisezione
def bisection(f, a, b, tol=1e-15, maxiter=10000):
 if f(a)*f(b) > 0:
  raise ValueError("f(a) and f(b) must have opposite signs")
 for i in range(maxiter):
  c = (a + b)/2
  if f(c) == 0 or (b - a)/2 < tol:
   return c
  if f(a)*f(c) < 0:
   b = c
  else:
   a = c
 raise RuntimeError("bisection method failed")

#f è la funzione ausiliaria. ovvero il rapporto tra i due raggi angolare, quello dell'arco maggiore e quello dell'arco minore, meno il rapporto dei raggi in pixel
f= lambda N: ((4*np.arcsin(np.sqrt((4 - N**2)/8*N**2)) - 2*np.arcsin(np.sqrt((4 - N**2)/8)))/(4*np.arcsin(np.sqrt((4 - N**2)/3*N**2)) - 2*np.arcsin(np.sqrt((4 - N**2)/3)))) - R
N= bisection(f, 1.327, 1.335)
dN=dR
print(f'Indice di rifrazione acqua {N} \pm {dN}')
'''

def CercaZero(f, a, b, c):
    if a >= b:
        return 'deve essere a < b!'
    if f(a)*f(b) >= 0:
        return 'usa degli a e b per cui f(a)*f(b) < 0'
    if f(a) == 0:
        return a
    if f(b) == 0:
        return b
    m = (a+b)/2
    if f(m) == 0:
        return m
    if b-a < 0.5*10**-c:
        return int(10**c*m)/10**c
    if f(a)*f(m) < 0:
        return CercaZero(f, a, m, c)
    else:
        return CercaZero(f, m, b, c)
f= lambda N: ((4*np.arcsin(np.sqrt((4 - N**2)/8*N**2)) - 2*np.arcsin(np.sqrt((4 - N**2)/8)))/(4*np.arcsin(np.sqrt((4 - N**2)/3*N**2)) - 2*np.arcsin(np.sqrt((4 - N**2)/3)))) - R 
N= CercaZero(f, 1.328, 1.333, 15)
print(N, dR)