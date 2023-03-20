import numpy as np
import pylab
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

tr1,Tr1,ta1,Ta1=pylab.loadtxt(r'C:\Users\nicob\OneDrive\Desktop\Laboratorio I\conducibilità termica\BlualluminioRossorame.txt',unpack=True)
ta2,Ta2,tr2,Tr2=pylab.loadtxt(r'C:\Users\nicob\OneDrive\Desktop\Laboratorio I\conducibilità termica\BlurameRossoalluminio.txt',unpack=True)

pylab.figure(0)

#pylab.errorbar(ta1,Ta1,marker='', color='blue')
pylab.errorbar(ta2,Ta2,marker='', color='red')
plt.xlabel('Tempo [s]')
plt.ylabel('Temperatura [$^\\circ$C]')
pylab.grid()


pylab.show()