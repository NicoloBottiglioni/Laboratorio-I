import wave

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


file_path= r"C:\Users\ACER\OneDrive\Desktop\rimbalzi\Nuovo memo 7.wav"
#apertura del file audio-leggere link su pdf dell'esperienza
stream= wave.open(file_path)
signal= np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)

#il seguente comando indica che se il file audio è tsereo dobbiamo prendere solo uno dei canali
if stream.getnchannels() == 2:
    signal = signal[::2]

#array dei tempi corrispondenti ai singoli campioni
t = np.arange(len(signal)) / stream.getframerate() + 2.0

#Grafico per l'analisi del file audio
plt.figure('Rimbalzi pallina')
plt.plot(t, signal)
plt.xlabel('tempo [s]')
plt.grid(which= 'both', ls='dashed', color='grey')
plt.show()
