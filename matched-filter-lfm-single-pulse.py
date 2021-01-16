#best Range compression pulse compression using matched filtering
from scipy.fftpack import fft, ifft, fftshift
#from scipy import conj, linspace, exp
from scipy.constants import pi
from matplotlib import pyplot as plt
from numpy import linspace,exp,conj
import numpy
import numpy as np 
from numpy.fft import fft, fftshift
import math 


########## Gaussian Noise #############
mean = 0
std = 1 
num_samples = 400 
GWN = numpy.random.normal(mean, std, size=num_samples)


##########################################

t = linspace(-2.5e-6, 2.5e-6, 400) # Pulsewidth (s)
R=4e2                #Range
c = 3e8                # speed of EM wave [m/s] 
t0 = 2*R/c # Time delay to the target (s)
fc = 8e8              # Center frequency of chirp [Hz] 
BWf = 10e6            # Frequency bandwidth of chirp [Hz]  
T1 =5e-6              # Pulse duration of single chirp [s] 
#generating chirp/lfm signals
Kchirp = BWf/T1;              #chirp pulse parameter 
st = exp(1j*2*pi*(fc*t+Kchirp/2*(t**2))) # transmited signal 
sr=exp(1j*2*pi*(fc*(t-t0)+Kchirp/2*(t-t0)**2)) # recieved signal 
Hf = fft(conj(st))
Si = fft(sr)
so = fftshift(ifft(Si * Hf))

####### Adding White Noise to Signal ####
srr = sr.real + GWN

########### Hanning Window ###########
new = abs(srr)
NNN = len(abs(srr))
window = [0]*NNN
for i in range(NNN):
    window[i] =  0.5*(1 - math.cos(2*3.14*(new[i]/NNN)))

    
# Plot the matched filter output
plt.figure(1)
plt.plot(t, abs(so))
plt.title('Matched Filter Output')
plt.xlabel('Time Delay (s)')
plt.ylabel('Amplitude')
plt.show()
plt.figure(1)
plt.plot(t, st.real)
plt.title('transmitted signal')
plt.xlabel('Time Delay (s)')
plt.ylabel('real part')
plt.show()
plt.figure(1)
plt.plot(t, srr)
plt.title('recieved signal')
plt.xlabel('Time Delay (s)')
plt.ylabel('real part')
plt.show()
