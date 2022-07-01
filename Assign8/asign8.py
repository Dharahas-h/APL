import numpy as np
import matplotlib.pyplot as plt

"""
x = np.linspace(0,2*np.pi, 129)
x = x[0:-1]
y = np.sin(5*x)
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/128
w = np.linspace(-64,64,129)
w = w[0:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=2)
plt.ylabel(r"$|Y|\rightarrow$")
plt.xlim([-15,15])
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
ii=np.where(abs(Y)>1e-3)
plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Phase of $Y\rightarrow$")
plt.xlabel(r"$k\rightarrow$")
plt.grid(True)
plt.show()

x = np.linspace(-4*np.pi,4*np.pi, 513)
x = x[0:-1]
y = (1 + 0.1*np.cos(x))*np.cos(10*x)
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512
w = np.linspace(-64,64,513)
w = w[0:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w, abs(Y),lw=2)
plt.ylabel(r"$|Y|\rightarrow$")
plt.xlim([-15,15])
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=2)
ii=np.where(abs(Y)>1e-3)
plt.plot(w[ii],np.angle(Y[ii]),'go',lw=2)
plt.xlim([-15,15])
plt.ylabel(r"Phase of $Y\rightarrow$")
plt.xlabel(r"$k\rightarrow$")
plt.grid(True)
plt.show()

t = np.linspace(-4*np.pi, 4*np.pi, 513)
t = t[0:-1]
y = (np.sin(t))**3
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512
w = np.linspace(-64, 64, 513)
w = w[0:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w, abs(Y))
plt.xlim([-5,5])
plt.ylabel(r"$|Y|\rightarrow$")
plt.grid()

plt.subplot(2,1,2)
plt.plot(w, np.angle(Y), ".", markersize = 4)
ii = np.where(abs(Y) > 1e-3)
plt.plot(w[ii], np.angle(Y[ii]), "go", markersize = 5)
plt.xlabel(r"$w\rightarrow$")
plt.ylabel(r"Phase of $Y\rightarrow$")
plt.xlim([-5,5])
plt.grid()
plt.show()

t = np.linspace(-4*np.pi, 4*np.pi, 513)
t = t[0:-1]
y = (np.cos(t))**3
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512
w = np.linspace(-64, 64, 513)
w = w[0:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w, abs(Y))
plt.xlim([-5,5])
plt.ylabel(r"$|Y|\rightarrow$")
plt.grid()

plt.subplot(2,1,2)
plt.plot(w, np.angle(Y), ".", markersize = 4)
ii = np.where(abs(Y) > 1e-3)
plt.plot(w[ii], np.angle(Y[ii]), "go", markersize = 5)
plt.xlabel(r"$w\rightarrow$")
plt.ylabel(r"Phase of $Y\rightarrow$")
plt.xlim([-5,5])
plt.grid()
plt.show()

t = np.linspace(-4*np.pi, 4*np.pi, 513)
t = t[0:-1]
y = np.cos(20*t + 5*np.cos(t))
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512
w = np.linspace(-64, 64, 513)
w = w[0:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w, abs(Y))
plt.xlim([-30,30])
plt.ylabel(r"$|Y|\rightarrow$")
plt.grid()

plt.subplot(2,1,2)
plt.plot(w, np.angle(Y), ".", markersize = 4)
ii = np.where(abs(Y) > 1e-3)
plt.plot(w[ii], np.angle(Y[ii]), "go", markersize = 5)
plt.xlabel(r"$w\rightarrow$")
plt.ylabel(r"Phase of $Y\rightarrow$")
plt.xlim([-30,30])
plt.grid()
plt.show()
"""
t = np.linspace(-32,32,513)
t = t[0:-1]
y = np.exp(-(t**2)/2.0)
Y = np.fft.fft(y)
Y = np.fft.fftshift(Y)/512.0
w = np.linspace(-64,64, 513)
w = w[0:-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(w, abs(Y))
plt.xlim([-15,15])
plt.ylabel(r"$|Y|\rightarrow$")
plt.grid()

plt.subplot(2,1,2)
plt.plot(w, np.angle(Y), ".", markersize = 4)
ii = np.where(abs(Y) > 1e-3)
plt.plot(w[ii], np.angle(Y[ii]), "go", markersize = 5)
plt.xlabel(r"$w\rightarrow$")
plt.ylabel(r"Phase of $Y\rightarrow$")
plt.xlim([-15,15])
plt.grid()
plt.show()
