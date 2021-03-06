import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import integrate


pi = np.pi
e = np.e


def f(x):
    return np.exp(x)


def g(x):
    return np.cos(np.cos(x))


x = np.linspace(-2*pi,4*pi,100)
plt.grid()
plt.plot(x,g(x))
plt.title('Plot of g(x)')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.show()


plt.grid()
plt.semilogy(x,f(x))
plt.title('Plot of f(x) in semilog scale')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


plt.grid()
plt.plot(x,f(x))
plt.title('Plot of f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


def u(x,k):
    return f(x)*np.cos(k*x)
def v(x,k):
    return f(x)*np.sin(k*x)
def w(x,k):
    return g(x)*np.cos(k*x)
def z(x,k):
    return g(x)*np.sin(k*x)


fa = np.zeros(26,)
fb = np.zeros(25,)
ga = np.zeros(26,)
gb = np.zeros(25,)


for i in range(26):
    fa[i],_ = sp.integrate.quad(u,0,2*pi,(i,))
    ga[i],_ = sp.integrate.quad(w,0,2*pi,(i,))
for i in range(25):
    fb[i],_ = sp.integrate.quad(v,0,2*pi,(i+1,))
    gb[i],_ = sp.integrate.quad(z,0,2*pi,(i+1,)) 


fa /= pi
fb /= pi
ga /= pi
gb /= pi
fa[0] /= 2
ga[0] /= 2


F = [None]*(len(fa)+len(fb))
F[0] = fa[0]
F[1::2] = fa[1:]
F[2::2] = fb
F = np.asarray(F)
# print(F)


plt.grid()
plt.semilogy(abs(F),'o',color = 'r',markersize = 4)
plt.title('Fourier Coefficients for f(x) by direct integration')
plt.xlabel('n')
plt.ylabel('Fourier Coefficients') 
plt.show()


plt.grid()
plt.loglog(abs(F),'o',color = 'r',markersize = 4)
plt.title('Fourier Coefficients for f(x) by direct integration')
plt.xlabel('n')
plt.ylabel('Fourier Coefficients') 
plt.show()


G = [None]*(len(fa)+len(fb))
G[0] = ga[0]
G[1::2] = ga[1:]
G[2::2] = gb
G = np.asarray(G)
# print(G)


plt.grid()
plt.semilogy(abs(G),'o',color = 'r',markersize = 4)
plt.title('Fourier Coefficients for g(x) by direct integration')
plt.xlabel('n')
plt.ylabel('Fourier Coefficients') 
plt.show()


plt.grid()
plt.loglog(abs(G),'o',color = 'r',markersize = 4)
plt.title('Fourier Coefficients for g(x) by direct integration')
plt.xlabel('n')
plt.ylabel('Fourier Coefficients') 
plt.show()
