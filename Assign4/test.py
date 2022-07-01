import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import integrate

def f(x):
    return np.exp(x);
def g(x):
    return np.cos(np.cos(x));

x = np.linspace(-2*np.pi, 4*np.pi, 200);

"""plt.figure(1)
plt.grid()
plt.semilogy(x, f(x), label="main")
plt.title("semilog plot of f(x)")
plt.xlabel("x")
plt.ylabel("log(f(x))")
plt.show()

plt.figure(2)
plt.grid()
plt.plot(x, g(x), label="main")
plt.title("plot of g(x)")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.show()
"""

def u(x,k):
    return f(x)*np.cos(k*x)
def v(x,k):
    return f(x)*np.sin(k*x)

def w(x,k):
      return g(x)*np.cos(k*x)
def y(x,k):
    return g(x)*np.sin(k*x)

fa = []
fb = []
ga = []
gb = []

for i in range(26):
    fa.append(sp.integrate.quad(u,0,2*np.pi, args=(i))[0])
    ga.append(sp.integrate.quad(w,0,2*np.pi, args=(i))[0])

for i in range(25):
    fb.append(sp.integrate.quad(v,0,2*np.pi, args=(i+1))[0])
    gb.append(sp.integrate.quad(y,0,2*np.pi, args=(i+1))[0])

fa = np.array(fa)/np.pi
fb = np.array(fb)/np.pi

ga = np.array(ga)/np.pi
gb = np.array(gb)/np.pi

coeffd = np.zeros(51)
coefgd = np.zeros(51)
coeffd[0] = fa[0]/2
coefgd[0] = ga[0]/2

j=1
for i in range(1,50,2):
    coeffd[i] = fa[j]
    coeffd[i+1] = fb[j-1]

    coefgd[i] = ga[j]
    coefgd[i+1] = gb[j-1]
    j= j+1
    

n = [i for i in range(51)]
"""
plt.figure(3)
plt.semilogy(n, abs(coeffd), "o", color="r")
plt.xlabel("n")
plt.ylabel("Coefficients")
plt.title("semilog plot of coefficients of f(x)")
plt.grid()

plt.figure(4)
plt.loglog(n, abs(coeffd), "o", color="r")
plt.xlabel("n")
plt.ylabel("Coefficients")
plt.title("loglog plot of coefficients of f(x)")
plt.grid()


plt.figure(5)
plt.semilogy(n, abs(coefgd), "o", color="r")
plt.xlabel("n")
plt.ylabel("Coefficients")
plt.title("semilog plot of coefficients of g(x)")
plt.grid()

plt.figure(6)
plt.loglog(n, abs(coefgd), "o", color="r")
plt.xlabel("n")
plt.ylabel("Coefficients")
plt.title("loglog plot of coefficients of g(x)")
plt.grid()

plt.show()
"""

""" -----------------------------------------------------------------------"""

            #Ac = b
x=np.linspace(0,2*np.pi,401)
x=x[:-1]            

bf = f(x)
bg = g(x)
A = np.zeros((400,51))  
A[:,0]=1            

for k in range(1,26):
    A[:,2*k-1]=np.cos(k*x)     
    A[:,2*k]=np.sin(k*x)       
                        
coeffl = np.linalg.lstsq(A,bf, rcond=-1)[0]        
coefgl = np.linalg.lstsq(A,bg, rcond=-1)[0]
"""
n = [i for i in range(51)]
plt.figure(7)
plt.plot(n, coeffl, "o", color="g")
plt.title("Coefficients of f(x) by least square method")
plt.xlabel("n")
plt.ylabel("coefficients")


plt.figure(8)
plt.plot(n, coefgl, "o", color="g")
plt.title("Coefficients of g(x) by least square method")
plt.xlabel("n")
plt.ylabel("coefficients")
plt.show()
"""
"""
plt.figure(3)
plt.semilogy(n, coeffl, "o", color="g")
plt.figure(4)
plt.loglog(n, coeffl, "o", color="g")

plt.figure(5)
plt.semilogy(n, coefgl, "o", color="g")
plt.figure(6)
plt.loglog(n, coefgl, "o", color="g")
plt.show()
"""

""" -------------------------------------------------------------------"""


devf = abs(abs(coeffl) - abs(coeffd))
devg = abs(abs(coefgl) - abs(coefgd))

maxdevf = devf.max()
maxdevg = devg.max()

print(maxdevf)
print(maxdevg)

Fls = A.dot(coeffl)
Gls = A.dot(coefgl)

plt.figure(10)
plt.plot(x,Fls, label="least squares")
plt.plot(x,f(x), label="orginal")
plt.title("Comparing f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()

plt.figure(11)
plt.plot(x, Gls, label='least squares')
plt.plot(x, g(x), label='original')
plt.title("Comparing g(x)")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.legend()
plt.show()











