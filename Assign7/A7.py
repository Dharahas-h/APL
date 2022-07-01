import numpy as np
import sympy as sp
import scipy.signal as sg
import matplotlib.pyplot as plt

s= sp.symbols('s')
"""
def lowpass(R1,R2,C1,C2,G,Vi):
    A = sp.Matrix([[0,0,1,-1/G], [-1/(s*R2*C2),1,0,0], [0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b = sp.Matrix([0,0,0,Vi/R1])
    V = A.inv()*b
    
    return (A,b,V)

A,b,V = lowpass(10000,10000,1e-9,1e-9,1.586,1)
Vo = V[3]

ww=np.logspace(0,8,801)
ss=1j*ww
hf=sp.lambdify(s,Vo,'numpy')
v=hf(ss)
"""
"""
plt.figure()
plt.title("Transfer function of LPF")
plt.xlabel(r'$\omega\rightarrow$')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.loglog(ww,abs(v),lw=2)
plt.grid(True)
plt.show()

H = sg.lti([-1.586e-4], [2e-14, 4.414e-9, 2e-4])
t = np.linspace(0, 0.001, 200000)
u = (t>0)
t,y,_ = sg.lsim(H,u,t)

plt.figure()
plt.title("Step response of LPF")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,u, label="input")
plt.plot(t,y, label="output")
plt.legend()
plt.grid()
plt.show()
"""

"""
t = np.linspace(0, 0.0008,30000)
Vi_l = np.sin(2000*np.pi*t)
Vi_h = np.cos(2e5*np.pi*t)

plt.figure()
plt.title("The input signal")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t, Vi_l, label=r'$Low\;\omega$')
plt.plot(t, Vi_h, label=r'$High\;\omega$')
plt.legend()
plt.grid()
plt.show()

H = sg.lti([-1.586e-4], [2e-14, 4.414e-9, 2e-4])
t,y,_ = sg.lsim(H,Vi_l + Vi_h, t)

plt.figure()
plt.title("Time domain response to the input signal")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$Vo\rightarrow$')
plt.grid()
plt.plot(t,y)
plt.show()
"""

def highpass(R1,R2,C1,C2,G,Vi):
    A = sp.Matrix([[0,0,1,-1/G], [-1/(1+1/(s*R2*C2)),1,0,0],\
                  [0,-G,G,1], [s*C1-s*C2-1/R1,s*C2,0,1/R1]])
    b = sp.Matrix([0,0,0,Vi*s*C1])
    V = A.inv()*b
    
    return (A,b,V)

A,b,V = highpass(1e4,1e4,1e-9,1e-9,1.586,1)
Vo = V[3]
ww=np.logspace(0,8,801)
ss=1j*ww
hf=sp.lambdify(s,Vo,'numpy')
v=hf(ss)
"""
plt.figure()
plt.title("Frequency response of HPF")
plt.xlabel(r'$\omega\rightarrow$')
plt.ylabel(r'$|H(j\omega)|\rightarrow$')
plt.semilogx(ww,abs(v),lw=2)
plt.grid(True)
plt.show()
"""
t = np.linspace(0,0.0001, 20000)
Vi_l = np.sin(2000*np.pi*t)*np.exp(-50000*t)
Vi_h = np.cos(2e5*np.pi*t)*np.exp(-50000*t)
"""
plt.figure()
plt.title("Input signal")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t, Vi_l, label=r'$Low\;\omega$')
plt.plot(t, Vi_h, label=r'$High\;\omega$')
plt.legend()
plt.grid()
plt.show()
"""
H = sg.lti([-1.586e-9,0,0],[2e-9,4.414e-4,20])
t,y,_ = sg.lsim(H, Vi_l + Vi_h, t)
"""
plt.figure()
plt.title("Response to the damping signal")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$Vo\rightarrow$')
plt.plot(t,y)
plt.grid()
plt.show()
"""
u = (t>0)
t,y,_ = sg.lsim(H, u, t)

plt.figure()
plt.title("Step response of the HPF")
plt.xlabel(r'$t\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.plot(t,u, label="input" )
plt.plot(t,y, label="output")
plt.legend()
plt.grid()
plt.show()



