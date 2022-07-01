import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

H = sp.lti([1], [10**-12, 10**-4, 1])

t = np.linspace(0,30*(10**-6), 1000)
u = np.cos(1000*t) - np.cos((10**6)*t)
t,y,_ = sp.lsim(H, u, t)

plt.figure()
plt.plot(t,y)
plt.grid()
plt.title("Response of the system")
plt.ylabel("y")
plt.xlabel("t")
plt.show()

t = np.linspace(0,10*(10**-3), 100000)
u = np.cos(1000*t) - np.cos((10**6)*t)
t,y,_ = sp.lsim(H, u, t)

plt.figure()
plt.plot(t,y)
plt.grid()
plt.title("Response of the system")
plt.ylabel("y")
plt.xlabel("t")
plt.show()
