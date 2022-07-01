from pylab import *
import scipy.special as sp
import numpy as np


def g(t,A,B):
    m= A*sp.jn(2,t) + B*t
    return m

def getLegend(values):
        arr = []
        i=1
        for x in values:
            name = "sig"+str(i)+" = "+str(round(x,3))
            arr.append(name)
            i=i+1
        return arr


c = loadtxt("fitting.dat")
t = c[:,0]
data = c[:,1:]
sigma = logspace(-1 ,-3,9)
        

#shadow plot
figure(1)
plot(t, data , label=getLegend(sigma))
plot(t, g(t,1.05,-0.105), c='black', label='True value')
legend()
xlabel(r'$t$',size=20)
ylabel(r'$f(t)+n$',size=20)
title(r'Plot of the data to be fitted')
grid(True)


figure(2)
#errorbarplot
errorbar(t[::5], data[:,0][::5] ,sigma[0],fmt='ro', label='errorbar')
plot(t, g(t, 1.05, -0.105), c='black', label='True value')
xlabel(r"$t$", size=20)
ylabel(r"$f(t)+n$", size=20)
title("Data points of sigma=0.10 along with exact function")
grid(True)
legend()


#g(t,A,B) as M.b
y = np.array(t).T
x = np.array(sp.jn(2,t)).T
M = c_[x,y]
A = 1.05
B = -0.105
p = np.array([A, B]).T
gt = np.dot(M,p)

#if (gt == np.array(g(t,1.05,-0.105)) results in True then the vectors are equal

#mean squared error
E = zeros((21,21))
A = linspace(0,2,21)
B = linspace(-0.2, 0 , 21)
for i in range(21):
    for j in range(21):
        m = (c[:, 1] - g(t, A[i], B[j]))**2
        E[i,j] = m.sum()/101

figure(3)
#contour
cs = contour(A, B, E, 20)
scatter(1.05,-0.105)
annotate("Exact location",(1.05,-0.105))
title("Contour plot of Eij")
xlabel(r"$A$")
ylabel(r"$B$")


figure(4)
#error plot
AB = []
for i in range(9):
    AB.append(linalg.lstsq(M, c[:, i+1], rcond=1)[0])
AB = np.array(AB)
Ao = np.ones((9,1))*1.05
Bo = np.ones((9,1))*(-0.105)
MS = c_[Ao, Bo]
MSer = AB - MS
MSer = absolute(MSer)

plot(sigma, MSer[:,0], ls="--",marker=".", label="Aerr")
plot(sigma, MSer[:,1], ls="--",marker=".", label="Berr")
title("variation of error with noice")
xlabel("Noice standard deviation")
ylabel("MS error")
legend()


figure(5)
loglog(sigma, MSer[:,0], marker=".", ls="", label="Aerr")
loglog(sigma, MSer[:,1], marker=".", ls="", label="Berr")
errorbar(sigma, MSer[:,0], yerr = 0.1, fmt="ro")
errorbar(sigma, MSer[:,1], yerr=0.1,fmt="bo")
xlabel("sigmaN")
ylabel("MSerror")
title("Vaiation of error with noise")
legend()
show()






