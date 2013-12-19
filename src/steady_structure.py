import numpy as np
from scipy.integrate import ode
from scipy import optimize as sciopt
import matplotlib.pyplot as plt

def func_mcj(x):
    ms_squared = ((GAMMA - 1.0) * x**2 + 2.0) / \
        (2.0 * GAMMA * x**2 - (GAMMA - 1.0))
    ms = np.sqrt(ms_squared)

    a = (GAMMA * ms_squared + 1.0) / (GAMMA + 1.0)
    b = (ms_squared * 2.0 * GAMMA * (GAMMA - 1.0)) / \
        ((1.0 - a)**2  * (GAMMA + 1.0))
    p = a + (1.0 - a) * np.sqrt(1.0 - b * Q)
    u = (1.0 - p) / (GAMMA * ms) + ms
    rho = ms / u

    return u - np.sqrt(p / rho)

def deriv(x, y):
    q = Q * (1.0 - y[0] - y[1])
    if q > Q:
        raise Exception("x = %g" % x)
#    if (1.0 - B * q) < 0:
#        raise Exception("negative expr for sqrt, x = %g, q = %g" % (x, q))
    p = A + (1.0 - A) * np.sqrt(1.0 - B * q)
    u = (1.0 - p) / (GAMMA * MS) + MS
    rho = MS / u
    T = p / rho
    ri = y[0] * np.exp(ONE_OVER_EPSI * (1.0 / TI - 1.0 / T))
    rb = y[0] * y[1] * np.exp(ONE_OVER_EPSB * (1.0 / tb - 1.0 / T))
    rc = y[1]
    return np.array([-(ri + rb) / u, (ri + rb - rc) / u])


temperature = []

# Parameters.
GAMMA = 1.2
OVERDRIVE = 1.2
Q = 3.0
EPSI = 1.0 / 20.0
EPSB = 1.0 / 8.0
ONE_OVER_EPSI = 20.0
ONE_OVER_EPSB = 8.0
TI = 3.0
TBLIST = [0.8, 0.85, 0.9, 0.95]

#x = np.linspace(0.50, 0.545, 100000)
#f = func_mcj(x[:])
#print(f[0:5])
#print(f[-5:-1])
#plt.plot(x, f)
#plt.show()
#MCJ = sciopt.bisect(func_mcj, 0.78, 0.81)
#print("MCJ = ", MCJ)
MCJ = 1.445
Mambient_squared = OVERDRIVE * MCJ**2
MS_SQUARED = ((GAMMA - 1.0) * Mambient_squared + 2.0) / \
    (2.0 * GAMMA * Mambient_squared - (GAMMA - 1.0))
MS = np.sqrt(MS_SQUARED)
A = (GAMMA * MS_SQUARED + 1.0) / (GAMMA + 1.0)
B = (MS_SQUARED * 2.0 * GAMMA * (GAMMA - 1.0)) / \
    ((1.0 - A)**2 * (GAMMA + 1.0))

f0 = 1.0
y0 = 0.0

RB = 15.0
N = 100000
ics = np.array([f0, y0])
x = np.linspace(0, RB, N)
soln = []
for j in range(len(TBLIST)):
    tb = TBLIST[j]
    cursoln = np.zeros((N, 2))
    soln.append(cursoln)
    cursoln[0, :] = ics
    integrator = ode(deriv).set_integrator('vode', method='bdf', order=5)
    integrator.set_initial_value(ics, 0)
    i = 0
    while integrator.successful() and integrator.t < RB:
        i += 1
        integrator.integrate(x[i])
        cursoln[i, :] = integrator.y

plt.figure()
plt.plot(x, soln[0][:, 0])
plt.plot(x, soln[0][:, 1])
plt.plot(x, soln[1][:, 0])
plt.plot(x, soln[1][:, 1])
plt.plot(x, soln[2][:, 0])
plt.plot(x, soln[2][:, 1])
plt.plot(x, soln[3][:, 0])
plt.plot(x, soln[3][:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
