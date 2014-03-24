import numpy as np
from scipy.integrate import ode
from scipy import optimize as sciopt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time


def func_mcj(x):
    ms_squared = ((GAMMA - 1.0) * x**2 + 2.0) / \
        (2.0 * GAMMA * x**2 - (GAMMA - 1.0))
    ms = np.sqrt(ms_squared)

    a = (GAMMA * ms_squared + 1.0) / (GAMMA + 1.0)
    b = (ms_squared * 2.0 * GAMMA * (GAMMA - 1.0)) / \
        ((1.0 - a)**2 * (GAMMA + 1.0))
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
TBLIST = [0.8]
TBLIST = [0.8, 0.85, 0.9, 0.95]

f0 = 1.0
y0 = 0.0

RB = 15.0
N = 100000
ics = np.array([f0, y0])
x = np.linspace(0, RB, N)
soln = []
markerStyles = ['sk', 'ok', 'Dk', '^k']
markersForF = [6000, 11500, 23000, 50000]
markersForY = [10000, 17000, 33000, 69000]

#lboundary = 0.4
#rboundary = 0.5484858816226549
##0.54542578792120033
#x = np.linspace(lboundary, rboundary, 10000)
#f = func_mcj(x[:])
#print(f[0:5])
#print(f[-5:-1])
#plt.plot(x, f)
##plt.show()
##MCJ = sciopt.newton(func_mcj, 0.54)
##print("MCJ = ", MCJ)
MCJ = 2.2415942168486818
MAMBIENT_SQUARED = OVERDRIVE * MCJ**2
MS_SQUARED = ((GAMMA - 1.0) * MAMBIENT_SQUARED + 2.0) / \
    (2.0 * GAMMA * MAMBIENT_SQUARED - (GAMMA - 1.0))
MS = np.sqrt(MS_SQUARED)
A = (GAMMA * MS_SQUARED + 1.0) / (GAMMA + 1.0)
B = (MS_SQUARED * 2.0 * GAMMA * (GAMMA - 1.0)) / \
    ((1.0 - A)**2.0 * (GAMMA + 1.0))

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
for i in range(len(TBLIST)):
    plt.plot(x, soln[i][:, 0], '-k')
    plt.plot(x, soln[i][:, 1], '--k')
    plt.plot(x[markersForF[i]], soln[i][markersForF[i], 0], markerStyles[i])
    plt.plot(x[markersForY[i]], soln[i][markersForY[i], 1], markerStyles[i])
plt.xlabel(r'$x$')
plt.ylabel(r'$f_1,f_2$')
plt.xlim([0, RB])
plt.ylim([0.0, 1.0])
plt.xticks(range(0, 16, 5))
ax = plt.gca()
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
#plt.show()
timestr = time.strftime("-%Y-%m-%dT%H%M")
plt.savefig('images/steady_structure' + timestr + '.eps')
