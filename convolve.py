import numpy as np
import sympy as sm
import sympy.abc as abc
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy


A = 200 #2500
V = 150
S = 80
T = S / V
DT = 0.0001
ZETA = 0.05
FREQ = 35
OMEGA = 2 * np.pi * FREQ
OMEGA_D = OMEGA * np.sqrt(1 - ZETA**2)
A1 = 5.5 
MASS = 0.433
K = OMEGA**2 * MASS
CC = 2 * np.sqrt(K * MASS)
C = ZETA * CC

TIMES = np.arange(0, T, DT)
omega_d = sm.symbols('omega_d')
OSCFUNC = abc.A * sm.exp(-abc.zeta * abc.omega * abc.t) * sm.sin(omega_d * abc.t)

def orig_path():
    accel_time = V / A
    coast_time = T - 2 * accel_time


    accfunc = sm.Piecewise((A, abc.t <= accel_time), (0, abc.t < accel_time + coast_time), (-A, abc.t <= T), (0, True))
    velfunc = sm.integrate(accfunc, abc.t)
    posfunc = sm.integrate(velfunc, abc.t)

    times3 = (TIMES[1] - TIMES[0]) * np.arange(3 * len(TIMES))
    pos = sm.utilities.lambdify(abc.t, posfunc, 'numpy')(TIMES)

    return times3, pos

def zv_shaper():
    C = abc.zeta * sm.pi / sm.exp(sm.sqrt(1 - abc.zeta**2))
    omega_d = abc.omega * sm.sqrt(1 - abc.zeta**2)
    denominator = 1 + C

    return [sm.Float(1) / denominator, C / denominator], [sm.Float(0), sm.pi / omega_d]

def oscillate(y, t, times, command):
    x, v = y
    dydt = [v, (1 / MASS) * (-C * v - K * (x - np.interp(t, times, command)))]
    return dydt

osc = OSCFUNC.subs({abc.A: A1, abc.zeta: ZETA, abc.omega: OMEGA, omega_d: OMEGA_D})
osc = sm.utilities.lambdify(abc.t, osc, 'numpy')(TIMES)

times3, pos = orig_path()

a, t = zv_shaper()
a = np.array([sm.N(x.subs({abc.zeta: ZETA, abc.omega: OMEGA})) for x in a])
t = np.array([sm.N(x.subs({abc.zeta: ZETA, abc.omega: OMEGA})) for x in t])

ts = np.sum(a * t)
print(t, ts)
# t -= ts
print(t)
kernel_times = np.arange(0, t[1] * 2, DT)
kernel = np.full_like(kernel_times, 0)
kernel[len(kernel) // 2] = a[0]
kernel[-1] = a[1]

print(len(kernel))
print(kernel)
x_orig = np.hstack([pos, np.full_like(pos, pos[-1]), pos[::-1]])
y_orig = np.hstack([np.full_like(pos, 0), pos, np.full_like(pos, pos[-1])])
# x_orig = np.arange(0, 3*T + 2*DT, DT) + 5
# y_orig = np.zeros_like(x_orig)#np.arange(0, 3*T, DT)
# x_orig = np.hstack([pos, np.full_like(pos, pos[-1]) + osc, pos[::-1]])
# y_orig = np.hstack([np.full_like(pos, 0), pos, np.full_like(pos, pos[-1]) + osc])


# plt.plot(times3, sol)
# plt.show()
#
# exit(0)
# x = np.convolve(x, kernel, 'same')
# y = np.convolve(y, kernel, 'same')
x_comm = scipy.signal.fftconvolve(x_orig, kernel, 'same')
y_comm = scipy.signal.fftconvolve(y_orig, kernel, 'same')
# x_comm = x_orig
# y_comm = y_orig

y0 = [0, 0]
# xsol = odeint(oscillate, y0, times3, args=(times3, x_orig))
x, xvel = zip(*odeint(oscillate, y0, times3, args=(times3, x_comm)))
y, yvel = zip(*odeint(oscillate, y0, times3, args=(times3, y_comm)))


xorig, xvel = zip(*odeint(oscillate, y0, times3, args=(times3, x_orig)))
yorig, yvel = zip(*odeint(oscillate, y0, times3, args=(times3, y_orig)))

plt.plot(x_comm, y_comm)
plt.plot(xorig, yorig)
plt.plot(x, y)
# plt.plot(times3, x_orig)
# plt.plot(times3, x)
plt.show()
