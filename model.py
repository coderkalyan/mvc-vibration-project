import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import fftconvolve


DT = 0.001

class Toolhead:
    def __init__(self, frequency, damping_ratio, mass):
        self.frequency = frequency
        self.damping_ratio = damping_ratio
        self.mass = mass

        self.natural_period = 2 * np.pi * self.frequency
        self.period = self.natural_period * np.sqrt(1 - self.damping_ratio ** 2)
        self.stiffness = self.mass * self.natural_period ** 2
        self.damping_coefficient_crit = 2 * self.mass * self.natural_period
        self.damping_coefficient = self.damping_ratio * self.damping_coefficient_crit

        print(self.damping_coefficient_crit, self.damping_coefficient, self.stiffness)

    def oscillate(self, system, time, command):
        x, v = system
        kin_pos = np.interp(time, command[0], command[1])
        kin_vel = np.interp(time, command[0], command[2])
        kin_acc = np.interp(time, command[0], command[3])
        print(kin_pos, kin_vel, kin_acc)
        diff = [
            v, 
            (1 / self.mass) * (-self.damping_coefficient * (v - kin_vel) - self.stiffness * (x - kin_pos)) - kin_acc,
        ]

        return diff

    def shaper_defs(self, shaper_name):
        k = self.damping_ratio * np.pi / np.exp(np.sqrt(1 - self.damping_ratio ** 2))

        if shaper_name == "ZV":
            return [1 / (1 + k), k / (1 + k)], [0, np.pi / self.period]


def original_path(accel, coast_vel, edge_len, dt):
    accel_time = coast_vel / accel
    accel_pos = 0.5 * accel * accel_time ** 2
    target_edge_time = accel_time * 2 + (edge_len - 2 * accel_pos) / coast_vel
    times = [accel_time, target_edge_time - accel_time, target_edge_time]

    t = sp.symbols('t')
    accel_func = sp.Piecewise(
        (accel, t <= times[0]),             # acceleration
        (0, t <= times[1]),                 # coast
        (-accel, t <= target_edge_time),    # deceleration
        (0, True)                           # default case
    )
    vel_func = sp.integrate(accel_func, t)
    pos_func = sp.integrate(vel_func, t)

    edge_t = np.arange(0, target_edge_time, dt)
    edge_pos = sp.utilities.lambdify(t, pos_func, 'numpy')(edge_t)
    edge_vel = sp.utilities.lambdify(t, vel_func, 'numpy')(edge_t)

    x = np.hstack([edge_pos, np.full_like(edge_pos, edge_pos[-1]), edge_pos[::-1]])
    y = np.hstack([np.full_like(edge_pos, 0), edge_pos, np.full_like(edge_pos, edge_pos[-1])])
    # xvel = np.hstack([edge_vel, np.full_like(edge_vel, 0), edge_vel[::-1]])
    # yvel = np.hstack([np.full_like(edge_vel, 0), edge_vel, np.full_like(edge_vel, 0)])
    total_t = np.arange(len(x)) * dt

    return total_t, (x, y)#, (xvel, yvel), (xacc, yacc)


def shape_path(path, shaper_def, dt):
    x, y = path
    a, t = shaper_def
    kernel_times = np.arange(0, t[1], dt)
    kernel = np.zeros_like(kernel_times)
    kernel[0] = a[0]
    kernel[-1] = a[1]
    kernel = np.pad(kernel, (len(kernel), 0))
    
    x = fftconvolve(x, kernel, 'same')
    y = fftconvolve(y, kernel, 'same')

    return (x, y), kernel

# toolhead = Toolhead(frequency=35, damping_ratio=0.1, mass=0.433)
toolhead = Toolhead(frequency=35, damping_ratio=0.1, mass=0.433)
times, (x_path, y_path) = original_path(5.500, 0.150, 0.080, DT)

(x_comm, y_comm), kernel = shape_path((x_path, y_path), toolhead.shaper_defs("ZV"), DT)
kernel_init = [0, 0]

kernel_sol = odeint(toolhead.oscillate, kernel_init, times, args=(([0], [0], [0], [kernel]),))

plt.plot(np.arange(len(kernel)) * DT, kernel)
plt.plot(np.arange(len(kernel)) * DT, kernel_sol)
plt.show()

# x_comm = x_path
# y_comm = y_path

xpos = x_comm / DT
ypos = y_comm / DT
xvel = np.pad(np.diff(xpos), (0, 1))
yvel = np.pad(np.diff(ypos), (0, 1))
xacc = np.pad(np.diff(xvel), (0, 1))
yacc = np.pad(np.diff(yvel), (0, 1))

x_init = [x_comm[0], 0]
y_init = [y_comm[0], 0]
x_sol = odeint(toolhead.oscillate, x_init, times, args=((times, x_comm, xvel, xacc),))
y_sol = odeint(toolhead.oscillate, y_init, times, args=((times, y_comm, yvel, yacc),))
x_response, xvel_response = np.stack(x_sol, axis=1)
y_response, yvel_response = np.stack(y_sol, axis=1)

plt.plot(x_comm, y_comm)
plt.plot(x_response, y_response)
plt.show()
