from sympy import *
init_printing()

a1, a2, zeta, omega, t1, t2, omegad = symbols("A_1 A_2 zeta omega t_1 t_2 omega_d")
u1 = a1 * exp(-zeta * omega * t1) * cos(omegad * t1)
u2 = a2 * exp(-zeta * omega * t2) * cos(omegad * t2)
v1 = a1 * exp(-zeta * omega * t1) * sin(omegad * t1)
v2 = a2 * exp(-zeta * omega * t2) * sin(omegad * t2)
ares = sqrt((u1 + u2) ** 2 + (v1 + v2) ** 2)
print(ares)
