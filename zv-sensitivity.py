import numpy as np
import matplotlib.pyplot as plt

OMEGA = 50
ZETA = 0.1
OMEGA_D = OMEGA * np.sqrt(1 - ZETA**2)

K = np.exp(ZETA * np.pi / np.sqrt(1 - ZETA**2))

omega_percent = np.linspace(0.8, 1.2, num=5)
omega_vals = omega_percent * OMEGA
zeta_percent = np.linspace(0.025, 6.0, num=5)
zeta_vals = zeta_percent * ZETA
omega_vals, zeta_vals = np.meshgrid(omega_vals, zeta_vals)
omega_d_vals = omega_vals * np.sqrt(1 - zeta_vals**2)

print(omega_vals)
print(zeta_vals)
# print(omega_d_vals)

a1 = K / (K + 1)
t1 = 0

k = np.exp(zeta_vals * np.pi / np.sqrt(1 - zeta_vals**2))
a2 = 1 / (k + 1)
t2 = np.pi / (omega_vals * np.sqrt(1 - zeta_vals**2))

# print(t2)

c1 = a1 * np.exp(OMEGA * ZETA * t1) * np.cos(OMEGA_D * t1)
s1 = a1 * np.exp(OMEGA * ZETA * t1) * np.sin(OMEGA_D * t1)
c2 = a1 * np.exp(omega_vals * zeta_vals * t2) * np.cos(omega_d_vals * t2)
s2 = a1 * np.exp(omega_vals * zeta_vals * t2) * np.sin(omega_d_vals * t2)

# print(c2)
# print(s2)

a_res = np.sqrt((c1 + c2) ** 2 + (s1 + s2) ** 2)

axes = plt.axes(projection='3d')
surf = axes.scatter(omega_vals, zeta_vals, a_res)
plt.show()
print(a_res)
