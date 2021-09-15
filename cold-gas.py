# %% - imports
import matplotlib.pyplot as plt
import numpy as np

from Engine import Engine

# %% - Initialization
pi = np.pi
g0 = 9.81       # gravitational acceleration [m/s^2]
Ra = 8.31447    # Gas constant

# %% - Baseline parameters

Pa = 0          # Ambient pressure [bar]
Dt = 0.26       # Throat diameter [m]

# Chamber parameters
pc = 206.4E5    # Chamber presssure [Pa]
Tc = 3500       # Chamber temperature [K]

# Nozzle parameters
De = 2.4        # Exit diameter [m]

# propellant properties
M_N2 = 28.013E-3  # Nitrogen molar mass [kg/mol]
M_H2 = 1E-3
R = Ra / M_H2
gamma_N2 = 1.40
gamma = 1.405

# %% - LR91-11
De = 1.6
Ae = pi / 4 * De**2

epsilon = 49.2
At = Ae / epsilon
Dt = np.sqrt(4 * At / pi)

LR91_11 = Engine(
    pc=5.93E6,
    Tc=3267,
    Dt=Dt,
    De=De,
    gamma=1.2176,
    **{'R': 375.3}
)

# %%

# thruster_1 = Engine(pc, Tc, Pa, Dt, De, gamma, R)
thruster_2 = Engine(
    # np.array([9E5, 10E5, 11E5, 12E5]),
    # np.array([298, 350, 400, 450]),
    pc,
    Tc,
    Pa,
    Dt,
    De,
    gamma,
    print_parameters=False,
    **{'R': R},
)
print('F_ideal', thruster_2.F_ideal)
print('Isp', thruster_2.Isp_ideal)
print('C*', thruster_2.C_star)
# thruster_3 = Engine(pc, 500, Pa, Dt, De, gamma, R)

epsilon = 50
At = Ae / epsilon
Dt = np.sqrt(4 / pi * At)

thruster_4 = Engine(pc, Tc, Pa, Dt, De, gamma, R)

# %% - Increase chamber pressure

# %% - Increase temperature

# %% - Decrease expansion ratio

# %% - Exhaust velocity

# pe = np.arange(1E5, 2E5, 0.1E5)
pe = 1E3
pc = np.arange(5E5, 10E5, 0.1E5)
ve = np.sqrt(2 * gamma / (gamma - 1) * Ra * Tc / M_N2 * (1 - (pe / pc)**((gamma - 1) / gamma)))

# print(ve)

plt.figure()
plt.plot(pe / pc, ve)
plt.show()
