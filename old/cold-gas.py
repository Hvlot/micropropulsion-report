# %% - Functions

import numpy as np


def rprint(parameter: str, parameter_val: float, decimals: int, unit: str):
    print(f'{parameter}: {round(parameter_val, decimals)} [{unit}]')


def calc_isp_ideal(C_star: float, gamma: float) -> float:
    """Calculates the ideal specific impulse [s]

    Input parameters:
    - C_star (Characteristic exhaust velocity [m/s])
    - gamma (ratio of specific heats [-])
    """
    return C_star / g0 * gamma * np.sqrt(2 / (gamma - 1) * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))


def vandenkerckhove(gamma: float) -> float:
    return np.sqrt(gamma) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))

# %% - Initialization


pi = np.pi
g0 = 9.81       # gravitational acceleration [m/s^2]
Ra = 8.31447    # Gas constant

# %% - Baseline parameters

Pa = 0  # Ambient pressure [bar]

# Chamber parameters
pc = 5E5        # Chamber presssure [bar]
Tc = 298        # Chamber temperature [K]

# Nozzle parameters
Dt = 1E-3            # Throat diameter [m]
De = 10E-3           # Exit diameter [m]
At = pi / 4 * Dt**2  # Throat area [m^2]
Ae = pi / 4 * De**2  # Exit area [m^2]
epsilon = Ae / At    # Expansion rattio [-]

# propellant properties
M_N2 = 28.013E-3  # Nitrogen molar mass [kg/mol]
R = Ra / M_N2
gamma = 1.40
Gamma = vandenkerckhove(gamma)

m = Gamma * pc * At / np.sqrt(R * Tc)   # Mass flow rate [kg/s]
C_star = pc * At / m                    # Characteristic exhaust velocity [m/s]
Isp_ideal = calc_isp_ideal(C_star, gamma)
F_thrust_ideal = Isp_ideal * m * g0

rprint('Ideal Isp', Isp_ideal, 2, 's')
rprint('Ideal Thrust', F_thrust_ideal, 3, 'N')
rprint('C*', C_star, 1, 'm/s')

# %% - Increase chamber pressure

pc = 10E5

m = Gamma * pc * At / np.sqrt(R * Tc)   # Mass flow rate [kg/s]
C_star = pc * At / m                    # Characteristic exhaust velocity [m/s]
Isp_ideal = calc_isp_ideal(C_star, gamma)
F_thrust_ideal = Isp_ideal * m * g0

rprint('Ideal Isp', Isp_ideal, 2, 's')
rprint('Ideal Thrust', F_thrust_ideal, 2, 'N')
rprint('C*', C_star, 1, 'm/s')

# %% - Increase temperature

pc = 5E5        # Chamber presssure [bar]
Tc = 500        # Chamber temperature [K]

m = Gamma * pc * At / np.sqrt(R * Tc)   # Mass flow rate [kg/s]
C_star = pc * At / m                    # Characteristic exhaust velocity [m/s]
Isp_ideal = calc_isp_ideal(C_star, gamma)
F_thrust_ideal = Isp_ideal * m * g0

rprint('Ideal Isp', Isp_ideal, 2, 's')
rprint('Ideal Thrust', F_thrust_ideal, 3, 'N')
rprint('C*', C_star, 1, 'm/s')

# %% - Decrease expansion ratio

Tc = 298        # Chamber temperature [K]
epsilon = 50
At = Ae / epsilon

m = Gamma * pc * At / np.sqrt(R * Tc)   # Mass flow rate [kg/s]
C_star = pc * At / m                    # Characteristic exhaust velocity [m/s]
Isp_ideal = calc_isp_ideal(C_star, gamma)
F_thrust_ideal = Isp_ideal * m * g0

rprint('Ideal Isp', Isp_ideal, 2, 's')
rprint('Ideal Thrust', F_thrust_ideal, 3, 'N')
rprint('C*', C_star, 1, 'm/s')
