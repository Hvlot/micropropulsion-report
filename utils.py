import numpy as np
from scipy.optimize import fsolve


def vandenkerckhove(gamma: float) -> float:
    return np.sqrt(gamma) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))


def area(D: float) -> float:
    """ Calculate the area [m^2]

    Parameters
    ----------
    D: Diameter [m]
    """
    return np.pi / 4 * D**2


def exhaust_velocity(gamma: float, R: float, Tc: float, pe: float, pc: float) -> float:
    """Calculate the exhaust velocity [m/s]"""
    ve_lim = np.sqrt(2 * gamma / (gamma - 1) * R * Tc)
    return ve_lim * np.sqrt(1 - (pe / pc)**((gamma - 1) / gamma))


def mass_flow_rate(Gamma: float, pc: float, At: float, R: float, Tc: float) -> float:
    """Calculate the mass flow rate [kg/s]

    - Gamma: vandenkerkhove parameter [-]
    - pc: chamber pressure [pa]
    - At: throat area [m^2]
    - R: specific gas constant [J/(kg*K)]
    - Tc: chamber temperature [K]
    """
    return Gamma * pc * At / np.sqrt(R * Tc)


def func(x, gamma, Gamma, epsilon, pc, d, e):
    """Function that relates the expansion ratio and the pressure ratio.

    Equation 7-3 [Zandbergen, TRP reader, p. 52]
    - x: exhaust pressure [pa]
    - pc: chamber pressure [pa]
    - epsilon: Area ratio [-]
    - d: 2/gamma [-]
    - e: (gamma-1)/gamma [-]
    """
    return epsilon - Gamma / np.sqrt(2 / e * (x / pc)**d * (1 - (x / pc)**e))


def exhaust_pressure(gamma, Gamma, epsilon, pc, d, e, x0=50):
    """Calculate the exhaust pressure [pa] with fsolve.

    - gamma: specific heat ratio [-]
    - d: 2/gamma [-]
    - e: (gamma-1)/gamma [-]
    """
    pe, infodict, ier, _ = fsolve(func, x0=x0, args=(gamma, Gamma, epsilon, pc, d, e), xtol=0.01, full_output=True)

    return pe[0], ier, infodict['nfev']


def characteristic_velocity(pc, At, m):
    """Calculate the characteristic exhaust velocity [m/s]"""
    return pc * At / m


def power(m, Tc, cp_liquid, Lh=2256E3, Ta=283, efficiency=0.6):
    return m / efficiency * (cp_liquid * (Tc - Ta) + Lh)


def chamber_temperature(R: float, pc: float, T_ref: float = 373.15, Lh: float = 2256E3, p_ref: float = 1.0142E5) -> float:
    """Calculate the chamber temperature [K].

    - R: specific gas constant [J/(kg*K)]
    - pc: chamber temperature [pa]
    """
    return T_ref * Lh / (T_ref * R * np.log(p_ref / pc) + Lh)


def calc_Cd(m_ideal, At, a, b, c):
    """Calculate the discharge coefficient [-].

    - a: ((gamma + 1) / 2)**(3 / 4)
    - b: ((72 - 32 * np.sqrt(6)) / (3 * (gamma + 1)) + 4 * np.sqrt(6) / 3)
    - c: (2 * np.sqrt(2) * (gamma - 1) * (gamma + 2) / (3 * np.sqrt(gamma + 1)))
    """
    w_t = 25.36E-6          # Throat width [m]
    mu_t = 1.1E-5           # Dynamic viscosity [Pa*s] (local or stagnation conditions?)
    Re_t = m_ideal * w_t / (mu_t * At)
    # print('Re', Re_t)
    Re_mod = Re_t * np.sqrt(2)

    Cd = 1 - a * b * 1 / np.sqrt(Re_mod) + c * 1 / Re_mod
    # print('Cd', Cd, 'Re', Re_t)
    return Cd, Re_t


def initialize_envelope(p, V, te, dt=0.1):

    l_tube = 0.3
    d_tube = 1.57E-3
    V_tube = l_tube * np.pi * d_tube**2 / 4

    t = np.arange(0, te, dt)

    m_exit = np.zeros((len(t), len(p), len(V)))
    m = np.zeros((len(t), len(p), len(V)))
    p_t = np.zeros((len(t), len(p), len(V)))
    pe = np.zeros((len(t), len(p), len(V)))
    V_t = np.zeros((len(t), len(p), len(V)))
    F_t = np.zeros((len(t), len(p), len(V)))
    P_t = np.zeros((len(t), len(p), len(V)))
    Tc_vec = np.zeros((len(t), len(p), len(V)))
    thrust_to_power = np.zeros((len(t), len(p), len(V)))
    Cd = np.zeros((len(t), len(p), len(V)))

    burn_time = np.zeros((len(p), len(V)))
    m_prop_left = np.zeros((len(p), len(V)))
    F_t_range = np.zeros((len(p), len(V)))
    while_condition = np.zeros((len(p), len(V)))
    m_initial = np.zeros((len(p), len(V)))
    m_initial_total = np.zeros((len(p), len(V)))

    arrays = {
        "time": t,
        "dt": dt,
        "burn_time": burn_time,
        "while_condition": while_condition,
        "thrust_to_power": thrust_to_power,
        "F_t": F_t,
        "P_t": P_t,
        "p_t": p_t,
        "m_prop_left": m_prop_left,
        "F_t_range": F_t_range,
        "Tc": Tc_vec,
        "mass_flow_rate": m,
        "thrust_coefficient": Cd
    }

    return arrays


def calc_dp(m, rho):
    """Calculate the pressure drop over the feed lines. """
    d_fl = 1.57E-3
    # d_fl = 4.5E-6
    A = np.pi / 4 * d_fl**2
    # A = d_fl * 0.1E-3
    v = m / (A * rho)
    mu = 1.1E-5
    Re = rho * v * d_fl / mu

    f = 64 / Re
    L = 0.3

    N_bends = 2

    dp = f * L / d_fl * 0.5 * rho * v**2 + 1.3 * N_bends * 0.5 * v**2

    return dp


def calc_dF(gamma, R, Tc, pe, pc, L, m, Ae):
    Ue = exhaust_velocity(gamma, R, Tc, pe, pc)
    # print('Ue', Ue)

    Te = Tc * (pe / pc)**((gamma - 1) / gamma)

    # Using the ideal gas law
    rho_e = pe / (R * Te)                           # Gas density at the nozzle exit [kg/m^3]

    # viscosity of the exhaust gas only dependent on exit temperature, which is constant.
    mu_e = 3.06E-6                                  # Dynamic viscosity [Pa*s]

    # As taken from example 7.2.5 in the TRP reader [Zandbergen]
    Re_e = rho_e * Ue * L / mu_e                    # Nozzle exit Reynolds number [-]
    Re2 = m * L / (mu_e * Ae)
    theta = 0.664 / np.sqrt(Re2) * L                # momentum loss thickness [m]
    R_e = 0.4E-3                                    # Nozzle radius [m]

    dF = rho_e * Ue * 2 * np.pi * R_e * theta * Ue  # momentum loss [N]

    # print('theta', theta, 'Re', Re_e, 'Re2', Re2)

    return dF

# %%
