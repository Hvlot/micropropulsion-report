import numpy as np
from scipy.optimize import fsolve


def vandenkerckhove(gamma: float) -> float:
    return np.sqrt(gamma) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))


# @njit(cache=True)
def area(D: float) -> float:
    """ Calculate the area [m^2]

    Arguments:
        D {float} -- Diameter [m]

    Parameters
    ----------
    D : float
        Diameter [m]
    """
    return np.pi / 4 * D**2


# @njit(cache=True)
def exhaust_velocity(gamma: float, R: float, Tc: float, pe: float, pc: float) -> float:
    ve_lim = np.sqrt(2 * gamma / (gamma - 1) * R * Tc)
    return ve_lim * np.sqrt(1 - (pe / pc)**((gamma - 1) / gamma))


# @njit(cache=True)
def mass_flow_rate(Gamma: float, pc: float, At: float, R: float, Tc: float) -> float:
    """Returns the mass flow rate [kg/s]"""
    return Gamma * pc * At / np.sqrt(R * Tc)


def func(x, gamma, Gamma, epsilon, pc, d, e):
    """Function that relates the expansion ratio and the pressure ratio.

    x: exhaust pressure
    d: 2/gamma
    e: (gamma-1)/gamma
    """
    return epsilon - Gamma / np.sqrt(2 / e * (x / pc)**d * (1 - (x / pc)**e))


def exhaust_pressure(gamma, Gamma, epsilon, pc, d, e, x0=50):
    pe, infodict, ier, _ = fsolve(func, x0=x0, args=(gamma, Gamma, epsilon, pc, d, e), xtol=0.01, full_output=True)

    return pe[0], ier, infodict['nfev']


# @njit(cache=True)
def characteristic_velocity(pc, At, m):
    """Calculate the characteristic exhaust velocity [m/s]"""
    return pc * At / m


# @njit(cache=True)
def power(m, Tc, cp_liquid, Lh=2256E3, Ta=298, efficiency=0.6):
    return m / efficiency * (cp_liquid * (Tc - Ta) + Lh)


# @njit(cache=True)
def chamber_temperature(R, pc, T_ref=373.15, Lh=2256E3, p_ref=1.0142E5):
    return T_ref * Lh / (T_ref * R * np.log(p_ref / pc) + Lh)


def calc_Cd(m_ideal, At, a, b, c):
    w_t = 25.36E-6          # Throat width [m]
    mu_t = 1.1E-5           # Dynamic viscosity [Pa*s]
    Re_t = m_ideal * w_t / (mu_t * At)

    Re_mod = Re_t * np.sqrt(2)

    Cd = 1 - a * b * 1 / np.sqrt(Re_mod) + c * 1 / Re_mod
    return Cd


def initialize_envelope(p, V, te, dt):
    te = 1300
    dt = 0.1

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

    burn_time = np.zeros((len(p), len(V)))
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
        "P_t": P_t
    }

    return arrays
