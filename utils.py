import numpy as np
from numba import njit
from scipy.optimize import fsolve

# @njit(cache=True)


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


def func(x, gamma, epsilon, pc):
    Gamma = vandenkerckhove(gamma)
    return epsilon - Gamma / np.sqrt(2 * gamma / (gamma - 1) * (x / pc)**(2 / gamma)
                                     * (1 - (x / pc)**((gamma - 1) / gamma)))


def exhaust_pressure(gamma, epsilon, pc, x0=100):
    pe, _, ier, _ = fsolve(func, x0=x0, args=(gamma, epsilon, pc), full_output=True)
    return pe[0], ier


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


def calc_Cd(m_ideal, At, gamma):
    w_t = 25.36E-6          # Throat width [m]
    mu_t = 1.1E-5           # Dynamic viscosity [Pa*s]
    Re_t = m_ideal * w_t / (mu_t * At)

    Re_mod = Re_t * np.sqrt(2)

    Cd = 1 - ((gamma + 1) / 2)**(3 / 4) * ((72 - 32 * np.sqrt(6)) / (3 * (gamma + 1)) + 4 * np.sqrt(6) / 3) * \
        1 / np.sqrt(Re_mod) + (2 * np.sqrt(2) * (gamma - 1) * (gamma + 2) / (3 * np.sqrt(gamma + 1))) * 1 / Re_mod
    return Cd
