# %% - imports & constants
import numpy as np

import utils
from utils import (calc_Cd, calc_dF, chamber_temperature, exhaust_pressure,
                   mass_flow_rate)

g0 = 9.81       # gravitational acceleration [m/s^2]
Ra = 8.31447    # Gas constant

# %% - Class definition


class Propellant:
    def __init__(self, M: float, gamma: float, rho: float):
        self.gamma = gamma
        self.Gamma = utils.vandenkerckhove(gamma)
        self.R = Ra / M
        self.rho = rho

        # Factors from the Tang and Fenn relation to calculate discharge coefficient.
        self.a = ((gamma + 1) / 2)**(3 / 4)
        self.b = ((72 - 32 * np.sqrt(6)) / (3 * (gamma + 1)) + 4 * np.sqrt(6) / 3)
        self.c = (2 * np.sqrt(2) * (gamma - 1) * (gamma + 2) / (3 * np.sqrt(gamma + 1)))

        self.d = (2 / gamma)
        self.e = (gamma - 1) / gamma


class Nozzle:
    def __init__(self, De: float, Dt: float):
        self.De = De
        self.Dt = Dt

        self.At = utils.area(Dt)
        self.Ae = utils.area(De)
        self.L = 0.6856E-3       # Nozzle wall length [m]

        self.epsilon = self.Ae / self.At    # Expansion ratio [-]


class Engine:
    def __init__(self, Tc, Dt, De, gamma, rho, M):
        self.Tc = Tc

        self.nozzle = Nozzle(De, Dt)
        self.propellant = Propellant(M, gamma, rho)

    def isp_ideal(self) -> float:
        """Calculates the ideal specific impulse [s]

        Input parameters:
        - C_star (Characteristic exhaust velocity [m/s])
        - gamma (ratio of specific heats [-])
        """
        gamma = self.propellant.gamma
        return self.C_star / g0 * gamma * np.sqrt(2 / (gamma - 1) * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))

    # def ideal_thrust(self):
    #     """Calculate the ideal thrust [N]"""
    #     return self.Isp_ideal * self.m * g0

    def envelope(self, m_initial, V0, p0, len_t, dt=0.1):
        gamma = self.propellant.gamma
        pc_ref = 5E5
        F_ref = 3.48E-3
        CF_ref = F_ref / (pc_ref * self.nozzle.At)
        Isp_ref = 94.9 / (CF_ref * np.sqrt(550))

        Ae = 0.8E-3 * 0.18E-3
        epsilon = Ae / self.nozzle.At

        m_exit = np.zeros(len_t)
        m = np.zeros(len_t)
        p_t = np.zeros(len_t)
        pe = np.zeros(len_t)
        V_t = np.zeros(len_t)
        F_t = np.zeros(len_t)
        P_t = np.zeros(len_t)
        Tc = np.zeros(len_t)
        thrust_to_power = np.zeros(len_t)

        # Initial values

        m[0] = mass_flow_rate(self.propellant.Gamma, p0, self.nozzle.At, self.propellant.R, self.Tc)
        p_t[0] = p0
        Tc[0] = self.Tc

        # Initialize iteration
        p_t[1] = V0 * p0 / (V0 + m_exit[0] / self.propellant.rho)
        Tc[1] = chamber_temperature(self.propellant.R, p0)

        # Ideal mass flow rate
        m_ideal = mass_flow_rate(self.propellant.Gamma, p_t[0], self.nozzle.At, self.propellant.R, Tc[0])

        Cd = calc_Cd(m_ideal, self.nozzle.At, self.propellant.a, self.propellant.b, self.propellant.c)

        m[1] = Cd * m_ideal
        m_exit[1] = m_exit[0] + m[1] * dt

        V_t[1] = V0 * p_t[1] / p_t[1]

        pe[1], ier, nevals = exhaust_pressure(gamma, self.propellant.Gamma, epsilon, p_t[1], self.propellant.d, self.propellant.e)

        CF = self.propellant.Gamma * np.sqrt(2 * gamma / (gamma - 1) * (1 - (pe[1] / p_t[1]) ** ((gamma - 1) / gamma))) + pe[1] / p_t[1] * epsilon
        F_t[1] = Isp_ref * CF * np.sqrt(Tc[1]) * 9.81 * m[1]
        m_prop_left = m_initial - m_exit[1]

        i = 1

        while F_t[i] > 0.12E-3 and m_prop_left > 0.2E-3:
            i += 1
            p_t[i] = V0 * p0 / (V0 + m_exit[i - 1] / self.propellant.rho)

            Tc[i] = chamber_temperature(self.propellant.R, p_t[i - 1])

            # mass flow rate parameters
            m_ideal = mass_flow_rate(self.propellant.Gamma, p_t[i - 1], self.nozzle.At, self.propellant.R, Tc[i])
            Cd = calc_Cd(m_ideal, self.nozzle.At, self.propellant.a, self.propellant.b, self.propellant.c)
            m[i] = Cd * m_ideal

            m_exit[i] = m_exit[i - 1] + m[i] * dt
            m_prop_left = m_initial - m_exit[i]

            # Thrust parameters
            pe[i], ier, nevals = exhaust_pressure(gamma, self.propellant.Gamma, epsilon, p_t[i], self.propellant.d, self.propellant.e, pe[i - 1])

            # eq. 8-3 from TRP reader [Zandbergen]
            CF = self.propellant.Gamma * np.sqrt(2 * gamma / (gamma - 1) *
                                                 (1 - (pe[i] / p_t[i])**((gamma - 1) / gamma))) + pe[i] / p_t[i] * epsilon
            Isp = Isp_ref * CF * np.sqrt(Tc[i])
            dF = calc_dF(self.propellant.gamma, self.propellant.R, self.Tc, pe[i], p_t[i], self.nozzle.L)
            F_t[i] = Isp * 9.81 * m[i] - dF

            P_t[i] = utils.power(m[i], Tc[i], 4187)

            thrust_to_power[i] = F_t[i] / P_t[i]

        if F_t[i] < 0.12E-3:
            while_condition = 1
        if m_prop_left < 0.2E-3:
            while_condition = 2
        burn_time = i * dt

        return burn_time, while_condition, F_t, P_t
