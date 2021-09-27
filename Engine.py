# %% - imports & constants
import numpy as np

import utils
from utils import (calc_Cd, chamber_temperature, exhaust_pressure,
                   exhaust_velocity, mass_flow_rate)

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
    def __init__(self, Tc, Dt, De, gamma, rho, M, p, V, print_parameters=True):
        # self.pc = pc
        self.Tc = Tc

        # self.initialize_envelope(p, V)

        self.nozzle = Nozzle(De, Dt)
        self.propellant = Propellant(M, gamma, rho)

        # chamber_temperature(500, 2E5)
        # Ideal mass flow rate
        # self.m = mass_flow_rate(self.propellant.Gamma, pc, self.nozzle.At, self.propellant.R, self.Tc)

        # self.C_star = utils.characteristic_velocity(pc, self.nozzle.At, self.m)
        # self.Isp_ideal = self.isp_ideal()
        # self.F_ideal = self.ideal_thrust()
        # self.pe, _ = utils.exhaust_pressure(gamma, self.nozzle.epsilon, pc)
        # self.ve = utils.exhaust_velocity(gamma, self.propellant.R, Tc, self.pe, pc)
        # self.F_opt = self.m * self.ve + self.pe * self.nozzle.Ae
        # self.F_SL = self.m * self.ve + (self.pe - 1E5) * self.nozzle.Ae
        # self.Isp_opt = self.F_opt / (self.m * g0)
        # self.Isp_SL = self.F_SL / (self.m * g0)

        if print_parameters:
            print()
            print('%----%')
            print()
            # self.rprint('Ideal Isp', self.Isp_ideal, 2, 's')
            # if kwargs.get('thrust_unit', None) == 1E3:
            #     self.rprint('Ideal Thrust', self.F_ideal * 1E3, 3, 'mN')
            # else:
            #     self.rprint('Ideal Thrust', self.F_ideal, 3, 'N')
            # self.rprint('C*', self.C_star, 1, 'm/s')

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

    def rprint(self, parameter: str, parameter_val: float, decimals: int, unit: str):
        print(f'{parameter}: {round(parameter_val, decimals)} [{unit}]')

    def envelope(self, m_initial, V0, p0, len_t, dt=0.1):
        gamma = self.propellant.gamma
        pc_ref = 5E5
        F_ref = 3.48E-3
        CF_ref = F_ref / (pc_ref * self.nozzle.At)
        Isp_ref = 94.9 / (CF_ref * np.sqrt(550))
        # Cf_ref = self.propellant.Gamma*np.sqrt(2*gamma/(gamma-1)*(1-pe/pc_ref)**((gamma-1)/gamma))+pe/pc_ref*epsilon

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

        m_ideal = mass_flow_rate(self.propellant.Gamma, p_t[0], self.nozzle.At, self.propellant.R, Tc[0])

        Cd = calc_Cd(m_ideal, self.nozzle.At, self.propellant.a, self.propellant.b, self.propellant.c)

        # m[1] = mass_flow_rate(self.propellant.Gamma, p_t[0], self.nozzle.At, self.propellant.R, Tc[0])
        m[1] = Cd * m_ideal

        m_exit[1] = m_exit[0] + m[1] * dt

        V_t[1] = V0 * p_t[1] / p_t[1]
        # M = p_t[1] * V_t[1] / (vlm.propellant.R * vlm.Tc)

        nevals_total = 0

        pe[1], ier, nevals = exhaust_pressure(gamma, self.propellant.Gamma, epsilon, p_t[1], self.propellant.d, self.propellant.e)

        nevals_total += nevals

        if ier == 1:
            CF = self.propellant.Gamma * np.sqrt(2 * gamma / (gamma - 1) * (1 - (pe[1] / p_t[1]) ** ((gamma - 1) / gamma))) + pe[1] / p_t[1] * epsilon
        else:
            CF = CF_ref
        # CF = CF_ref
        F_t[1] = Isp_ref * CF * np.sqrt(Tc[1]) * 9.81 * m[1]

        m_prop_left = m_initial - m_exit[1]

        i = 1

        while F_t[i] > 0.12E-3 and m_prop_left > 0.2E-3:
            i += 1
            p_t[i] = V0 * p0 / (V0 + m_exit[i - 1] / self.propellant.rho)

            Tc[i] = chamber_temperature(self.propellant.R, p_t[i - 1])
            # m[i, j, k] = p_t[i - 1] * self.nozzle.At * self.propellant.Gamma / np.sqrt(self.propellant.R * Tc[i, j, k])

            m_ideal = mass_flow_rate(self.propellant.Gamma, p_t[i - 1], self.nozzle.At, self.propellant.R, Tc[i])

            # V_t[i, j, k] = V0 * p_t[1] / p_t[i, j, k]
            # M = p_t[i, j, k] * V_t[i, j, k] / (vlm.propellant.R * vlm.Tc)

            pe[i], ier, nevals = exhaust_pressure(gamma, self.propellant.Gamma, epsilon, p_t[i], self.propellant.d, self.propellant.e, pe[i - 1])
            # if i % 10 == 0:
            # pe[i], ier, nevals = exhaust_pressure(gamma, self.propellant.Gamma, epsilon, p_t[i], self.propellant.d, self.propellant.e, pe[i - 10])
            # nevals_total += nevals

            # eq. 8-3 from TRP reader [Zandbergen]
            CF = self.propellant.Gamma * np.sqrt(2 * gamma / (gamma - 1) *
                                                 (1 - (pe[i] / p_t[i])**((gamma - 1) / gamma))) + pe[i] / p_t[i] * epsilon

            Ue = exhaust_velocity(self.propellant.gamma, self.propellant.R, self.Tc, pe[i], p_t[i])

            rho_c = 997         # Chamber density [kg/m^3]
            R_e = 0.4E-3        # Nozzle radius [m]

            Te = self.Tc * (pe[i] / p_t[i])**((gamma - 1) / gamma)
            # print('Te', Te, 'pe', pe[i])

            # viscosity of the exhaust gas only dependent on exit temperature, which is constant.
            mu_e = 3.06E-6      # Dynamic viscosity [Pa*s]
            # Using the ideal gas law
            rho_e = pe[i] / (self.propellant.R * Te)
            Re_e = rho_e * Ue * self.nozzle.L / mu_e
            # print('Re', Re_e)
            theta = 0.664 / np.sqrt(Re_e) * self.nozzle.L

            dF = rho_e * Ue * 2 * np.pi * R_e * theta * Ue

            Cd = calc_Cd(m_ideal, self.nozzle.At, self.propellant.a, self.propellant.b, self.propellant.c)
            # Cd = 1
            m[i] = Cd * m_ideal

            m_exit[i] = m_exit[i - 1] + m[i] * dt
            m_prop_left = m_initial - m_exit[i]

            Isp = Isp_ref * CF * np.sqrt(Tc[i])
            F_t[i] = Isp * 9.81 * m[i] - dF
            # print('F', F_t[i], 'dF', dF, 'F-dF', F_t[i] - dF)

            P_t[i] = utils.power(m[i], Tc[i], 4187)

            thrust_to_power[i] = F_t[i] / P_t[i]

        if F_t[i] < 0.12E-3:
            while_condition = 1
        if m_prop_left < 0.2E-3:
            while_condition = 2
        burn_time = i * dt

        # if (nevals_avg := nevals_total / i) > 6:
        # print('average number of function calls', nevals_total / i)
        # print('burn time', burn_time)

        # print('min', np.min(pe[pe != 0]), 'mean', np.mean(pe[pe != 0]), 'max', np.max(pe[pe != 0]))

        return burn_time, while_condition, F_t, P_t
        # return (j, k, i * self.dt)
        # print(f'Process {j, k}')
        # return m_exit, m, p_t, pe, V_t, F_t, P_t, Tc, thrust_to_power, burn_time
