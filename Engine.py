# %% - imports & constants
import numpy as np

import utils
from utils import (calc_Cd, chamber_temperature, exhaust_pressure,
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


class Nozzle:
    def __init__(self, De: float, Dt: float):
        self.De = De
        self.Dt = Dt

        self.At = utils.area(Dt)
        self.Ae = utils.area(De)
        self.epsilon = self.Ae / self.At    # Expansion ratio [-]


class Engine:
    def __init__(self, Tc, Dt, De, gamma, rho, M, p, V, print_parameters=True):
        # self.pc = pc
        self.Tc = Tc

        self.initialize_envelope(p, V)

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

    def initialize_envelope(self, p, V):
        self.te = 1300
        self.dt = 0.5

        l_tube = 0.3
        d_tube = 1.57E-3
        self.V_tube = l_tube * np.pi * d_tube**2 / 4

        self.t = np.arange(0, self.te, self.dt)

        self.m_exit = np.zeros((len(self.t), len(p), len(V)))
        self.m = np.zeros((len(self.t), len(p), len(V)))
        self.p_t = np.zeros((len(self.t), len(p), len(V)))
        self.pe = np.zeros((len(self.t), len(p), len(V)))
        self.V_t = np.zeros((len(self.t), len(p), len(V)))
        self.F_t = np.zeros((len(self.t), len(p), len(V)))
        self.P_t = np.zeros((len(self.t), len(p), len(V)))
        self.Tc_vec = np.zeros((len(self.t), len(p), len(V)))
        self.thrust_to_power = np.zeros((len(self.t), len(p), len(V)))

        self.burn_time = np.zeros((len(p), len(V)))
        self.m_initial = np.zeros((len(p), len(V)))
        self.m_initial_total = np.zeros((len(p), len(V)))

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

    def envelope(self, m_initial, V0, p0, m_input, Tc_input, pe_input, j, k):
        gamma = self.propellant.gamma
        pc_ref = 5E5
        F_ref = 3.48E-3
        CF_ref = F_ref / (pc_ref * self.nozzle.At)
        Isp_ref = 94.9 / (CF_ref * np.sqrt(550))
        # Cf_ref = self.propellant.Gamma*np.sqrt(2*gamma/(gamma-1)*(1-pe/pc_ref)**((gamma-1)/gamma))+pe/pc_ref*epsilon

        Ae = 0.8E-3 * 0.18E-3
        epsilon = Ae / self.nozzle.At

        # m_exit = np.zeros(len(t))
        # m = np.zeros(len(t))
        # p_t = np.zeros(len(t))
        # pe = np.zeros(len(t))
        # V_t = np.zeros(len(t))
        # F_t = np.zeros(len(t))
        # P_t = np.zeros(len(t))
        # Tc = np.zeros(len(t))
        # thrust_to_power = np.zeros(len(t))

        # Initial values

        self.m[0, j, k] = mass_flow_rate(self.propellant.Gamma, p0, self.nozzle.At, self.propellant.R, self.Tc)
        self.p_t[0, j, k] = p0
        self.Tc_vec[0, j, k] = self.Tc

        # Initialize iteration
        self.p_t[1, j, k] = V0 * p0 / (V0 + self.m_exit[0, j, k] / self.propellant.rho)
        self.Tc_vec[1, j, k] = chamber_temperature(self.propellant.R, p0)

        m_ideal = mass_flow_rate(self.propellant.Gamma,
                                 self.p_t[0, j, k], self.nozzle.At, self.propellant.R, self.Tc_vec[0, j, k])

        Cd = calc_Cd(m_ideal, self.nozzle.At, self.propellant.gamma)

        # m[1] = mass_flow_rate(self.propellant.Gamma, p_t[0], self.nozzle.At, self.propellant.R, Tc[0])
        self.m[1, j, k] = Cd * m_ideal

        self.m_exit[1, j, k] = self.m_exit[0, j, k] + self.m[1, j, k] * self.dt

        self.V_t[1, j, k] = V0 * self.p_t[1, j, k] / self.p_t[1, j, k]
        # M = p_t[1] * V_t[1] / (vlm.propellant.R * vlm.Tc)

        self.pe[1, j, k], ier = exhaust_pressure(gamma, epsilon, self.p_t[1, j, k])

        if ier == 1:
            CF = self.propellant.Gamma * np.sqrt(2 * gamma / (gamma - 1) * (1 -
                                                 (self.pe[1, j, k] / self.p_t[1, j, k]) ** ((gamma - 1) / gamma))) + self.pe[1, j, k] / self.p_t[1, j, k] * epsilon
        else:
            CF = CF_ref
        self.F_t[1, j, k] = Isp_ref * CF * np.sqrt(self.Tc_vec[1, j, k]) * 9.81 * self.m[1, j, k]

        m_prop_left = m_initial - self.m_exit[1, j, k]

        i = 1

        while self.F_t[i, j, k] > 0.12E-3 and m_prop_left > 0.2E-3:
            i += 1
            self.p_t[i, j, k] = V0 * p0 / (V0 + self.m_exit[i - 1, j, k] / self.propellant.rho)
            if Tc_input is not None:
                self.Tc_vec[i, j, k] = Tc_input[i, j, k]
            else:
                self.Tc_vec[i, j, k] = chamber_temperature(self.propellant.R, self.p_t[i - 1, j, k])
                # m[i, j, k] = p_t[i - 1] * self.nozzle.At * self.propellant.Gamma / np.sqrt(self.propellant.R * Tc[i, j, k])
            if m_input is not None:
                self.m[i, j, k] = m_input[i, j, k]
            else:
                m_ideal = mass_flow_rate(self.propellant.Gamma,
                                         self.p_t[i - 1, j, k],
                                         self.nozzle.At,
                                         self.propellant.R,
                                         self.Tc_vec[i, j, k])
                Cd = calc_Cd(m_ideal, self.nozzle.At, self.propellant.gamma)
                self.m[i, j, k] = Cd * m_ideal

            self.m_exit[i, j, k] = self.m_exit[i - 1, j, k] + self.m[i, j, k] * self.dt
            m_prop_left = m_initial - self.m_exit[i, j, k]

            # V_t[i, j, k] = V0 * p_t[1] / p_t[i, j, k]
            # M = p_t[i, j, k] * V_t[i, j, k] / (vlm.propellant.R * vlm.Tc)
            if pe_input is not None:
                self.pe[i, j, k] = pe_input[i, j, k]
            else:
                self.pe[i, j, k], ier = exhaust_pressure(gamma, epsilon, self.p_t[i, j, k], self.pe[i - 1, j, k])
            if ier == 1:
                CF = self.propellant.Gamma * np.sqrt(2 * gamma / (gamma - 1) *
                                                     (1 - (self.pe[i, j, k] / self.p_t[i, j, k])**((gamma - 1) / gamma))) + self.pe[i, j, k] / self.p_t[i, j, k] * epsilon
            else:
                CF = CF_ref
            Isp = Isp_ref * CF * np.sqrt(self.Tc_vec[i, j, k])
            self.F_t[i, j, k] = Isp * 9.81 * self.m[i, j, k]
            self.P_t[i, j, k] = utils.power(self.m[i, j, k], self.Tc_vec[i, j, k], 4187)
            try:
                self.thrust_to_power[i, j, k] = self.F_t[i, j, k] / self.P_t[i, j, k]
            except BaseException:
                print('Ft', self.F_t[i, j, k], 'Pt', self.P_t[i, j, k], 'Isp', Isp)

        self.burn_time[j, k] = i * self.dt
        print(f'Process {j, k}')
        # return m_exit, m, p_t, pe, V_t, F_t, P_t, Tc, thrust_to_power, burn_time
