# %% - imports
import concurrent.futures
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm

import utils
from Engine import Engine

pi = np.pi

At = 4.5E-9

l_tube = 0.3
d_tube = 1.57E-3
V_tube = l_tube * pi * d_tube**2 / 4
rho_H20 = 997

dt = 0.2
t0 = 0
te = 1300

t = np.arange(t0, te, dt)
# p = np.arange(0.2E5, 1.4E5, 0.1E4)
p = np.arange(0.3E5, 1.3E5, 0.1E4)
# V = np.arange(0.05, 0.6, 0.01) * V_tube
V = np.arange(0.05, 0.35, 0.01) * V_tube

m_exit = np.zeros((len(t), len(p), len(V)))
m = np.zeros((len(t), len(p), len(V)))
p_t = np.zeros((len(t), len(p), len(V)))
pe = np.zeros((len(t), len(p), len(V)))
V_t = np.zeros((len(t), len(p), len(V)))
F_t = np.zeros((len(t), len(p), len(V)))
P_t = np.zeros((len(t), len(p), len(V)))
Tc = np.zeros((len(t), len(p), len(V)))
thrust_to_power = np.zeros((len(t), len(p), len(V)))

burn_time = np.zeros((len(p), len(V)))
m_initial = np.zeros((len(p), len(V)))
m_initial_total = np.zeros((len(p), len(V)))

try:
    with open('mass_flow_data', 'rb') as f:
        m_input = pickle.load(f)
except BaseException:
    m_input = None

try:
    with open('chamber_temp_data', 'rb') as f:
        Tc_input = pickle.load(f)
except BaseException:
    Tc_input = None

try:
    with open('exit_pressure_data', 'rb') as f:
        pe_input = pickle.load(f)
except BaseException:
    pe_input = None

# %%


def main_iteration(vlm, p_subspace, V_subspace, row, col, len_p_chunk, len_V_chunk):
    for z, V0 in enumerate(V_subspace):
        for y, p0 in enumerate(p_subspace):
            # pbar.update(1)
            vlm.m_initial[y + col * len_p_chunk, z + row * len_V_chunk] = (V_tube - V0) * vlm.propellant.rho
            # m_initial_total[j, k] = m_initial[j, k] + p_t[0, j, k] * V_t[0, j, k] / (vlm.propellant.R * 600)
            # m_exit[:, j, k], m[:, j, k], p_t[:, j, k], pe[:, j, k], V_t[:, j, k], F_t[:, j, k], P_t[:,
            # j, k], Tc[:, j, k], thrust_to_power[:, j, k], time =
            # executor.submit(vlm.envelope, vlm.m_initial[j, k], V0, p0, m_input, Tc_input, pe_input, j, k)
            vlm.envelope(vlm.m_initial[y + col * len_p_chunk, z + row * len_V_chunk], V0, p0, m_input, Tc_input, pe_input, y +
                         col *
                         len_p_chunk, z +
                         row *
                         len_V_chunk)

    # print('y0', y)
    # print('z0', z)
    # print('x index', row * len_V_chunk)
    # print('y index', col * len_p_chunk)
    # print('y', y - len(p_subspace))
    # print('z', z - len(V_subspace))
    burn_time = vlm.burn_time[(col * len_p_chunk):y + (col * len_p_chunk + 1), (row * len_V_chunk):z + (row * len_V_chunk + 1)]
    while_condition = vlm.while_condition[(col * len_p_chunk):y + (col * len_p_chunk + 1), (row * len_V_chunk):z + (row * len_V_chunk + 1)]
    # return ((col * len_p_chunk, y + (col * len_p_chunk + 1), (row * len_V_chunk), z + (row * len_V_chunk + 1)), burn_time)
    return ((col * len_p_chunk, y + (col * len_p_chunk + 1), (row * len_V_chunk), z + (row * len_V_chunk + 1)), while_condition)


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # with tqdm(total=len(p) * len(V)) as pbar:
        vlm = Engine(
            Tc=550,
            Dt=np.sqrt(4 * At / pi),
            De=2 * np.sqrt(4 * At / pi),
            gamma=1.33,
            rho=rho_H20,
            M=18.0153E-3,
            p=p,
            V=V,
            print_parameters=False,
        )
        num_processes = 4
        print('len p', len(p))
        print('len V', len(V))
        len_p_chunk = len(p) // num_processes
        len_V_chunk = len(V) // num_processes

        n = np.arange(0, num_processes * len_p_chunk, num_processes)
        m = np.arange(0, num_processes * len_V_chunk, num_processes)

        results = []

        for col in range(num_processes):
            for row in range(num_processes):
                results.append(executor.submit(main_iteration, vlm, p[col * len_p_chunk:(col + 1) * len_p_chunk], V[row * len_V_chunk:(row + 1) * len_V_chunk],
                                               row,
                                               col,
                                               len_p_chunk,
                                               len_V_chunk))

        for f in concurrent.futures.as_completed(results):
            output = f.result()
            coords = output[0]
            # vlm.burn_time[coords[0]:coords[1], coords[2]:coords[3]] = output[1]
            vlm.while_condition[coords[0]:coords[1], coords[2]:coords[3]] = output[1]

        burn_time = vlm.burn_time
        while_condition = vlm.while_condition
        # burn_time[j, k] = time

    coordinates = np.unravel_index(np.argmax(burn_time), np.array(burn_time).shape)
    max_burn_time = burn_time[coordinates]
    print('max burn time', max_burn_time)
    print('while condition value', while_condition[92, 8])
    print('while condition value 2', while_condition[70, 5])
    print('while condition value 2', while_condition[70, 25])
    print('coordinates', coordinates)
    plt.figure()
    plt.contourf(V / V_tube, p, while_condition)
    plt.scatter(V[8] / V_tube, p[92], marker='x', s=60)
    plt.scatter(V[5] / V_tube, p[70], marker='x', s=60)
    plt.scatter(V[25] / V_tube, p[70], marker='x', s=60)
    plt.colorbar()
    plt.show()

    with open('while_condition_data', 'wb') as f:
        pickle.dump(while_condition, f)
    # with open('mass_flow_data', 'wb') as f:
    #     pickle.dump(m, f)
    # with open('chamber_temp_data', 'wb') as f:
    #     pickle.dump(Tc, f)
    # with open('exit_pressure_data', 'wb') as f:
    #     pickle.dump(pe, f)
    # with open('burn_time_data', 'wb') as f:
    #     pickle.dump(burn_time, f)
    # with open('thrust_data', 'wb') as f:
    #     pickle.dump(F_t, f)
    # with open('thrust_to_power_data', 'wb') as f:
    #     pickle.dump(thrust_to_power, f)

    F_t[F_t == 0] = None
    P_t[P_t == 0] = None
    thrust_to_power[thrust_to_power == 0] = None
    mean_F_t = np.nanmean(F_t, axis=0)
    mean_thrust_to_power = np.nanmean(thrust_to_power, axis=0)

    # %% - Normalize values

    normalized_burn_time = burn_time / max_burn_time

    normalized_m_initial = m_initial / m_initial[0, 0]

    max_mean_F_t = mean_F_t[np.unravel_index(np.argmax(mean_F_t), np.array(mean_F_t).shape)]
    normalized_mean_thrust = mean_F_t / max_mean_F_t

    max_mean_thrust_to_power = mean_thrust_to_power[np.unravel_index(
        np.nanargmax(mean_thrust_to_power), np.array(mean_thrust_to_power).shape)]
    normalized_mean_thrust_to_power = mean_thrust_to_power / max_mean_thrust_to_power

    w1, w2, w3, w4 = (0.4, 0.25, 0.25, 0.1)
    score = w1 * normalized_burn_time + w2 * normalized_mean_thrust + \
        w3 * normalized_mean_thrust_to_power + w4 * (1 - normalized_m_initial)

    # w1, w2, w3 = (0.4, 0.4, 0.2)
    # score = w1*normalized_burn_time + w2*normalized_mean_thrust + w3*(1-normalized_m_initial)

    plt.figure()
    plt.contourf(V / V_tube, p, score)
    plt.colorbar()
    plt.show()

    max_score_x, max_score_y = np.unravel_index(np.nanargmax(score), np.array(score).shape)
    print(max_score_x, max_score_y)
    # levels = [750, 950, 1250]

    # p_max, V_max = np.unravel_index(np.argmax(burn_time), np.array(burn_time).shape)

    # plt.figure()
    # contour = plt.contour(V / V_tube, p, burn_time, levels=levels)
    # plt.clabel(contour, levels, fontsize=10)
    # # plt.plot(col, row, 'b+')
    # plt.scatter(V[V_max] / V_tube, p[p_max], c='k', marker='x')
    # plt.scatter(V[7] / V_tube, p[100], c='k', marker='o')
    # plt.show()

    # plt.figure()
    # # for F in F_t:
    # plt.plot(t, F_t * 1E3)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Thrust [mN]')
    # plt.legend(p)

    # plt.ylim(0.1, 0.8)
    # plt.show()

    # %% Thrust starting phase
    # p_in = 5E5

    # def model(y, t, p):
    #     c1, c2, c3, n, p_in = p
    #     dydt = 1 / c3 * (c2 * p_in - c2 * y - c1 * y**n)

    #     return dydt

    # n = 0.5

    # c1 = 5E-9
    # c2 = vlm.nozzle.At / vlm.C_star
    # V_plenum = 0.27E-9
    # c3 = V_plenum / (vlm.propellant.R * vlm.Tc)

    # p = (c1, c2, c3, n, p_in)

    # # Initial conditions
    # z0 = 0

    # t = np.arange(0, 2E-3, 0.01E-3)

    # z = odeint(model, z0, t, args=(p,))
    # y = z[:, 0]

    # plt.figure()
    # plt.plot(t * 1E3, y * 1E-5)
    # plt.show()

    # %%
    print('hi')
