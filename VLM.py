# %% - imports
import concurrent.futures
# import pickle
import cProfile
import time

import matplotlib.pyplot as plt
import numpy as np

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

# m_exit = np.zeros((len(t), len(p), len(V)))
# m = np.zeros((len(t), len(p), len(V)))
# p_t = np.zeros((len(t), len(p), len(V)))
# pe = np.zeros((len(t), len(p), len(V)))
# V_t = np.zeros((len(t), len(p), len(V)))
# F_t = np.zeros((len(t), len(p), len(V)))
# P_t = np.zeros((len(t), len(p), len(V)))
# Tc = np.zeros((len(t), len(p), len(V)))
# thrust_to_power = np.zeros((len(t), len(p), len(V)))

burn_time = np.zeros((len(p), len(V)))
m_initial = np.zeros((len(p), len(V)))
m_initial_total = np.zeros((len(p), len(V)))

# try:
#     with open('mass_flow_data', 'rb') as f:
#         m_input = pickle.load(f)
# except BaseException:
#     m_input = None

# try:
#     with open('chamber_temp_data', 'rb') as f:
#         Tc_input = pickle.load(f)
# except BaseException:
#     Tc_input = None

# try:
#     with open('exit_pressure_data', 'rb') as f:
#         pe_input = pickle.load(f)
# except BaseException:
#     pe_input = None

# %%


def main_iteration(vlm, p_subspace, V_subspace, row, col, len_p_chunk, len_V_chunk, len_t, dt):
    pr = cProfile.Profile()
    pr.enable()

    # Process stuff to be profiled

    burn_time_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    while_condition_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    F_t_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    P_t_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))

    for z, V0 in enumerate(V_subspace):
        for y, p0 in enumerate(p_subspace):
            # pbar.update(1)

            j = y + col * len_p_chunk
            k = z + row * len_V_chunk
            m_initial = (V_tube - V0) * vlm.propellant.rho
            # m_initial_total[j, k] = m_initial[j, k] + p_t[0, j, k] * V_t[0, j, k] / (vlm.propellant.R * 600)
            # m_exit[:, j, k], m[:, j, k], p_t[:, j, k], pe[:, j, k], V_t[:, j, k], F_t[:, j, k], P_t[:,
            # j, k], Tc[:, j, k], thrust_to_power[:, j, k], time =
            # executor.submit(vlm.envelope, vlm.m_initial[j, k], V0, p0, m_input, Tc_input, pe_input, j, k)

            burn_time_subarray[y, z], while_condition_subarray[y, z], F_t_subarray[:, y,
                                                                                   z], P_t_subarray[:, y, z] = vlm.envelope(m_initial, V0, p0, len_t, dt)

    output_dict = {
        "burn_time": burn_time_subarray,
        "while_condition": while_condition_subarray,
        "thrust": F_t_subarray,
        "power": P_t_subarray
    }

    # burn_time = vlm.burn_time[(col * len_p_chunk):y + (col * len_p_chunk + 1), (row * len_V_chunk):z + (row * len_V_chunk + 1)]
    # while_condition = vlm.while_condition[(col * len_p_chunk):y + (col * len_p_chunk + 1), (row * len_V_chunk):z + (row * len_V_chunk + 1)]
    pr.disable()
    pr.print_stats(sort='tottime')  # sort as you wish
    return ((col * len_p_chunk, y + (col * len_p_chunk + 1), (row * len_V_chunk), z + (row * len_V_chunk + 1)), output_dict)
    # return ((col * len_p_chunk, y + (col * len_p_chunk + 1), (row * len_V_chunk), z + (row * len_V_chunk + 1)), while_condition)


if __name__ == "__main__":
    start = time.perf_counter()

    # with open('exclusion_coords', 'rb') as f:
    #     exclusion_coords = pickle.load(f)

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

    arrays = utils.initialize_envelope(p, V, te, dt)
    burn_time = arrays['burn_time']
    while_condition = arrays['while_condition']
    # thrust_to_power = arrays['thrust_to_power']
    F_t = arrays['F_t']
    P_t = arrays['P_t']
    t = arrays['time']
    dt = arrays['dt']

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # with tqdm(total=len(p) * len(V)) as pbar:
        num_processes = 4

        len_p_chunk = len(p) // num_processes
        len_V_chunk = len(V) // num_processes

        results = []

        for col in range(num_processes):
            for row in range(num_processes):
                p_subarray = p[col * len_p_chunk:(col + 1) * len_p_chunk]
                V_subarray = V[row * len_V_chunk:(row + 1) * len_V_chunk]
                results.append(executor.submit(main_iteration, vlm, p_subarray, V_subarray, row, col, len_p_chunk, len_V_chunk, len(t), dt))

        for f in concurrent.futures.as_completed(results):
            output = f.result()
            # Structure of the output: [(tuple of the region ranges), {dictionary with output arrays}]
            coords = output[0]
            burn_time[coords[0]:coords[1], coords[2]:coords[3]] = output[1]['burn_time']
            while_condition[coords[0]:coords[1], coords[2]:coords[3]] = output[1]['while_condition']
            F_t[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['thrust']
            P_t[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['power']

    def find_2d_max(arr):
        coordinates = np.unravel_index(np.argmax(arr), np.array(arr).shape)
        return arr[coordinates], coordinates

    # coordinates = np.unravel_index(np.argmax(burn_time), np.array(burn_time).shape)
    max_burn_time, coordinates = find_2d_max(burn_time)
    max_while_condtion, wc_coordinates = find_2d_max(while_condition)
    print('max value while condition', max_while_condtion, wc_coordinates)

    print('max burn time', max_burn_time)
    print('coordinates', coordinates)

    finish = time.perf_counter()
    print('Time to finish', finish - start, 'seconds')

    F_t[F_t == 0] = None
    P_t[P_t == 0] = None
    thrust_to_power = F_t / P_t
    # thrust_to_power[thrust_to_power == 0] = None
    mean_F_t = np.nanmean(F_t, axis=0)
    mean_thrust_to_power = np.nanmean(thrust_to_power, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    img = axs[0, 0].contourf(V / V_tube, p, burn_time)
    axs[0, 0].set_title('Burn time [s]')
    # plt.scatter(V[5] / V_tube, p[70], marker='x', s=60)
    # plt.scatter(V[25] / V_tube, p[70], marker='x', s=60)
    plt.colorbar(img, ax=axs[0, 0])

    img = axs[1, 0].contourf(V / V_tube, p, while_condition, levels=[0, 1, 2])
    axs[1, 0].set_title('Thrust < 0.12 mN (1) or m_left < 0.2 g (2)')
    plt.colorbar(img, ax=axs[1, 0])

    plt.scatter(V[coordinates[1]] / V_tube, p[coordinates[0]], marker='x', s=60)

    img = axs[0, 1].contourf(V / V_tube, p, mean_F_t * 1E3)
    axs[0, 1].set_title('Mean thrust [mN]')
    plt.colorbar(img, ax=axs[0, 1])
    img = axs[1, 1].contourf(V / V_tube, p, mean_thrust_to_power)
    axs[1, 1].set_title('Thrust to power [N/W]')
    plt.colorbar(img, ax=axs[1, 1])
    # plt.imshow(while_condition)
    plt.tight_layout()
    plt.show()

    # with open('while_condition_data', 'wb') as f:
    #     pickle.dump(while_condition, f)
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
