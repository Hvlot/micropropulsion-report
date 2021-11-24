# %% - imports
import concurrent.futures
import copy
import cProfile
import pickle
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
rho_H20 = 999.7

dt = 0.1
t0 = 0
te = 1500

t = np.arange(t0, te, dt)
# p = np.arange(0.2E5, 1.4E5, 0.1E4)
p = np.arange(0.3E5, 1.3E5, 0.1E4)
# V = np.arange(0.05, 0.6, 0.01) * V_tube
V = np.arange(0.05, 0.37, 0.005) * V_tube

burn_time = np.zeros((len(p), len(V)))
m_initial = np.zeros((len(p), len(V)))
m_initial_total = np.zeros((len(p), len(V)))


def find_2d_max(arr):
    coordinates = np.unravel_index(np.nanargmax(arr), np.array(arr).shape)
    return arr[coordinates], coordinates

# %%


def main_iteration(vlm, p_subspace, V_subspace, row, col, len_p_chunk, len_V_chunk, len_t, dt):
    pr = cProfile.Profile()
    pr.enable()

    burn_time_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    m_initial_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    m_prop_left_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    while_condition_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    F_t_range_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    F_t_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    P_t_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    p_t_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    Tc_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))

    for z, V0 in enumerate(V_subspace):
        for y, p0 in enumerate(p_subspace):
            # pbar.update(1)

            j = y + col * len_p_chunk
            k = z + row * len_V_chunk
            m_initial = (V_tube - V0) * vlm.propellant.rho
            m_initial_subarray[y, z] = m_initial

            burn_time_subarray[y, z], while_condition_subarray[y, z], F_t_subarray[:, y, z], F_t_range_subarray[y, z], P_t_subarray[:,
                                                                                                                                    y, z], m_prop_left_subarray[y, z], p_t_subarray[:, y, z], Tc_subarray[:, y, z] = vlm.envelope(m_initial, V0, p0, len_t, dt)

    output_dict = {
        "burn_time": burn_time_subarray,
        "while_condition": while_condition_subarray,
        "thrust": F_t_subarray,
        "thrust_range": F_t_range_subarray,
        "power": P_t_subarray,
        "m_initial": m_initial_subarray,
        "m_prop_left": m_prop_left_subarray,
        "chamber_pressure": p_t_subarray,
        "Tc": Tc_subarray
    }

    pr.disable()
    pr.print_stats(sort='tottime')
    return ((col * len_p_chunk, y + 1 + col * len_p_chunk, (row * len_V_chunk), z + 1 + row * len_V_chunk), output_dict)


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
    )

    arrays = utils.initialize_envelope(p, V, te, dt)
    burn_time = arrays['burn_time']
    m_prop_left = arrays['m_prop_left']
    while_condition = arrays['while_condition']
    # thrust_to_power = arrays['thrust_to_power']
    F_t = arrays['F_t']
    F_t_range = arrays['F_t_range']
    P_t = arrays['P_t']
    p_t = arrays['p_t']
    Tc = arrays['Tc']
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
            m_initial[coords[0]:coords[1], coords[2]:coords[3]] = output[1]['m_initial']
            m_prop_left[coords[0]:coords[1], coords[2]:coords[3]] = output[1]['m_prop_left']
            F_t_range[coords[0]:coords[1], coords[2]:coords[3]] = output[1]['thrust_range']
            while_condition[coords[0]:coords[1], coords[2]:coords[3]] = output[1]['while_condition']
            F_t[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['thrust']
            P_t[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['power']
            p_t[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['chamber_pressure']
            Tc[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['Tc']

    # There should be a better way to do this, but pressures that are too high should not be considered.
    max_burn_time, coordinates = find_2d_max(burn_time[:96, :])
    max_while_condtion, wc_coordinates = find_2d_max(while_condition)
    print('max value while condition', max_while_condtion, wc_coordinates)

    print('max burn time', max_burn_time)
    print('coordinates', coordinates)

    finish = time.perf_counter()
    print('Time to finish', finish - start, 'seconds')

    F_t[F_t == 0] = None
    P_t[P_t == 0] = None
    m_prop_left[m_prop_left == 0] = None
    thrust_to_power = F_t / P_t
    # thrust_to_power[thrust_to_power == 0] = None
    # mean_F_t = np.nanmean(F_t, axis=0)
    mean_thrust_to_power = np.nanmean(thrust_to_power, axis=0)

    # %% - Normalize values

    normalized_burn_time = burn_time / max_burn_time

    normalized_m_initial = m_initial / m_initial[0, 0]

    max_m_prop_left, _ = find_2d_max(m_prop_left)
    normalized_m_prop_left = m_prop_left / max_m_prop_left

    mean_P_t = np.nanmean(P_t, axis=0)
    max_mean_P_t, _ = find_2d_max(mean_P_t)
    normalized_mean_P_t = mean_P_t / max_mean_P_t

    # max_mean_F_t, _ = find_2d_max(mean_F_t)
    # normalized_mean_thrust = mean_F_t / max_mean_F_t
    F_req_max = 3E-3
    F_req_min = 0.12E-3
    normalized_thrust_range = F_t_range / (F_req_max - F_req_min)

    max_mean_thrust_to_power, _ = find_2d_max(mean_thrust_to_power)
    normalized_mean_thrust_to_power = mean_thrust_to_power / max_mean_thrust_to_power

# %% - Plot score
    w1, w2, w3, w4 = (0.40, 0.25, 0.10, 0.25)
    score = w1 * normalized_burn_time + w2 * normalized_thrust_range + \
        w3 * (1 - normalized_mean_P_t) + w4 * normalized_m_prop_left

    for (x, y), elem in np.ndenumerate(score):
        if np.nanmax(P_t[:, x, y]) >= 4.0:
            score[x, y] = 0
        if p[x] > 125000:
            score[x, y] = 0

    max_score, coords = find_2d_max(score)

    best_scores = []
    score2 = copy.deepcopy(score)
    best_score, coordinates = find_2d_max(score2)
    while best_score > 0.98 * max_score:
        best_score, coordinates = find_2d_max(score2)
        best_scores.append((best_score, coordinates))
        score2[coordinates] = 0
    print(best_scores)
    max_score_x, max_score_y = coords
    print(max_score_x, max_score_y)

    plt.figure()
    plt.contourf(V / V_tube, p, score / max_score)
    plt.colorbar()

    plt.title('Baseline model score')
    plt.xlabel('Initial volume [V/V_tube]')
    plt.ylabel('Pressure [pa]')

    plt.scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color="black")
    x_text = V[max_score_y + 2] / V_tube
    y_text = p[max_score_x - 2]
    plt.text(s=round(max_score, 3), x=x_text, y=y_text, horizontalalignment="left")

    plt.show()

# # %% - Plot contours
#     fig, axs = plt.subplots(3, 2, figsize=(12, 6))

#     img = axs[0, 0].contourf(V / V_tube, p, burn_time)
#     axs[0, 0].set_title('Burn time [s]')
#     # plt.scatter(V[5] / V_tube, p[70], marker='x', s=60)
#     # plt.scatter(V[25] / V_tube, p[70], marker='x', s=60)
#     plt.colorbar(img, ax=axs[0, 0])

#     img = axs[1, 0].contourf(V / V_tube, p, while_condition, levels=[0, 1, 2])
#     axs[1, 0].set_title('Thrust < 0.12 mN (1) or m_left < 0.2 g (2)')
#     plt.colorbar(img, ax=axs[1, 0])

#     img = axs[0, 1].contourf(V / V_tube, p, normalized_thrust_range)
#     axs[0, 1].set_title('Normalized thrust range [-]')
#     plt.colorbar(img, ax=axs[0, 1])
#     img = axs[1, 1].contourf(V / V_tube, p, mean_thrust_to_power)
#     axs[1, 1].set_title('Thrust to power [N/W]')
#     plt.colorbar(img, ax=axs[1, 1])
#     img = axs[2, 0].contourf(V / V_tube, p, normalized_m_prop_left)
#     axs[2, 0].set_title('Propellant left [-]')
#     plt.colorbar(img, ax=axs[2, 0])
#     img = axs[2, 1].contourf(V / V_tube, p, score)
#     axs[2, 1].set_title('Score [-]')
#     plt.colorbar(img, ax=axs[2, 1])
#     # plt.imshow(while_condition)

#     axs[0, 0].scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color="white")
#     axs[0, 1].scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color="white")
#     axs[1, 0].scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color="white")
#     axs[1, 1].scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color="white")
#     axs[2, 0].scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color="white")
#     axs[2, 1].scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color="white")

#     plt.tight_layout()
#     plt.show()

# %% - dump variables to pickle (Cd and div loss model)
# with open('pickle/09-11-2021/Cd_and_div_loss/Tc', 'wb') as f:
#     pickle.dump(Tc, f)

# with open('pickle/09-11-2021/Cd_and_div_loss/P_t', 'wb') as f:
#     pickle.dump(P_t, f)

# with open('pickle/09-11-2021/Cd_and_div_loss/p_t', 'wb') as f:
#     pickle.dump(p_t, f)

# with open('pickle/09-11-2021/Cd_and_div_loss/F_t', 'wb') as f:
#     pickle.dump(F_t, f)

# with open('pickle/09-11-2021/Cd_and_div_loss/thrust_to_power', 'wb') as f:
#     pickle.dump(thrust_to_power, f)

# with open('pickle/09-11-2021/Cd_and_div_loss/m_prop_left', 'wb') as f:
#     pickle.dump(m_prop_left, f)

# with open('pickle/09-11-2021/Cd_and_div_loss/burn_time', 'wb') as f:
#     pickle.dump(burn_time, f)

# with open('pickle/09-11-2021/Cd_and_div_loss/score', 'wb') as f:
#     pickle.dump(score, f)

# # %% - dump variables to pickle (baseline model)
# with open('pickle/08-11-2021/baseline_model/Tc', 'wb') as f:
#     pickle.dump(Tc, f)

# with open('pickle/08-11-2021/baseline_model/P_t', 'wb') as f:
#     pickle.dump(P_t, f)

# with open('pickle/08-11-2021/baseline_model/pressure', 'wb') as f:
#     pickle.dump(p_t, f)

# with open('pickle/08-11-2021/baseline_model/F_t', 'wb') as f:
#     pickle.dump(F_t, f)

# with open('pickle/08-11-2021/baseline_model/thrust_to_power', 'wb') as f:
#     pickle.dump(thrust_to_power, f)

# with open('pickle/08-11-2021/baseline_model/m_prop_left', 'wb') as f:
#     pickle.dump(m_prop_left, f)

# with open('pickle/08-11-2021/baseline_model/burn_time', 'wb') as f:
#     pickle.dump(burn_time, f)

# with open('pickle/08-11-2021/baseline_model/score', 'wb') as f:
#     pickle.dump(score, f)


# # %% - Thrust range from baseline model

# with open('pickle/08-11-2021/baseline_model/F_t', 'rb') as f:
#     baseline_F_t = pickle.load(f)

# thrust_range = (np.nanmax(baseline_F_t, axis=0) - np.nanmin(baseline_F_t, axis=0)) / (3E-3 - 0.12E-3)

# plt.figure()
# plt.contourf(V / V_tube, p, thrust_range)

# plt.title('Baseline model thrust range')
# plt.xlabel('Initial volume [V/V_tube]')
# plt.ylabel('Pressure [pa]')

# plt.colorbar()
# plt.show()

# %% - Score from baseline model

# with open('pickle/08-11-2021/baseline_model/score', 'rb') as f:
#     baseline_score = pickle.load(f)
# with open('pickle/08-11-2021/baseline_model/F_t', 'rb') as f:
#     baseline_F_t = pickle.load(f)

# thrust_range = (np.nanmax(baseline_F_t, axis=0) - np.nanmin(baseline_F_t, axis=0)) / (3E-3 - 0.12E-3)
# max_thrust_range, _ = find_2d_max(thrust_range)
# baseline_normalized_thrust_range = thrust_range / max_thrust_range

# with open('pickle/08-11-2021/baseline_model/burn_time', 'rb') as f:
#     baseline_burn_time = pickle.load(f)

# baseline_max_burn_time, baseline_coords = find_2d_max(baseline_burn_time)
# baseline_normalized_burn_time = baseline_burn_time / baseline_max_burn_time

# with open('pickle/08-11-2021/baseline_model/P_t', 'rb') as f:
#     baseline_P_t = pickle.load(f)

# baseline_mean_P_t = np.nanmean(baseline_P_t, axis=0)
# baseline_max_mean_P_t, _ = find_2d_max(baseline_mean_P_t)
# baseline_normalized_mean_P_t = baseline_mean_P_t / baseline_max_mean_P_t

# with open('pickle/08-11-2021/baseline_model/m_prop_left', 'rb') as f:
#     baseline_m_prop_left = pickle.load(f)

# baseline_max_m_prop_left, _ = find_2d_max(baseline_m_prop_left)
# baseline_normalized_m_prop_left = baseline_m_prop_left / baseline_max_m_prop_left

# w1, w2, w3, w4 = (0.40, 0.25, 0.10, 0.25)
# baseline_score = w1 * baseline_normalized_burn_time + w2 * baseline_normalized_thrust_range + \
#     w3 * (1 - baseline_normalized_mean_P_t) + w4 * baseline_normalized_m_prop_left

# baseline_max_score, baseline_coords = find_2d_max(baseline_score)
# score2 = copy.deepcopy(baseline_score)
# highest_scores = []
# for _ in range(10):
#     highest_score, coords = find_2d_max(score2)
#     score2[coords] = 0
#     highest_scores.append((highest_score, coords))


# plt.figure()
# plt.contourf(V / V_tube, p, baseline_score / baseline_max_score)

# plt.title('Baseline model score')
# plt.xlabel('Initial volume [V/V_tube]')
# plt.ylabel('Pressure [pa]')
# plt.colorbar()

# plt.scatter(V[baseline_coords[1]] / V_tube, p[baseline_coords[0]], marker='x', s=60, color="black")
# plt.scatter(V[highest_scores[4][1][1]] / V_tube, p[highest_scores[4][1][0]], marker='x', s=60, color="black")
# x_text = V[baseline_coords[1] + 2] / V_tube
# y_text = p[baseline_coords[0] - 2]
# plt.text(s=round(baseline_max_score, 3), x=x_text, y=y_text, horizontalalignment="left")
# plt.text(s=round(baseline_score[87, 14], 3), x=V[14 + 2] / V_tube, y=p[87 - 2], horizontalalignment="left")
# plt.plot()


# # %% - Burn time surface from baseline model

# with open('pickle/08-11-2021/baseline_model/burn_time', 'rb') as f:
#     baseline_burn_time = pickle.load(f)

# baseline_max_burn_time, baseline_coords = find_2d_max(baseline_burn_time[:96, :])

# plt.figure()
# plt.contourf(V / V_tube, p, baseline_burn_time / baseline_max_burn_time)
# plt.hlines(y=125000, xmin=0.05, xmax=max(V / V_tube), color="black", linestyles='--')

# plt.title('Baseline model burn time')
# plt.xlabel('Initial volume [V/V_tube]')
# plt.ylabel('Pressure [pa]')
# plt.colorbar()

# plt.scatter(V[baseline_coords[1]] / V_tube, p[baseline_coords[0]], marker='x', s=60, color="black")
# x_text = V[baseline_coords[1] + 2] / V_tube
# y_text = p[baseline_coords[0] - 2]
# plt.text(s=f'{round(baseline_max_burn_time,3)} [s]', x=x_text, y=y_text, horizontalalignment="left")
# plt.plot()


# # %% - Propellant mass left from baseline model

# with open('pickle/08-11-2021/baseline_model/m_prop_left', 'rb') as f:
#     baseline_m_prop_left = pickle.load(f)

# plt.figure()
# plt.contourf(V / V_tube, p, baseline_m_prop_left * 1E3)

# plt.title('Baseline model propellant mass left')
# plt.xlabel('Initial volume [V/V_tube]')
# plt.ylabel('Pressure [pa]')

# plt.colorbar()
# plt.show()

# # Max thrust
# plt.figure()
# plt.contourf(V / V_tube, p, np.nanmax(baseline_F_t, axis=0))

# plt.title('Baseline model max thrust')
# plt.xlabel('Initial volume [V/V_tube]')
# plt.ylabel('Pressure [pa]')

# plt.colorbar()
# plt.show()

# # Min thrust
# plt.figure()
# plt.contourf(V / V_tube, p, np.nanmin(baseline_F_t, axis=0) * 1E3)

# plt.title('Baseline model min thrust')
# plt.xlabel('Initial volume [V/V_tube]')
# plt.ylabel('Pressure [pa]')

# plt.colorbar()
# plt.show()

# # %% - Plot line of best scores
# best_scores = []
# score2 = copy.deepcopy(score)
# best_score, coordinates = find_2d_max(score2)
# best_burn_time = burn_time[coordinates]
# best_thrust_range = F_t_range[coordinates]
# # while (best_score >= 0.95 * max_score
# #        and best_burn_time > 0.95 * burn_time[max_score_x, max_score_y]
# #        and best_thrust_range > 0.9 * F_t_range[max_score_x, max_score_y]):
# while best_score >= 0.95 * max_score:
#     best_score, coordinates = find_2d_max(score2)
#     best_burn_time = burn_time[coordinates]
#     best_thrust_range = F_t_range[coordinates]
#     best_scores.append((best_score, coordinates))
#     score2[coordinates] = 0

# best_score_1 = best_scores[4]
# best_score_2 = best_scores[10]
# best_score_3 = best_scores[17]
# best_score_4 = best_scores[29]
# best_score_5 = best_scores[-5]

# x2 = best_score_1[1][1]
# y2 = best_score_1[1][0]
# x3 = best_score_2[1][1]
# y3 = best_score_2[1][0]
# x4 = best_score_3[1][1]
# y4 = best_score_3[1][0]
# x5 = best_score_4[1][1]
# y5 = best_score_4[1][0]
# # x6 = best_score_5[1][1]
# # y6 = best_score_5[1][0]

# x = [V[max_score_y] / V_tube, V[x2] / V_tube, V[x3] / V_tube, V[x4] / V_tube, V[x5] / V_tube]
# y = [p[max_score_x], p[y2], p[y3], p[y4], p[y5]]

# plt.figure()
# plt.contourf(V / V_tube, p, score)
# plt.colorbar()

# plt.title('Baseline model score')
# plt.xlabel('Initial volume [V/V_tube]')
# plt.ylabel('Pressure [pa]')

# plt.scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color="black")
# plt.scatter(V[x5] / V_tube, p[y5], marker='x', s=60, color="black")
# # plt.scatter(V[x3] / V_tube, p[y3], marker='x', s=60, color="black")
# plt.plot(x, y)

# x_text = V[max_score_y + 2] / V_tube
# y_text = p[max_score_x - 2]
# x_text_2 = V[x5 + 2] / V_tube
# y_text_2 = p[y5 - 2]
# # x_text_3 = V[x3 + 2] / V_tube
# # y_text_3 = p[y3 - 2]

# plt.text(s=f'{round(max_score, 3)}, {burn_time[max_score_x, max_score_y]}', x=x_text, y=y_text, horizontalalignment="left")
# plt.text(s=f'{round(best_score_4[0], 3)}, {round(burn_time[y5, x5], 1)}', x=x_text_2, y=y_text_2, horizontalalignment="left")
# # plt.text(s=f'{round(best_score_3[0], 3)}, {burn_time[y3, x3]}', x=x_text_3, y=y_text_3, horizontalalignment="left")

# plt.show()

# V1 = round(V[max_score_y] / V_tube, 3)
# V2 = round(V[x5] / V_tube, 3)
# V3 = round(V[x3] / V_tube, 3)

# p1 = round(p[max_score_x] * 1E-5, 2)
# p2 = round(p[y5] * 1E-5, 2)
# p3 = round(p[y3] * 1E-5, 2)

# max_P_t = np.nanmax(P_t, axis=0)

# plt.figure()
# plt.plot(t, P_t[:, max_score_x, max_score_y], label=f'p: {p1}, V: {V1}')
# plt.plot(t, P_t[:, y5, x5], label=f'p: {p2}, V: {V2}')

# plt.title('Input power')
# plt.xlabel('Time [s]')
# plt.ylabel('Power [W]')
# plt.ylim((0.5, 4))
# plt.legend()
# plt.grid()
# plt.show()

# print(f'Propellant mass left: {round(m_prop_left[max_score_x, max_score_y]*1E3, 4)} g (p: {p1}, V: {V1})')
# print(f'Propellant mass left: {round(m_prop_left[y5, x5]*1E3, 4)} g (p: {p2}, V: {V2})')
# print(f'Propellant mass left: {round(m_prop_left[y3, x3]*1E3, 4)} g (p: {p3}, V: {V3})')

# print()

# print(f'Thrust range: {round(F_t_range[max_score_x, max_score_y]*1E3, 4)} mN (p: {p1}, V: {V1})')
# percentage_difference = 100 - F_t_range[y5, x5] / F_t_range[max_score_x, max_score_y] * 100
# percentage_difference_2 = 100 - F_t_range[y3, x3] / F_t_range[max_score_x, max_score_y] * 100
# print(f'Thrust range: {round(F_t_range[y5, x5]*1E3, 4)} mN (-{round(percentage_difference, 1)}%) (p: {p2}, V: {V2})')
# print(f'Thrust range: {round(F_t_range[y3, x3]*1E3, 4)} mN (-{round(percentage_difference_2, 1)}%) (p: {p3}, V: {V3})')

# print()

# print(f'Maximum power: {round(max_P_t[max_score_x, max_score_y], 2)} W (p: {p1}, V: {V1})')
# percentage_difference = 100 - max_P_t[y5, x5] / max_P_t[max_score_x, max_score_y] * 100
# percentage_difference_2 = 100 - max_P_t[y3, x3] / max_P_t[max_score_x, max_score_y] * 100
# print(f'Maximum power: {round(max_P_t[y5, x5], 2)} W (-{round(percentage_difference, 1)}%) (p: {p2}, V: {V2})')
# print(f'Maximum power: {round(max_P_t[y3, x3], 2)} W (-{round(percentage_difference_2, 1)}%) (p: {p3}, V: {V3})')

# # %% - All feasible solutions
# best_scores = []
# score2 = copy.deepcopy(score)
# best_score, coordinates = find_2d_max(score2)
# best_burn_time = burn_time[coordinates]
# best_thrust_range = F_t_range[coordinates]
# for i in score2:
#     # while best_score >= 0.95 * max_score:
#     best_score, coordinates = find_2d_max(score2)
#     best_burn_time = burn_time[coordinates]
#     best_thrust_range = F_t_range[coordinates]
#     # if (best_score >= 0.95 * max_score
#     #     and best_burn_time > 0.95 * burn_time[max_score_x, max_score_y]
#     #     and best_thrust_range > 0.90 * F_t_range[max_score_x, max_score_y]
#     #         and np.nanmax(P_t[:, coordinates]) < 3.7):
#     # if (best_burn_time > 0.97 * burn_time[max_score_x, max_score_y]
#     #     and best_thrust_range > 0.85 * F_t_range[max_score_x, max_score_y]):
#     if best_score >= 0.98 * max_score:
#         best_scores.append((best_score, coordinates))
#     score2[coordinates] = 0

# plt.figure()
# plt.contourf(V[5:25] / V_tube, p[70:97], score[70:97, 5:25], alpha=0.5)
# plt.xlabel('V/V_tube [-]')
# plt.ylabel('Pressure [Pa]')
# plt.title('Score')
# plt.grid(alpha=0.3)
# plt.colorbar()

# p_min = min(map(lambda x: x[1][0], best_scores))
# V_max = max(map(lambda x: x[1][1], best_scores))

# for i in best_scores:
#     # Janky way of taking the first point
#     if i[0] == max_score:
#         print('i', i, 'V', V[i[1][1]] / V_tube, 'p', p[i[1][0]])
#         plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="red")
#     # elif p[i[1][0]] < 109000 and V[i[1][1]]/V_tube >= 0.12:
#         # print('i', i, 'V', V[i[1][1]] / V_tube, 'p', p[i[1][0]])
#         # plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="orange")
#     elif m_prop_left[i[1]] > m_prop_left[max_score_x, max_score_y]:
#         plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="cyan")
#     elif F_t_range[i[1]] >= F_t_range[max_score_x, max_score_y]:
#         plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="orange")
#     else:
#         plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="black")
#     # plt.scatter(V[V_max] / V_tube, p[p_min], marker='x', s=15, color="orange")

# # %% - Compare two extremes on the
# point_1 = (max_score_x, max_score_y)
# p_min = min(map(lambda x: x[1][0], best_scores))
# V_max = max(map(lambda x: x[1][1], best_scores))
# point_2 = (p_min, V_max)

# # Input power
# plt.figure()
# plt.plot(t, P_t[:, max_score_x, max_score_y], label=f'V/V_tube: {round(V[max_score_y]/V_tube, 3)}, p: {round(p[max_score_x]*1E-5, 3)} bar')
# plt.plot(t, P_t[:, p_min, V_max], label=f'V/V_tube: {round(V[V_max]/V_tube, 3)}, p: {p[p_min]*1E-5} bar')

# plt.legend()
# plt.grid(alpha=0.5)
# plt.title('Input power')
# plt.xlabel('Time [s]')
# plt.ylabel('Power [W]')

# plt.show()

# # Thrust
# plt.figure()
# plt.plot(t[3:], F_t[3:, max_score_x, max_score_y] * 1E3, label=f'V/V_tube: {round(V[max_score_y]/V_tube, 3)}, p: {round(p[max_score_x]*1E-5, 3)} bar')
# plt.plot(t[3:], F_t[3:, p_min, V_max] * 1E3, label=f'V/V_tube: {round(V[V_max]/V_tube, 3)}, p: {p[p_min]*1E-5} bar')

# plt.legend()
# plt.grid(alpha=0.5)
# plt.title('Thrust')
# plt.xlabel('Time [s]')
# plt.ylabel('Thrust [mN]')

# plt.show()

# print(
#     f'Maximum thrust: {round(np.nanmax(F_t[:, max_score_x, max_score_y]*1E3), 2)} [mN] (V/V_tube: {round(V[max_score_y]/V_tube, 3)}, p: {round(p[max_score_x]*1E-5, 3)} bar)')
# print(f'Maximum thrust: {round(np.nanmax(F_t[:, p_min, V_max]*1E3), 2)} [mN] (V/V_tube: {round(V[V_max]/V_tube, 3)}, p: {p[p_min]*1E-5} bar)')

# print(
#     f'Maximum input power: {round(np.nanmax(P_t[:, max_score_x, max_score_y]), 2)} [W] (V/V_tube: {round(V[max_score_y]/V_tube, 3)}, p: {round(p[max_score_x]*1E-5, 3)} bar)')
# print(f'Maximum input power: {round(np.nanmax(P_t[:, p_min, V_max]), 2)} [W] (V/V_tube: {round(V[V_max]/V_tube, 3)}, p: {p[p_min]*1E-5} bar)')

# print(
#     f'Propellant mass left: {round(m_prop_left[max_score_x, max_score_y]*1E3, 3)} [g] (V/V_tube: {round(V[max_score_y]/V_tube, 3)}, p: {round(p[max_score_x]*1E-5, 3)} bar)')
# print(f'Propellant mass left: {round(m_prop_left[p_min, V_max]*1E3, 3)} [g] (V/V_tube: {round(V[V_max]/V_tube, 3)}, p: {p[p_min]*1E-5} bar)')

# # %% - Thrust range gradient

# grad_y = np.gradient(F_t_range*1E3, axis=0)
# grad_x = np.gradient(F_t_range*1E3, axis=1)
# mag = np.sqrt(grad_y**2+grad_x**2)

# plt.figure(figsize=(8,5))
# plt.contour(V/V_tube, p, burn_time)
# plt.colorbar(label="Burn time [s]")

# plt.contourf(V/V_tube, p, grad_y, cmap="plasma")
# plt.colorbar(label="dF/dp [mN/Pa]")

# plt.title('Thrust range gradient (Isochor)')
# plt.xlabel('V/V_tube [-]')
# plt.ylabel('Pressure [Pa]')

# plt.tight_layout()

# plt.figure(figsize=(8,5))
# plt.contour(V/V_tube, p, burn_time)
# plt.colorbar(label="Burn time [s]")

# plt.contourf(V/V_tube, p, grad_x, cmap="plasma")
# plt.colorbar(label="dF/d(V/V_tube) [mN]")

# plt.title('Thrust range gradient (Isobar)')
# plt.xlabel('V/V_tube [-]')
# plt.ylabel('Pressure [Pa]')

# plt.tight_layout()

# # %%
# plt.figure(figsize=(8,5))
# plt.contour(V/V_tube, p, burn_time)
# plt.colorbar(label="Burn time [s]")

# plt.contourf(V/V_tube, p, mag, cmap="plasma")
# plt.colorbar()

# plt.title('Thrust range gradient magnitude')
# plt.xlabel('V/V_tube [-]')
# plt.ylabel('Pressure [Pa]')

# plt.tight_layout()
