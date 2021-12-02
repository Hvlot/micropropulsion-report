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
p = np.arange(0.3E5, 1.5E5, 0.1E4)
# V = np.arange(0.05, 0.6, 0.01) * V_tube
V = np.arange(0.05, 0.39, 0.005) * V_tube
# V = np.arange(0.15, 0.165, 0.005) * V_tube


burn_time = np.zeros((len(p), len(V)))
m_initial = np.zeros((len(p), len(V)))
m_initial_total = np.zeros((len(p), len(V)))


def percentage_loss(val1, val2):
    if val1 > val2:
        raise ValueError('Value 1 should be smaller than value 2')
    return (1 - val1 / val2) * 100


def find_2d_max(arr):
    coordinates = np.unravel_index(np.nanargmax(arr), np.array(arr).shape)
    return arr[coordinates], coordinates


def plot_contour(var, title, label, show_max=False, max_annotation=False, show_max_score=False, max_pressure_idx=95, *args, **kwargs):
    plt.figure()
    plt.contourf(V / V_tube, p, var)
    plt.colorbar(label=label)

    plt.title(title)

    plt.xlabel('V/V_tube [-]')
    plt.ylabel('Pressure [Pa]')

    line = kwargs.get('hlines', {})
    color = line.get('color', 'black')

    if show_max_score:
        plt.scatter(V[max_score_y] / V_tube, p[max_score_x],
                    s=60,
                    marker='x',
                    color=line.get('color', 'black'))
        # plt.text(x=V[max_score_y - kwargs.get('shift_left', 0)] / V_tube, y=p[max_score_x - kwargs.get('shift_down', 0)],
        #          s=f'{max_score:.3f}', horizontalalignment="left")

    elif show_max:
        max_var_score, (max_x, max_y) = find_2d_max(var[:max_pressure_idx, :])
        plt.scatter(V[max_y] / V_tube, p[max_x],
                    s=60,
                    marker='x',
                    color=line.get('color', 'black'))

        print('max annotation', max_annotation)
        if max_annotation:

            plt.text(x=V[max_y - kwargs.get('shift_left', 0)] / V_tube, y=p[max_x - kwargs.get('shift_down', 0)],
                     s=f'{max_var_score:.3f}', horizontalalignment="left")

    if line:
        y = line['y']
        plt.hlines(
            y=y,
            xmin=np.min(V) / V_tube,
            xmax=np.max(V) / V_tube,
            color=color,
            linestyle='--')

    plt.show()

# %%


def main_iteration(vlm, p_subspace, V_subspace, row, col, len_p_chunk, len_V_chunk, len_t, dt):
    pr = cProfile.Profile()
    pr.enable()

    burn_time_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    m_initial_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    m_prop_left_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    mass_flow_rate_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    F_t_range_subarray = np.zeros((len(p_subspace), len(V_subspace)))
    F_t_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    P_t_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    p_t_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    Tc_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))
    Cd_subarray = np.zeros((len_t, len(p_subspace), len(V_subspace)))

    for z, V0 in enumerate(V_subspace):
        for y, p0 in enumerate(p_subspace):
            # pbar.update(1)

            j = y + col * len_p_chunk
            k = z + row * len_V_chunk
            m_initial = (V_tube - V0) * vlm.propellant.rho
            m_initial_subarray[y, z] = m_initial

            burn_time_subarray[y, z], mass_flow_rate_subarray[:, y, z], F_t_subarray[:, y, z], F_t_range_subarray[y, z], P_t_subarray[:,
                                                                                                                                      y, z], m_prop_left_subarray[y, z], p_t_subarray[:, y, z], Tc_subarray[:, y, z], Cd_subarray[:, y, z] = vlm.envelope(m_initial, V0, p0, len_t, dt)

    output_dict = {
        "burn_time": burn_time_subarray,
        "mass_flow_rate": mass_flow_rate_subarray,
        "thrust": F_t_subarray,
        "thrust_range": F_t_range_subarray,
        "power": P_t_subarray,
        "m_initial": m_initial_subarray,
        "m_prop_left": m_prop_left_subarray,
        "chamber_pressure": p_t_subarray,
        "Tc": Tc_subarray,
    }

    if vlm.CASE == 'updated_model':
        output_dict['Cd'] = Cd_subarray

    pr.disable()
    pr.print_stats(sort='tottime')
    return ((col * len_p_chunk, y + 1 + col * len_p_chunk, (row * len_V_chunk), z + 1 + row * len_V_chunk), output_dict)


if __name__ == "__main__":
    start = time.perf_counter()

    vlm = Engine(
        Tc=550,
        Dt=np.sqrt(4 * At / pi),
        De=2 * np.sqrt(4 * At / pi),
        gamma=1.33,
        rho=rho_H20,
        M=18.0153E-3,
        CASE='baseline'
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
    mass_flow_rate = arrays['mass_flow_rate']
    if vlm.CASE == 'updated_model':
        Cd = arrays['thrust_coefficient']

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
            mass_flow_rate[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['mass_flow_rate']
            F_t[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['thrust']
            P_t[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['power']
            p_t[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['chamber_pressure']
            Tc[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['Tc']
            if vlm.CASE == 'updated_model':
                Cd[:, coords[0]:coords[1], coords[2]:coords[3]] = output[1]['Cd']

    p_max_idx = np.where(np.nanmax(P_t, axis=0) > 4)[0][0]
    print('maximum pressure', p[p_max_idx])
    # There should be a better way to do this, but pressures that are too high should not be considered.
    max_burn_time, coordinates = find_2d_max(burn_time[:p_max_idx, :])
    # max_while_condtion, wc_coordinates = find_2d_max(while_condition)
    # print('max value while condition', max_while_condtion, wc_coordinates)

    print('max burn time', max_burn_time)
    print('coordinates', coordinates)

    finish = time.perf_counter()
    print('Time to finish', finish - start, 'seconds')

    F_t[F_t == 0] = None
    P_t[P_t == 0] = None
    m_prop_left[m_prop_left == 0] = None
    Tc[Tc == 0] = None
    mass_flow_rate[mass_flow_rate == 0] = None
    if vlm.CASE == 'updated_model':
        Cd[Cd == 0] = None

    thrust_to_power = F_t / P_t
    # thrust_to_power[thrust_to_power == 0] = None
    # mean_F_t = np.nanmean(F_t, axis=0)
    mean_thrust_to_power = np.nanmean(thrust_to_power, axis=0)

# %% - Normalize values
    # normalized_thrust_range = F_t_range / (F_req_max - F_req_min)
    normalized_thrust_range = F_t_range / np.nanmax(F_t_range)
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

    max_mean_thrust_to_power, _ = find_2d_max(mean_thrust_to_power)
    normalized_mean_thrust_to_power = mean_thrust_to_power / max_mean_thrust_to_power

    Isp = F_t / ((1 - 0.0670) * 9.81 * mass_flow_rate)

# %% - Plot score

    weights = (
        (0.40, 0.30, 0.05, 0.25),
        (0.40, 0.40, 0.05, 0.15),
        (0.40, 0.25, 0.05, 0.30),
        (0.35, 0.25, 0.15, 0.25),
        (0.30, 0.25, 0.20, 0.25),
        (0.25, 0.25, 0.25, 0.25),
        (0.20, 0.30, 0.20, 0.30),
        (0.40, 0.25, 0.10, 0.25),
    )

    for weight in weights:
        w1, w2, w3, w4 = weight
        if sum((w1, w2, w3, w4)) != 1:
            raise ValueError('Sum of weights not equal to 1')
        score = w1 * normalized_burn_time + w2 * normalized_thrust_range + \
            w3 * (1 - normalized_mean_P_t) + w4 * normalized_m_prop_left

        max_score, coords = find_2d_max(score[:p_max_idx, :])

        best_scores = []
        score2 = copy.deepcopy(score)
        best_score, coordinates = find_2d_max(score2[:p_max_idx, :])
        while best_score >= 0.98 * max_score:
            best_score, coordinates = find_2d_max(score2[:p_max_idx, :])
            best_scores.append((best_score, coordinates))
            score2[coordinates] = 0
        # print(best_scores)
        max_score_x, max_score_y = coords

        # if vlm.CASE == 'baseline':
        #     plot_contour(score, 'Baseline model score', '-', show_max=True, hlines=p[p_max_idx], shift_left = -2, shift_down = 2)
        # elif vlm.CASE == 'updated_model':
        #     plot_contour(score, 'Updated model score', '-', show_max=True, hlines=p[p_max_idx])

        print('weights', weight)
        print(f'{max_score:.3f}, {p[max_score_x]*1E-5:.2f}, {V[max_score_y] / V_tube:.3f}')
        print(f'N > 98% max score: {len(best_scores)}')
        print()

# %% - Print operational envelope
    print(f'Operational envelope for p0 = {p[max_score_x]*1E-5:.2f} bar, V0/V_tube = {V[max_score_y]/V_tube:.3f}')
    print(f'Max burn time: \t\t {max_burn_time:.1f} s')
    print(f'Thrust range: \t\t {F_t_range[max_score_x, max_score_y]*1E3:.3f} mN')
    print(f'Maximum thrust: \t {np.nanmax(F_t[:, max_score_x, max_score_y])*1E3:.3f} mN')
    print(f'Initial propellant: \t {m_initial[max_score_x, max_score_y]*1E3:.3f} g')
    print(f'Remaining propellant: \t {m_prop_left[max_score_x, max_score_y]*1E3:.3f} g')
    print(f'Average Isp: \t\t {np.nanmean(Isp[:, max_score_x, max_score_y]):.3f} s')

# %% - Plot some things for the updated model
    if vlm.CASE == 'updated_model':
        title = 'Burn time (updated_model)'
    elif vlm.CASE == 'baseline':
        title = 'Burn time (baseline)'
    plot_contour(
        burn_time,
        title,
        's',
        show_max=True,
        max_annotation=True,
        hlines={'y': p[p_max_idx]},
        max_pressure_idx=p_max_idx,
        shift_left=-2,
        shift_down=2
    )

    if vlm.CASE == 'updated_model':
        title = 'Score (updated model)'
    elif vlm.CASE == 'baseline':
        title = 'Score (baseline)'
    plot_contour(
        score,
        title,
        '-',
        show_max=True,
        max_annotation=True,
        hlines={'y': p[p_max_idx]},
        max_pressure_idx=p_max_idx,
        shift_left=-2,
        shift_down=2
    )

    if vlm.CASE == 'updated_model':
        title = 'Mean chamber temperature (updated model)'
    elif vlm.CASE == 'baseline':
        title = 'Mean chamber temperature (baseline)'
    plot_contour(
        np.nanmean(Tc, axis=0),
        title,
        'K',
        show_max_score=True,
        hlines={'y': p[p_max_idx]},
        shift_left=-2,
        shift_down=2
    )

    if vlm.CASE == 'updated_model':
        plot_contour(
            np.nanmean(Cd, axis=0),
            'Discharge coefficient',
            '-',
            hlines={'y': p[p_max_idx]},
            shift_left=-2,
            show_max_score=True,
            shift_down=2
        )

    if vlm.CASE == 'updated_model':
        title = 'Thrust range (updated_model)'
    elif vlm.CASE == 'baseline':
        title = 'Thrust range (baseline)'
    plot_contour(
        F_t_range * 1E3,
        title,
        'mN',
        show_max_score=True,
        hlines={'y': p[p_max_idx]},
        shift_left=-2,
        shift_down=2
    )
    if vlm.CASE == 'updated_model':
        title = 'Propellant mass left (updated_model)'
    elif vlm.CASE == 'baseline':
        title = 'Propellant mass left (baseline)'
    plot_contour(
        m_prop_left * 1E3,
        title,
        'g',
        show_max_score=True,
        hlines={'y': p[p_max_idx], 'color': 'white'},
        shift_left=-2,
        shift_down=2
    )

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


# # %% - Propellant mass left from baseline model

# with open('pickle/08-11-2021/baseline_model/m_prop_left', 'rb') as f:
#     baseline_m_prop_left = pickle.load(f)

# plot_contour(baseline_m_prop_left*1E3, )
# plt.figure()
# plt.contourf(V / V_tube, p, baseline_m_prop_left * 1E3, 'Baseline model propellant mass left', 'g')


# %% - All feasible solutions
    best_scores = []
    score2 = copy.deepcopy(score)
    best_score, coordinates = find_2d_max(score2[:p_max_idx, :])
    best_burn_time = burn_time[coordinates]
    best_thrust_range = F_t_range[coordinates]
    for i in score2:
        best_score, coordinates = find_2d_max(score2[:p_max_idx, :])
        best_burn_time = burn_time[coordinates]
        best_thrust_range = F_t_range[coordinates]
        if best_score >= 0.98 * max_score:
            best_scores.append((best_score, coordinates))
        score2[coordinates] = 0

    plt.figure()
    plt.contourf(V[5:15] / V_tube, p[90:115], score[90:115, 5:15], alpha=0.7)
    plt.xlabel('V/V_tube [-]')
    plt.ylabel('Pressure [Pa]')
    plt.title('Score')
    plt.grid(alpha=0.3)
    plt.colorbar()

    p_min = min(map(lambda x: x[1][0], best_scores))
    V_max = max(map(lambda x: x[1][1], best_scores))

    burn_times = []

    for i in best_scores:
        if i[0] == max_score:
            print('i', i, 'V', V[i[1][1]] / V_tube, 'p', p[i[1][0]])
            plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="red")
        elif m_prop_left[i[1]] > 0.2E-3:
            plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="cyan")
        elif F_t_range[i[1]] > F_t_range[max_score_x, max_score_y]:
            plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="olive")
        else:
            plt.scatter(V[i[1][1]] / V_tube, p[i[1][0]], marker='x', s=15, color="black")
        plt.scatter(V[V_max] / V_tube, p[p_min], marker='x', s=15, color="orange")

        burn_times.append(burn_time[i[1][0], i[1][1]])

    print(f'Minimum found burn time: {min(burn_times):.1f} [s] (-{percentage_loss(min(burn_times), max_burn_time):.2f}%)')


# %% - Compare two extremes on the score contour
    point_1 = (max_score_x, max_score_y)
    p_min = min(map(lambda x: x[1][0], best_scores))
    V_max = max(map(lambda x: x[1][1], best_scores))
    point_2 = (p_min, V_max)

    coords_max_score = f'(V/V_tube: {V[max_score_y]/V_tube:.3f}, p: {p[max_score_x]*1E-5:.3f} bar)'
    coords_suboptimal = f'(V/V_tube: {V[V_max]/V_tube:.3f}, p: {p[p_min]*1E-5:.3f} bar)'

    # Input power
    plt.figure()
    plt.plot(t, P_t[:, max_score_x, max_score_y], label=f'{coords_max_score[1:-1]}')
    plt.plot(t, P_t[:, p_min, V_max], label=f'{coords_suboptimal[1:-1]}')

    plt.legend()
    plt.grid(alpha=0.5)
    plt.title('Input power')
    plt.xlabel('Time [s]')
    plt.ylabel('Power [W]')

    plt.show()

    # Thrust
    plt.figure()
    plt.plot(t[3:], F_t[3:, max_score_x, max_score_y] * 1E3, label=f'{coords_max_score[1:-1]}')
    plt.plot(t[3:], F_t[3:, p_min, V_max] * 1E3, label=f'{coords_suboptimal[1:-1]}')

    plt.legend()
    plt.grid(alpha=0.5)
    plt.title('Thrust')
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [mN]')

    plt.show()

    # percentage_loss = (1 - burn_time[p_min, V_max] / max_burn_time) * 100
    percentage_difference = percentage_loss(burn_time[p_min, V_max], max_burn_time)

    print(f'Burn time: {max_burn_time} [s] {coords_max_score}')
    print(f'Burn time: {burn_time[p_min, V_max]:.1f} [s] -{percentage_difference:.2f}% {coords_suboptimal}')

    thrust_range_percentage_loss = percentage_loss(np.nanmax(F_t[:, p_min, V_max]), np.nanmax(F_t[:, max_score_x, max_score_y]))
    print(f'Maximum thrust: {np.nanmax(F_t[:, max_score_x, max_score_y]*1E3):.2f} [mN] {coords_max_score}')
    print(f'Maximum thrust: {np.nanmax(F_t[:, p_min, V_max]*1E3):.2f} [mN] -{thrust_range_percentage_loss:.2f}% {coords_suboptimal}')

    input_power_percentage_loss = percentage_loss(np.nanmax(P_t[:, p_min, V_max]), np.nanmax(P_t[:, max_score_x, max_score_y]))
    print(f'Maximum input power: {np.nanmax(P_t[:, max_score_x, max_score_y]):.2f} [W] {coords_max_score}')
    print(f'Maximum input power: {np.nanmax(P_t[:, p_min, V_max]):.2f} [W] -{input_power_percentage_loss:.2f}% {coords_suboptimal}')

    propellant_mass_gained = 100 * m_prop_left[p_min, V_max] / m_prop_left[max_score_x, max_score_y] - 100
    print(f'Propellant mass left: {m_prop_left[max_score_x, max_score_y]*1E3:.3f} [g] {coords_max_score}')
    print(f'Propellant mass left: {m_prop_left[p_min, V_max]*1E3:.3f} [g] +{propellant_mass_gained:.2f}% {coords_suboptimal}')

# %% - Thrust range gradient

    grad = np.gradient(F_t_range[:-1, :-1] * 1E3)
    mag_grad = np.sqrt(grad[0]**2 + grad[1]**2)

    plt.figure(figsize=(7, 4))

    plt.contour(V[:-1] / V_tube, p[:-1], F_t_range[:-1, :-1] * 1E3)
    # plt.pcolor(V[:-1]/V_tube, p[:-1], F_t_range[:-1,:-1]*1E3, cmap='plasma')
    # plt.colorbar(label='Thrust range [mN]')

    plt.contourf(V[:-1] / V_tube, p[:-1], mag_grad, cmap="plasma")
    # plt.pcolor(V[:-1]/V_tube, p[:-1], mag_grad[:-1,:-1], cmap='plasma')
    plt.colorbar(label=r'$|\nabla \Delta F|$')
    plt.scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color='black')
    # plt.scatter(V[10]/V_tube, p[80], marker='x', s=60, color='black')
    # plt.scatter(V[10]/V_tube, p[40], marker='x', s=60, color='black')
    # plt.scatter(V[20]/V_tube, p[40], marker='x', s=60, color='black')

    plt.title('Thrust range')
    plt.xlabel('V/V_tube')
    plt.ylabel('Pressure [Pa]')

    plt.tight_layout()

# %% - Burn time gradient

    grad = np.gradient(burn_time[:-1, :-1])
    mag_grad = np.sqrt(grad[0]**2 + grad[1]**2)

    plt.figure(figsize=(7, 4))
    plt.contour(V[:-1] / V_tube, p[:-1], burn_time[:-1, :-1])
    plt.colorbar(label='Burn time [s]')
    plt.contourf(V[:-1] / V_tube, p[:-1], mag_grad, cmap="plasma")
    plt.colorbar(label=r'$|\nabla t_b|$')
    plt.scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color='black')

    plt.title('Burn time')
    plt.xlabel('V/V_tube')
    plt.ylabel('Pressure [Pa]')

    plt.tight_layout()

# %% - Propellant mass left gradient

    grad = np.gradient(m_prop_left * 1E3)
    mag_grad = np.sqrt(grad[0]**2 + grad[1]**2)

    plt.figure(figsize=(7, 4))
    plt.contour(V / V_tube, p, m_prop_left * 1E3)
    plt.colorbar(label='Propellant mass left [g]')
    plt.contourf(V / V_tube, p, mag_grad, cmap="plasma")
    plt.colorbar(label=r'$|\nabla m|$')
    plt.scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color='white')

    plt.title('Propellant mass left - gradient')
    plt.xlabel('V/V_tube')
    plt.ylabel('Pressure [Pa]')

    plt.tight_layout()

# %% - Mean input power

    grad = np.gradient(1 - normalized_mean_P_t)
    mag_grad = np.sqrt(grad[0]**2 + grad[1]**2)

    plt.figure(figsize=(7, 4))

    plt.contour(V / V_tube, p, 1 - normalized_mean_P_t)
    # plt.pcolor(V/V_tube, p, 1-normalized_mean_P_t, cmap="plasma")
    plt.colorbar(label='Mean input power [W]')
    plt.contourf(V / V_tube, p, mag_grad, cmap="plasma")
    # plt.pcolor(V/V_tube, p, mag_grad, vmin=0, vmax=0.014, cmap="plasma")
    plt.colorbar(label=r'$|\nabla P_{input}|$')
    plt.scatter(V[max_score_y] / V_tube, p[max_score_x], marker='x', s=60, color='white')

    plt.title('Mean input power')
    plt.xlabel('V/V_tube')
    plt.ylabel('Pressure [Pa]')

    plt.tight_layout()

# %% - Thrust range vs mean thrust
    p1_idx = 95
    V1_idx = 2
    p2_idx = -2
    V2_idx = -2

    print(f'Top left: {np.nanmean(F_t[:, 95, 2])*1E3:.3f} mN')
    print(f'Top right: {np.nanmean(F_t[:, -2, -2])*1E3:.3f} mN')

    plt.figure()

    plt.plot(t[3:], F_t[3:, p1_idx, V1_idx] * 1E3, label=f'p: {p[p1_idx]*1E-5:.2f} bar, V/V_tube: {V[V1_idx]/V_tube:.2f}')
    plt.plot(t[3:], F_t[3:, p2_idx, V2_idx] * 1E3, label=f'p: {p[p2_idx]*1E-5:.2f} bar, V/V_tube: {V[V2_idx]/V_tube:.2f}')

    plt.title('Thrust')
    plt.xlabel('Time [s]')
    plt.ylabel('Mean thrust [mN]')
    plt.grid(alpha=0.55)

    plt.legend()
