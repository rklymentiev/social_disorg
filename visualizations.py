import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
import matplotlib as mpl
import importlib.util
import json
import os


if __name__ == "__main__":
    # select the folder with simulation results
    results_path = askdirectory(title='Select the folder with simulation results (e.g., 20260528_145008_results)')
    if not results_path:
        raise ValueError("No folder selected. Please select a folder with simulation results.")


    # load params.py from the selected folder
    spec = importlib.util.spec_from_file_location(
        "params",
        f"{results_path}/params.py"
    )

    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    # load the simulation results files
    with open(os.path.join(results_path, "ratio_social_learners.json"), 'r') as f:
        ratio_social_learners = json.load(f)
    with open(os.path.join(results_path, "average_fitness.json"), 'r') as f:
        average_fitness = json.load(f)

    # create visualizations

    # check what parameter has been used
    if len(params.q_vals) > 1:
        free_parameter = {
            'name': "$q$",
            'name_to_plot': 'Environmental chance probability $q$',
            'values': params.q_vals
        }
    elif len(params.r_vals) > 1:
        free_parameter = {
            'name': "$r$",
            'name_to_plot': 'Turnover fraction $r$',
            'values': params.r_vals
        }
    elif len(params.n_patches_vals) > 1:
        free_parameter = {
            'name': "number of patches",
            'name_to_plot': 'Number of patches',
            'values': params.n_patches_vals
        }
    else:
        raise ValueError("No free parameter found. Please check the params.py file.")

    # fraction of social learners over time
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.arange(1, params.T + 1)
    cmap = plt.get_cmap('Blues_r')

    norm = mpl.colors.Normalize(
        vmin=min(free_parameter['values']),
        vmax=max(free_parameter['values']
                 ))

    # plot lines
    for i, comb in enumerate(ratio_social_learners.keys()):
        param_val = free_parameter['values'][i]
        mean = np.array(ratio_social_learners[comb]).mean(axis=0)
        std = np.array(ratio_social_learners[comb]).std(axis=0)
        color = cmap(norm(param_val))
        ax.plot(x, mean, color=color)

        # ax.fill_between(
        #     x,
        #     mean - std,
        #     mean + std,
        #     alpha=0.3,
        #     color=color
        # )

    # Colorbar
    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        pad=0.15)
    cbar.set_label(free_parameter['name'])

    ax.set_xlabel('Time step $t$')
    ax.set_ylabel('Fraction of social learners')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, params.T + params.T * 0.05])
    ax.set_box_aspect(.6)
    ax.minorticks_on()
    plt.savefig(os.path.join(results_path, f"fraction_social_by_{free_parameter['name']}_t.{params.img_format}"))
    plt.close()

    # average fitness value over time
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.arange(1, params.T + 1)
    cmap = plt.get_cmap('Blues_r')
    norm = mpl.colors.Normalize(
        vmin=min(free_parameter['values']),
        vmax=max(free_parameter['values']))

    # plot lines
    for i, comb in enumerate(average_fitness.keys()):
        param_val = free_parameter['values'][i]
        mean = np.array(average_fitness[comb]).mean(axis=0)
        std = np.array(average_fitness[comb]).std(axis=0)
        color = cmap(norm(param_val))
        ax.plot(x, mean, color=color)
        # ax.fill_between(
        #     x,
        #     mean - std,
        #     mean + std,
        #     alpha=0.3,
        #     color=color
        # )
    # Colorbar
    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        pad=0.15)
    cbar.set_label(free_parameter['name'])
    ax.set_xlabel('Time step $t$')
    ax.set_ylabel('Average fitness $w$')
    # ax.set_title(f"Average over {params.n_iterations} iterations")
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, params.T + params.T * 0.05])
    ax.set_box_aspect(.6)
    ax.minorticks_on()
    plt.savefig(os.path.join(results_path, f"avg_fitness_by_{free_parameter['name']}_t.{params.img_format}"))
    plt.close()

    # fraction of social learners by free parameter (average over the last n time steps)
    fig, ax = plt.subplots()
    tmp_mean = []
    tmp_std = []

    for comb in ratio_social_learners.keys():
        arr = np.array(ratio_social_learners[comb])
        tmp_mean.append(arr[:, -params.n_last:].mean())
        tmp_std.append(arr[:, -params.n_last:].std())

    tmp_mean = np.array(tmp_mean)
    tmp_std = np.array(tmp_std)

    ax.plot(free_parameter['values'], tmp_mean)
    ax.fill_between(
        free_parameter['values'],
        tmp_mean - tmp_std,
        tmp_mean + tmp_std,
        alpha=0.3
    )

    ax.set_ylim([0, 1.05])
    ax.set_xlabel(free_parameter['name_to_plot'])
    ax.set_ylabel('Fraction of social learners')
    ax.set_box_aspect(0.6)
    ax.minorticks_on()
    plt.savefig(os.path.join(results_path, f"fraction_social_by_{free_parameter['name']}.{params.img_format}"))
    plt.close()

    # average fitness value by free parameter (average over the last n time steps)
    fig, ax = plt.subplots()

    tmp_mean = []
    tmp_std = []

    for comb in average_fitness.keys():
        arr = np.array(average_fitness[comb])
        tmp_mean.append(arr[:, -params.n_last:].mean())
        tmp_std.append(arr[:, -params.n_last:].std())

    tmp_mean = np.array(tmp_mean)
    tmp_std = np.array(tmp_std)

    ax.plot(free_parameter['values'], tmp_mean)
    ax.fill_between(
        free_parameter['values'],
        tmp_mean - tmp_std,
        tmp_mean + tmp_std,
        alpha=0.3
    )

    ax.set_ylim([0, 1.1])
    ax.set_xlabel(free_parameter['name_to_plot'])
    ax.set_ylabel('Average fitness $w$')
    ax.set_box_aspect(0.6)
    ax.minorticks_on()
    plt.savefig(os.path.join(results_path, f"average_fitness_by_{free_parameter['name']}.{params.img_format}"))
    plt.close()

