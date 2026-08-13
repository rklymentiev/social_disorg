import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib.util
import os
import sys


def get_runs(df: pd.DataFrame, col: str, value: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull out the (n_iterations, T) time series for avg_fitness and
    fraction_social for one value of the free parameter.

    Parameters
    ----------
    df: pd.DataFrame
        Full results loaded from results.csv.gz.
    col: str
        Column name of the free parameter (e.g. 'r', 'q', 'gamma').
    value: float
        The free parameter's value to select rows for.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (avg_fitness, fraction_social), each shape (n_iterations, T),
        with rows ordered by iteration and columns ordered by t.
    """
    # np.isclose rather than == since free-parameter values may be floats
    # (e.g. r_vals rounded to 2 decimals) that don't compare exactly equal
    # after a round-trip through the CSV
    subset = df[np.isclose(df[col], value)]
    if subset.empty:
        raise ValueError(f"No rows found in results.csv.gz for free-parameter value {value}.")

    duplicated = subset.duplicated(subset=["iteration", "t"], keep=False)
    if duplicated.any():
        # this means the CSV has more than one row for the same
        # (iteration, t) after filtering on the free parameter alone -
        # almost always because another parameter also varies across
        # multiple values in params.py, so filtering on `col` doesn't
        # isolate a single run. pivot() would otherwise fail here with an
        # opaque "Index contains duplicate entries" error.
        other_free_cols = [
            c for c in ("r", "q", "n_patches", "p_edge", "gamma")
            if c != col and subset[c].nunique() > 1
        ]
        if other_free_cols:
            details = ", ".join(f"{c}={sorted(subset[c].unique())}" for c in other_free_cols)
            raise ValueError(
                f"Rows for {col}={value} contain more than one value for: {details}. "
                f"This script assumes only one free parameter varies at a time - check "
                f"that params.py has a single value for every free parameter except {col}."
            )
        raise ValueError(
            f"Rows for {col}={value} contain duplicate (iteration, t) entries, but "
            f"every other column is identical too - this almost always means the "
            f"same parameter combination was simulated and appended to "
            f"results.csv.gz more than once, typically because the free-parameter "
            f"list used to generate it (e.g. {col}_vals in params.py) had a "
            f"duplicate or near-duplicate value. Check for that, or regenerate "
            f"results.csv.gz."
        )

    # pivot to (iteration, t) so each row is one iteration's full time
    # series - unlike a manual reshape, this can't silently misalign rows
    # if some entries were duplicated, missing, or out of order
    avg_fitness = subset.pivot(index="iteration", columns="t", values="avg_fitness").values
    fraction_social = subset.pivot(index="iteration", columns="t", values="fraction_social").values
    return avg_fitness, fraction_social


def generate_visualizations(results_path: str) -> None:
    """
    Load results.csv.gz and params.py from a simulation output folder and
    save all four plots back into that same folder.

    Parameters
    ----------
    results_path: str
        Path to a results folder produced by social_disorg.run_simulation()
        (e.g. "output/20260813_110813_results"), containing params.py and
        results.csv.gz.
    """
    # load params.py from the selected folder
    spec = importlib.util.spec_from_file_location(
        "params",
        f"{results_path}/params.py"
    )

    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    # load the simulation results (pandas detects the gzip compression from
    # the .gz extension automatically)
    data = pd.read_csv(os.path.join(results_path, "results.csv.gz"))

    # check what parameter has been used
    if len(params.q_vals) > 1:
        free_parameter = {
            'name': "$q$",
            'name_to_plot': 'Environmental chance probability $q$',
            'values': params.q_vals,
            'col': "q",
        }
    elif len(params.r_vals) > 1:
        free_parameter = {
            'name': "$r$",
            'name_to_plot': 'Turnover fraction $r$',
            'values': params.r_vals,
            'col': "r",
        }
    elif len(params.n_patches_vals) > 1:
        free_parameter = {
            'name': "number of patches",
            'name_to_plot': 'Number of patches',
            'values': params.n_patches_vals,
            'col': "n_patches",
        }
    elif len(getattr(params, 'p_vals', [0])) > 1:
        free_parameter = {
            'name': "$p$",
            'name_to_plot': 'Social network edge probability $p$',
            'values': params.p_vals,
            'col': "p_edge",
        }
    elif len(getattr(params, 'gamma_vals', [0])) > 1:
        free_parameter = {
            'name': "γ",
            'name_to_plot': 'Population turnover rate $\\gamma$',
            'values': params.gamma_vals,
            'col': "gamma",
        }
    else:
        raise ValueError("No free parameter found. Please check the params.py file.")

    print(f"Free parameter: {free_parameter['name']}.")

    # catch a duplicate (or near-duplicate, after rounding) value in the
    # free-parameter list itself - e.g. two equal entries in q_vals from a
    # copy-paste mistake. That makes social_disorg.py simulate and append
    # the same combination twice, which otherwise only surfaces later as an
    # opaque "duplicate (iteration, t)" error inside get_runs().
    values = np.asarray(free_parameter['values'], dtype=float)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if np.isclose(values[i], values[j]):
                raise ValueError(
                    f"{free_parameter['col']}_vals in params.py has duplicate (or "
                    f"near-duplicate) values {values[i]} and {values[j]}. Each value "
                    f"should appear once - remove the duplicate and re-run the "
                    f"simulation, since results.csv.gz was likely written with that "
                    f"combination simulated and appended twice."
                )

    # pull out the (n_iterations, T) arrays for each free-parameter value once,
    # reused across all four plots below instead of re-filtering the data each time
    runs_by_value = {
        val: get_runs(data, free_parameter['col'], val)
        for val in free_parameter['values']
    }

    # fraction of social learners over time
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.arange(1, params.T + 1)
    cmap = plt.get_cmap('Blues_r')

    norm = mpl.colors.Normalize(
        vmin=min(free_parameter['values']),
        vmax=max(free_parameter['values']
                 ))

    # plot lines
    for param_val in free_parameter['values']:
        _, fraction_social_runs = runs_by_value[param_val]
        mean = fraction_social_runs.mean(axis=0)
        std = fraction_social_runs.std(axis=0)
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
    for param_val in free_parameter['values']:
        avg_fitness_runs, _ = runs_by_value[param_val]
        mean = avg_fitness_runs.mean(axis=0)
        std = avg_fitness_runs.std(axis=0)
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

    for param_val in free_parameter['values']:
        _, fraction_social_runs = runs_by_value[param_val]
        tmp_mean.append(fraction_social_runs[:, -params.n_last:].mean())
        tmp_std.append(fraction_social_runs[:, -params.n_last:].std())

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

    for param_val in free_parameter['values']:
        avg_fitness_runs, _ = runs_by_value[param_val]
        tmp_mean.append(avg_fitness_runs[:, -params.n_last:].mean())
        tmp_std.append(avg_fitness_runs[:, -params.n_last:].std())

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

    print(f"Visualizations saved in {results_path}.")


if __name__ == "__main__":
    # accept the results folder as a command-line argument (e.g. from
    # run_all.py chaining straight from social_disorg.run_simulation()) so
    # this can run non-interactively; fall back to the folder-picker dialog
    # when run standalone with no argument
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # imported lazily: only needed for the interactive fallback, so a
        # wrapper script that always passes a path (e.g. run_all.py) never
        # needs tkinter installed at all
        from tkinter.filedialog import askdirectory
        results_path = askdirectory(title='Select the folder with simulation results (e.g., 20260528_145008_results)')
        if not results_path:
            raise ValueError("No folder selected. Please select a folder with simulation results.")

    generate_visualizations(results_path)