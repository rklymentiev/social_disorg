import numpy as np
from tqdm import tqdm
from itertools import product
from datetime import datetime
from multiprocessing import Pool, cpu_count
import params
import os
import shutil
import json

def fitness_function(
        b_vals: np.ndarray,
        s_vals: np.ndarray,
        c_vals: np.ndarray,
        penalty: float,
        s_max: int) -> np.ndarray:
    """
    Calculate fitness value for each offender.

    Parameters
    ----------
    b_vals: np.ndarray
        Array with behavior values for each offender.
    s_vals: np.ndarray
        Array with cue values for each site.
    c_vals: np.ndarray
        Array with cost values for each offender (based on their learning strategy).
    penalty: float
        Exponent for the penalty function (h in the model).
    s_max: int
        Maximum value of cues.

    Returns
    -------
    np.ndarray
        Array with fitness values for each offender. If values are negative, they are set to 0.
    """
    if not(b_vals.shape == s_vals.shape == c_vals.shape):
        raise ValueError("The shape of b_vals, s_vals, and c_vals do not match.")

    # cues and behaviors are integers on a circle, i.e,
    # if s=1 and b=1000, they are adjacent, and the distance should be 1
    d_vals = np.min([np.abs(s_vals - b_vals), s_max - np.abs(s_vals - b_vals)], axis=0)
    w_vals = penalty**(-d_vals.astype(float)) - c_vals
    return np.clip(w_vals, a_min=0, a_max=None)

def social_learning(
        b_vals: np.ndarray,
        l_vals: np.ndarray,
        n:int) -> np.ndarray:
    """
    Social learning and behavior update.

    Parameters
    ----------
    b_vals: np.ndarray
        Array with behavior values for each offender.
    l_vals: np.ndarray
        Array of strategy values for each offender.
    n: int
        Number of offenders/sites.

    Returns
    -------
    np.ndarray
        Array with updated behavior values.
    """
    if not(b_vals.shape == l_vals.shape == (n, )):
        raise ValueError("The shape of b_vals and l_vals do not match.")

    # create a 2D array of neighbor values with shape (2, n)
    # where the first row contains the left neighbors and
    # the second row contains the right neighbors
    neighbors_left = np.take(b_vals, np.arange(-1, n-1), mode='wrap')
    neighbors_right = np.take(b_vals, np.arange(1, n+1), mode='wrap')
    neighbors_both = np.append(neighbors_left.reshape(1, -1), neighbors_right.reshape(1, -1), axis=0)
    # select one of two neighbors and copy their behavior (without an error)
    indx = np.random.choice([0, 1], size=(n,))
    copied_behav = neighbors_both[indx, np.arange(n)]
    cndtn = np.where(l_vals == 1)
    b_vals[cndtn] = copied_behav[cndtn]
    return b_vals

def asocial_learning(
        b_vals: np.ndarray,
        l_vals: np.ndarray,
        s_vals: np.ndarray,
        s_max: int,
        p: float,
        n: int) -> np.ndarray:
    """
    Asocial learning and behavior update.

    Parameters
    ----------
    b_vals: np.ndarray
        Array with behavior values for each offender.
    l_vals: np.ndarray
        Array of strategy values for each offender.
    s_vals: np.ndarray
        Array with cue values for each site.
    s_max: int
        Maximum value of cues/behavior.
    p: float
        Probability of asocial learning being successful.
    n: int
        Number of offenders/sites.

    Returns
    -------
    np.ndarray
        Array with updated behavior values.
    """
    if not (b_vals.shape == l_vals.shape == s_vals.shape == (n,)):
        raise ValueError("The shape of b_vals, l_vals, and s_vals do not match or do not match n.")

    # probability of success of asocial learning
    prob_success = np.random.choice([0, 1], size=(n,), p=[1 - p, p])
    # if successful, the copy the cue without an error
    cndtn = np.where((prob_success == 1) & (l_vals == 0))
    b_vals[cndtn] = s_vals[cndtn]
    # if not successful, learn a random behavior U[1,B]
    cndtn = np.where((prob_success == 0) & (l_vals == 0))
    b_vals[cndtn] = np.random.randint(1, s_max+1, size=(len(cndtn[0]),))
    return b_vals

def behavior_replication(
        b_vals: np.ndarray,
        l_vals: np.ndarray,
        w_vals: np.ndarray,
        n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Stochastic behavior replication with local neighbor transmission.

    Parameters
    ----------
    b_vals: np.ndarray
        Array with behavior values for each offender.
    l_vals: np.ndarray
        Array of strategy values for each offender.
    w_vals: np.ndarray
        Array with fitness values for each offender (used as replication probability).
    n: int
        Number of offenders/sites.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated (b_vals, l_vals) after replication.
    """
    if not (b_vals.shape == l_vals.shape == w_vals.shape == (n,)):
        raise ValueError("The shape of b_vals, l_vals, and w_vals do not match or do not match n.")

    # probability of replication is proportional to fitness value
    prob_replication = np.random.rand(n) < w_vals

    b_vals_new = b_vals.copy()
    l_vals_new = l_vals.copy()

    # indices of replicating agents, in ascending order (same order the
    # original for-loop would have processed them in)
    idx = np.where(prob_replication)[0]
    if idx.size == 0:
        return b_vals_new, l_vals_new

    # for each replicating agent, pick the right (+1) or left (-1) neighbor
    # with equal probability, all at once instead of one draw per agent
    direction = np.where(np.random.rand(idx.size) < 0.5, 1, -1)
    targets = (idx + direction) % n

    # assign the behavior and learning strategy to the chosen neighbor.
    # Note: when two replicating agents target the same neighbor, numpy's
    # fancy-index assignment keeps the *last* write in `targets` order,
    # which is the same "last write wins" semantics as the original
    # ascending-i for-loop (verified to produce identical results given
    # the same random stream).
    b_vals_new[targets] = b_vals[idx]
    l_vals_new[targets] = l_vals[idx]
    return b_vals_new, l_vals_new

def innovation(
        l_vals: np.ndarray,
        n: int,
        innovation_rate: float) -> np.ndarray:
    """
    Innovation step where offenders can randomly change their learning strategy.

    Parameters
    ----------
    l_vals: np.ndarray
        Array of strategy values for each offender.
    n: int
        Number of offenders/sites.
    innovation_rate: float
        Probability of an offender innovating (changing their learning strategy).

    Returns
    -------
    np.ndarray
        Array with updated learning strategy values.
    """
    indx_to_innovate = np.random.rand(n) < innovation_rate
    l_vals[indx_to_innovate] ^= 1
    return l_vals

def residential_instability(
        b_vals: np.ndarray,
        l_vals: np.ndarray,
        n: int,
        prop_to_shuffle: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Residential instability step where a proportion of offenders are randomly shuffled.

    Parameters
    ----------
    b_vals: np.ndarray
        Array with behavior values for each offender.
    l_vals: np.ndarray
        Array with strategy values for each offender.
    n: int
        Number of offenders/sites.
    prop_to_shuffle: float
        Proportion of offenders to shuffle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated (b_vals, l_vals) after residential instability.
    """
    n_pairs = int(prop_to_shuffle * n / 2)
    if n_pairs == 0:
        return b_vals, l_vals

    # draw all pair indices in one call instead of two calls per pair
    # (same distribution and same sequential-swap semantics as before,
    # since each swap still depends on the state left by the previous one)
    idx1_all = np.random.randint(0, n, size=n_pairs)
    idx2_all = np.random.randint(0, n, size=n_pairs)
    for idx1, idx2 in zip(idx1_all, idx2_all):
        # swap behaviors
        b_vals[idx1], b_vals[idx2] = b_vals[idx2], b_vals[idx1]
        # swap strategies
        l_vals[idx1], l_vals[idx2] = l_vals[idx2], l_vals[idx1]
    return b_vals, l_vals

def temporal_change(
        s_vals: np.ndarray,
        prob_change: float,
        n: int,
        s_max: int) -> np.ndarray:
    """
    Perform temporal change of cues with a given probability.

    Parameters
    ----------
    s_vals: np.ndarray
        Array with cues values.
    prob_change: float
        Probability to change cues' value.
    n: int
        Number of sites.
    s_max: int
        Maximum value of cues.

    Returns
    -------
    np.ndarray
        Array with updated cues values.
    """
    if np.random.rand() < prob_change:
        # change the value of cues
        s_val = np.random.randint(1, s_max+1, size=1)  # random cue values in the [1, S] range
        return np.full(shape=(n,), fill_value=s_val)
    else:
        return s_vals

def assign_costs(
        l_vals: np.ndarray,
        c_dict: dict) -> np.ndarray:
    """
    Assign cost values to offenders based on their learning strategies.

    Parameters
    ----------
    l_vals: np.ndarray
        Array of strategy values for each offender.
    c_dict: dict
        Dictionary mapping strategy values to their corresponding costs.

    Returns
    -------
    np.ndarray
        Array with assigned cost values for each offender.
    """
    c_vals = np.full_like(l_vals, fill_value=-1, dtype=float)
    for l, c in c_dict.items():
        c_vals[l_vals == l] = c
    return c_vals

def env_init(
        n: float,
        n_patches: int,
        s_max: int,
        l_options: list,
        ratio_asocial: float,
        c_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize the environment

    Parameters
    ----------
    n: int
        Number of sites/offenders.
    n_patches: int
        Number of patches.
    s_max: int
        Maximum value of cues/behavior.
    l_options: list
        List of possible learning strategies.
    ratio_asocial: float
        Proportion of asocial learners in the population at the initialization.
    c_dict: dict
        Dictionary mapping strategy values to their corresponding costs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Initialized arrays for cues, behaviors, learning strategies, and costs.
    """
    s_vals = np.full(shape=(n,), fill_value=-1)
    # split cues into chunks
    split_points = np.sort(
        np.random.choice(np.arange(1, n), size=n_patches-1, replace=False)
    )
    chunks = np.split(s_vals, split_points)
    for chunk in chunks:
        chunk[:] = np.random.randint(1, s_max+1, size=1)  # random cue values in the [1, S] range
    s_vals = np.concatenate(chunks)

    b_vals = np.full_like(s_vals, fill_value=-1)  # behavior values for each offender (initially set to -1)
    # assign learning strategies according to probability of being asocial or social learners
    l_vals = np.random.choice(l_options, size=(n,), p=[ratio_asocial, 1-ratio_asocial])
    # assign cost values for each strategy
    c_vals = assign_costs(l_vals=l_vals, c_dict=c_dict)
    return s_vals, b_vals, l_vals, c_vals

def run_single_iteration(args: tuple) -> tuple[list, list]:
    """
    Run one full iteration (env init + T timesteps) for a given parameter
    combination. Designed to be called by a worker process, so it takes a
    single tuple argument and re-seeds its own random state.

    Parameters
    ----------
    args: tuple
        (r, q, n_patches, seed) - the free-parameter values for this run and
        a seed for this worker's random state (or None to draw fresh entropy).

    Returns
    -------
    tuple[list, list]
        (ratio_social_learners, average_fitness) time series for this run.
    """
    r, q, n_patches, seed = args
    # Each worker process needs its own random state. Passing `seed=None`
    # (the default when params.SEED is None) draws fresh OS entropy per
    # worker, so parallel runs don't accidentally share the same stream
    # (which can happen with fork-based multiprocessing otherwise).
    np.random.seed(seed)

    cues, behaviors, strategies, costs = env_init(
        n=params.N, n_patches=n_patches, s_max=params.S,
        l_options=params.learning_strategies,
        ratio_asocial=params.prop_asocial, c_dict=params.c)

    ratio_social_learners = []
    average_fitness = []

    for _ in range(params.T):
        # STEP 1&2: LEARNING & BEHAVIORAL DEPLOYMENT
        # social learners
        behaviors = social_learning(b_vals=behaviors, l_vals=strategies, n=params.N)
        # asocial learners
        behaviors = asocial_learning(
            b_vals=behaviors, l_vals=strategies, s_max=params.S,
            s_vals=cues, p=params.prob_success_asocial, n=params.N)

        # STEP 3: FITNESS COMPUTATION
        w = fitness_function(
            b_vals=behaviors, s_vals=cues,
            c_vals=costs, penalty=params.h, s_max=params.S)

        # STEP 4: REPLICATION & DISPERSAL
        behaviors, strategies = behavior_replication(
            b_vals=behaviors, l_vals=strategies,
            w_vals=w, n=params.N)

        # STEP 5: INNOVATION
        strategies = innovation(l_vals=strategies, n=params.N, innovation_rate=params.mu)

        # STEP 6: RESIDENTAL INSTABILITY
        behaviors, strategies = residential_instability(
            b_vals=behaviors, l_vals=strategies,
            n=params.N, prop_to_shuffle=r)

        # reassign cost values for each offenders' strategy
        costs = assign_costs(l_vals=strategies, c_dict=params.c)

        # STEP 7: ENVIRONMENTAL CHANGE
        cues = temporal_change(s_vals=cues, prob_change=q, n=params.N, s_max=params.S)

        # values for each simulation run [time1, time2, ...]
        ratio_social_learners.append(np.mean(strategies))
        average_fitness.append(np.mean(w))

    return ratio_social_learners, average_fitness

if '__main__' == __name__:
    ratio_social_learners_agg = dict()
    average_fitness_agg = dict()

    # n_workers = min(int(cpu_count()/2), params.n_iterations)
    n_workers = params.n_cores

    # iterate over all combinations of free parameters
    for combo_idx, (r, q, n_patches) in enumerate(
            product(params.r_vals, params.q_vals, params.n_patches_vals)):

        key = f"r={r}, q={q}, n_patches={n_patches}"

        # per-iteration seeds: if params.SEED is set, derive a distinct,
        # reproducible seed for each (combo, iteration) pair; if it's None,
        # pass None through so each worker draws its own fresh entropy
        if params.SEED is not None:
            iter_seeds = [params.SEED + combo_idx * params.n_iterations + i
                          for i in range(params.n_iterations)]
        else:
            iter_seeds = [None] * params.n_iterations

        tasks = [(r, q, n_patches, seed) for seed in iter_seeds]

        ratio_social_learners_runs = []
        average_fitness_runs = []
        with Pool(processes=n_workers) as pool:
            for ratio_social_learners, average_fitness in tqdm(
                    pool.imap(run_single_iteration, tasks),
                    total=params.n_iterations,
                    desc=f"Parameters r={r}, q={q}, n_patches={n_patches}"):
                ratio_social_learners_runs.append(ratio_social_learners)
                average_fitness_runs.append(average_fitness)

        # merge all iterations together, [[iteration1_values], [iteration2_values], ...]
        ratio_social_learners_agg[key] = ratio_social_learners_runs
        average_fitness_agg[key] = average_fitness_runs

    # create output folders
    os.makedirs('output', exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join("output", f"{current_time}_results")
    os.makedirs(save_dir, exist_ok=True)

    # copy the params.py file
    shutil.copy("params.py", os.path.join(save_dir, "params.py"))

    with open(os.path.join(save_dir, "ratio_social_learners.json"), mode="w") as f:
        json.dump(ratio_social_learners_agg, f)

    with open(os.path.join(save_dir, "average_fitness.json"), mode="w") as f:
        json.dump(average_fitness_agg, f)

