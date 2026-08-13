import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from datetime import datetime
from multiprocessing import Pool, cpu_count
import params
import os
import shutil

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


def ring_neighbors_mask(n: int) -> np.ndarray:
    """
    Boolean (n, n) mask of the fixed ring-lattice edges (each agent linked
    to its two spatial neighbors). Doesn't depend on p_edge, so it's built
    once and reused both to seed the social network and to know which
    edges are structural (tied to site position, not resident identity)
    when population turnover replaces an agent.

    Parameters
    ----------
    n: int
        Number of agents.

    Returns
    -------
    np.ndarray
        (n, n) boolean mask, True where two agents are ring neighbors.
    """
    mask = np.zeros((n, n), dtype=bool)
    idx = np.arange(n)
    mask[idx, (idx + 1) % n] = True
    mask[idx, (idx - 1) % n] = True
    return mask

def build_social_network(
        n: int,
        p_edge: float,
        ring_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the starting social network: the ring lattice plus optional
    random Erdos-Renyi links controlled by p_edge. The adjacency matrix is
    kept as mutable state (not just discarded after deriving the neighbor
    list) so population turnover can later rewire individual agents'
    social ties without rebuilding the whole network.

    Parameters
    ----------
    n: int
        Number of agents.
    p_edge: float
        Probability of a random edge between any given pair of agents.
        0 disables the social network entirely, leaving only the two ring
        neighbors (matching the original behavior exactly).
    ring_mask: np.ndarray
        (n, n) boolean mask from ring_neighbors_mask.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        adj: (n, n) boolean adjacency matrix.
        neighbor_list: (n, n) int array. For agent i, the first degrees[i]
            entries of row i are the indices of its neighbors (ring +
            social, order doesn't matter). Columns beyond degrees[i] are
            unused padding and are never selected.
        degrees: (n,) int array with each agent's neighbor count.
    """
    adj = ring_mask.copy()

    if p_edge > 0:
        # symmetric Erdos-Renyi edges, no self-loops, built with a single
        # random draw over the upper triangle and mirrored
        rand_mat = np.triu(np.random.rand(n, n) < p_edge, k=1)
        adj |= rand_mat | rand_mat.T

    np.fill_diagonal(adj, False)

    degrees = adj.sum(axis=1)
    # push True (neighbor) entries to the front of each row; the column
    # index at each sorted position is the corresponding neighbor's index
    neighbor_list = np.argsort(~adj, axis=1, kind='stable')
    return adj, neighbor_list, degrees

def rewire_social_network(
        adj: np.ndarray,
        ring_mask: np.ndarray,
        neighbor_list: np.ndarray,
        degrees: np.ndarray,
        idx_replace: np.ndarray,
        p_edge: float,
        n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update the social network after population turnover. A new, naive
    individual keeps the ring (spatial) neighbors tied to their site - that
    connection is about location, not the previous resident - but loses
    the predecessor's social-network ties and draws fresh ones with
    probability p_edge, since a naive arrival hasn't built any
    relationships yet. Only the rows/columns actually touched by the
    turnover are recomputed (not the whole n x n network), which keeps
    this cheap even though it runs every timestep.

    Parameters
    ----------
    adj: np.ndarray
        Current (n, n) boolean adjacency matrix (mutated in place).
    ring_mask: np.ndarray
        (n, n) boolean mask from ring_neighbors_mask - what a replaced
        agent's row/column is reset to before new random ties are drawn.
    neighbor_list: np.ndarray
        (n, n) padded neighbor list to keep in sync (mutated in place).
    degrees: np.ndarray
        (n,) neighbor counts to keep in sync (mutated in place).
    idx_replace: np.ndarray
        Indices of agents replaced this step (from population_turnover).
        No-op if empty.
    p_edge: float
        Probability of a random social-network edge for the new arrivals.
    n: int
        Number of agents.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Updated (adj, neighbor_list, degrees).
    """
    if idx_replace.size == 0:
        return adj, neighbor_list, degrees

    # rows whose neighbor list is about to change: the replaced agents
    # themselves, plus anyone who had a tie to them (that tie is dropped
    # below, so their row needs refreshing too)
    affected = set(idx_replace.tolist())
    for i in idx_replace:
        affected.update(np.where(adj[i])[0].tolist())

    # drop the old residents' social ties; keep only the site's ring
    # (spatial) neighbors, which don't depend on who lives there
    adj[idx_replace, :] = ring_mask[idx_replace, :]
    adj[:, idx_replace] = ring_mask[:, idx_replace]

    if p_edge > 0:
        # draw fresh social ties for the new, naive arrivals. Note: if two
        # replaced agents happen to both be replaced in the same step, the
        # tie between them is decided by two independent draws OR'd
        # together (slightly above p_edge) rather than one - a rare edge
        # case given idx_replace is normally a small fraction of n.
        new_edges = np.random.rand(idx_replace.size, n) < p_edge
        new_edges[np.arange(idx_replace.size), idx_replace] = False  # no self-loops
        adj[idx_replace, :] |= new_edges
        adj[:, idx_replace] |= new_edges.T
        for i in idx_replace:
            affected.update(np.where(adj[i])[0].tolist())

    affected = np.fromiter(affected, dtype=int, count=len(affected))
    degrees[affected] = adj[affected].sum(axis=1)
    neighbor_list[affected] = np.argsort(~adj[affected], axis=1, kind='stable')

    return adj, neighbor_list, degrees

# def social_learning(
#         b_vals: np.ndarray,
#         l_vals: np.ndarray,
#         n:int) -> np.ndarray:
#     """
#     Social learning and behavior update. Randomly choose an immediate neighbor
#     (left or right in the circular environment). Copy that neighbor's behavior
#     b perfectly.
#
#     Parameters
#     ----------
#     b_vals: np.ndarray
#         Array with behavior values for each offender.
#     l_vals: np.ndarray
#         Array of strategy values for each offender.
#     n: int
#         Number of offenders/sites.
#
#     Returns
#     -------
#     np.ndarray
#         Array with updated behavior values.
#     """
#     if not(b_vals.shape == l_vals.shape == (n, )):
#         raise ValueError("The shape of b_vals and l_vals do not match.")
#
#     # create a 2D array of neighbor values with shape (2, n)
#     # where the first row contains the left neighbors and
#     # the second row contains the right neighbors
#     neighbors_left = np.take(b_vals, np.arange(-1, n-1), mode='wrap')
#     neighbors_right = np.take(b_vals, np.arange(1, n+1), mode='wrap')
#     neighbors_both = np.append(neighbors_left.reshape(1, -1), neighbors_right.reshape(1, -1), axis=0)
#     # select one of two neighbors and copy their behavior (without an error)
#     indx = np.random.choice([0, 1], size=(n,))
#     copied_behav = neighbors_both[indx, np.arange(n)]
#     cndtn = np.where(l_vals == 1)
#     b_vals[cndtn] = copied_behav[cndtn]
#     return b_vals

def social_learning(
        b_vals: np.ndarray,
        l_vals: np.ndarray,
        n: int,
        neighbor_list: np.ndarray,
        degrees: np.ndarray) -> np.ndarray:
    """
    Social learning and behavior update. Each social learner copies the
    behavior of one uniformly random neighbor - by default its two ring
    (spatial) neighbors, plus any random social-network links from
    build_social_network if p_edge > 0.

    Parameters
    ----------
    b_vals: np.ndarray
        Array with behavior values for each offender.
    l_vals: np.ndarray
        Array of strategy values for each offender.
    n: int
        Number of offenders/sites.
    neighbor_list: np.ndarray
        (n, n) array from build_social_network.
    degrees: np.ndarray
        (n,) neighbor count per agent, from build_social_network.

    Returns
    -------
    np.ndarray
        Array with updated behavior values.
    """
    if not (b_vals.shape == l_vals.shape == (n,)):
        raise ValueError("The shape of b_vals and l_vals do not match.")

    # pick one uniformly random neighbor per agent - works for any degree,
    # so it covers ring-only (degree 2, p_edge=0) and ring+social alike
    chosen_pos = (np.random.rand(n) * degrees).astype(int)
    chosen = neighbor_list[np.arange(n), chosen_pos]
    copied_behav = b_vals[chosen]

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
    Asocial learning and behavior update. Sense the cue s at their current site.
    Acquire correct behavior (b=s) with probability p;
    otherwise, acquire a random behavior.

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
        n: int,
        neighbor_list: np.ndarray,
        degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    neighbor_list: np.ndarray
        (n, n) array from build_social_network. Dispersal targets are drawn
        from each agent's own neighbor list (ring + social network), so
        with p_edge=0 this reduces to the original left/right-only spread.
    degrees: np.ndarray
        (n,) neighbor count per agent, from build_social_network.

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

    # for each replicating agent, pick a uniformly random neighbor (ring +
    # social network) to disperse its behavior/strategy to
    chosen_pos = (np.random.rand(idx.size) * degrees[idx]).astype(int)
    targets = neighbor_list[idx, chosen_pos]

    # # for each replicating agent, pick the right (+1) or left (-1) neighbor
    # # with equal probability, all at once instead of one draw per agent
    # direction = np.where(np.random.rand(idx.size) < 0.5, 1, -1)
    # targets = (idx + direction) % n

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

def population_turnover(
        b_vals: np.ndarray,
        l_vals: np.ndarray,
        gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Population turnover step where a proportion of offenders are randomly replaced.

    Parameters
    ----------
    b_vals: np.ndarray
        Array with behavior values for each offender.
    l_vals: np.ndarray
        Array with strategy values for each offender.
    gamma: float
        Proportion of offenders to replace.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Updated (b_vals, l_vals) after population turnover, and the
        indices of the agents replaced this step (empty array if none).
        Pass idx_replace to rewire_social_network to keep the social
        network in sync with the new arrivals.
    """
    n = len(b_vals)
    n_replace = int(gamma * n)
    if n_replace == 0:
        return b_vals, l_vals, np.array([], dtype=int)

    # draw all indices to replace in one call instead of two calls per replacement
    idx_replace = np.random.choice(n, size=n_replace, replace=False)
    # assign new random behaviors and strategies to the selected indices
    b_vals[idx_replace] = np.random.randint(1, params.B + 1, size=n_replace)
    l_vals[idx_replace] = np.random.choice(
        params.learning_strategies, size=n_replace,
        p=[params.prop_asocial, 1 - params.prop_asocial])
    return b_vals, l_vals, idx_replace

def temporal_change(
        s_vals: np.ndarray,
        prob_change: float,
        n: int,
        s_max: int) -> np.ndarray:
    """
    Perform temporal change of cues with a given probability.
    IMPORTANT! Right now it only works for a single patch.
    I need to fix this function so if n_patches > 1, then the
    cues are changed in chunks (i.e., all sites in a patch have the same cue value).

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
        c_dict: dict,
        p_edge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    p_edge: float
        Probability of a random social-network edge between any pair of
        agents (0 disables the social network; see build_social_network).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Initialized arrays for cues, behaviors, learning strategies, costs,
        and the social network's (adj, ring_mask, neighbor_list, degrees).
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

    # b_vals = np.full_like(s_vals, fill_value=-1)  # behavior values for each offender (initially set to -1)
    b_vals = np.random.randint(1, s_max+1, size=(n,))  # random behavior values in the [1, S] range
    # assign learning strategies according to probability of being asocial or social learners
    l_vals = np.random.choice(l_options, size=(n,), p=[ratio_asocial, 1-ratio_asocial])
    # assign cost values for each strategy
    c_vals = assign_costs(l_vals=l_vals, c_dict=c_dict)
    # build the social network: ring lattice + optional random edges.
    # ring_mask and adj are kept (not just neighbor_list/degrees) so
    # population turnover can rewire individual agents later via
    # rewire_social_network.
    ring_mask = ring_neighbors_mask(n=n)
    adj, neighbor_list, degrees = build_social_network(n=n, p_edge=p_edge, ring_mask=ring_mask)
    return s_vals, b_vals, l_vals, c_vals, adj, ring_mask, neighbor_list, degrees

def run_single_iteration(args: tuple) -> tuple[list, list]:
    """
    Run one full iteration (env init + T timesteps) for a given parameter
    combination. Designed to be called by a worker process, so it takes a
    single tuple argument and re-seeds its own random state.

    Parameters
    ----------
    args: tuple
        (r, q, n_patches, gamma, seed) - the free-parameter values for this run and
        a seed for this worker's random state (or None to draw fresh entropy).

    Returns
    -------
    tuple[list, list]
        (ratio_social_learners, average_fitness) time series for this run.
    """
    r, q, n_patches, gamma, p_edge, seed = args
    # Each worker process needs its own random state. Passing `seed=None`
    # (the default when params.SEED is None) draws fresh OS entropy per
    # worker, so parallel runs don't accidentally share the same stream
    # (which can happen with fork-based multiprocessing otherwise).
    np.random.seed(seed)

    cues, behaviors, strategies, costs, adj, ring_mask, neighbor_list, degrees = env_init(
        n=params.N, n_patches=n_patches, s_max=params.S,
        l_options=params.learning_strategies,
        ratio_asocial=params.prop_asocial, c_dict=params.c,
        p_edge=p_edge)

    ratio_social_learners = []
    average_fitness = []

    for _ in range(params.T):
        # STEP 1&2: LEARNING & BEHAVIORAL DEPLOYMENT
        # social learners
        behaviors = social_learning(
            b_vals=behaviors, l_vals=strategies, n=params.N,
            neighbor_list=neighbor_list, degrees=degrees)
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
            w_vals=w, n=params.N,
            neighbor_list=neighbor_list, degrees=degrees)

        # STEP 5: INNOVATION
        strategies = innovation(
            l_vals=strategies,
            n=params.N,
            innovation_rate=params.mu)

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

        # STEP 8: POPULATION TURNOVER
        behaviors, strategies, idx_replace = population_turnover(
            b_vals=behaviors, l_vals=strategies,
            gamma=gamma)
        # keep the social network in sync: new arrivals keep the site's
        # ring neighbors but get fresh social ties instead of inheriting
        # the previous resident's
        adj, neighbor_list, degrees = rewire_social_network(
            adj=adj, ring_mask=ring_mask,
            neighbor_list=neighbor_list, degrees=degrees,
            idx_replace=idx_replace, p_edge=p_edge, n=params.N)

    return ratio_social_learners, average_fitness

def run_simulation() -> str:
    """
    Run the full simulation sweep over every combination of free
    parameters, writing results incrementally to results.csv.gz.

    Returns
    -------
    str
        Path to the timestamped output folder this run wrote to (e.g.
        "output/20260813_110813_results") - useful for chaining straight
        into visualizations.generate_visualizations() without having to
        pick the folder again (see run_all.py).
    """
    # n_workers = min(int(cpu_count()/2), params.n_iterations)
    n_workers = params.n_cores

    # catch duplicate (or near-duplicate, after rounding) values within any
    # single free-parameter list up front - e.g. two equal entries in
    # q_vals from a copy-paste mistake. Left unchecked, product() below
    # would simulate and append that exact combination twice, and the
    # duplication would only surface much later as an opaque pandas error
    # in visualizations.py.
    for name, vals in [("r_vals", params.r_vals), ("q_vals", params.q_vals),
                        ("n_patches_vals", params.n_patches_vals),
                        ("gamma_vals", params.gamma_vals), ("p_vals", params.p_vals)]:
        vals = np.asarray(vals, dtype=float)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if np.isclose(vals[i], vals[j]):
                    raise ValueError(
                        f"{name} in params.py has duplicate (or near-duplicate) "
                        f"values {vals[i]} and {vals[j]}. Each value should appear "
                        f"once - remove the duplicate before running the simulation."
                    )

    # create output folder up front so results can be written incrementally,
    # combo by combo, instead of holding everything in memory until the end.
    # honors params.output_dir if set, otherwise falls back to '../output/'
    output_dir = getattr(params, 'output_dir', None) or '../output/'
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(output_dir, f"{current_time}_results")
    os.makedirs(save_dir, exist_ok=True)

    # copy the params.py file
    shutil.copy("params.py", os.path.join(save_dir, "params.py"))

    csv_path = os.path.join(save_dir, "results.csv.gz")
    fieldnames = ["r", "q", "n_patches", "p_edge", "gamma", "iteration", "t", "avg_fitness", "fraction_social"]

    # iterate over all combinations of free parameters
    for combo_idx, (r, q, n_patches, gamma, p_edge) in tqdm(
            enumerate(
                product(
                    params.r_vals, params.q_vals, params.n_patches_vals, params.gamma_vals, params.p_vals)),
            desc="Running simulations",
            total=len(params.r_vals) * len(params.q_vals) * len(params.n_patches_vals) * len(params.gamma_vals) * len(params.p_vals)):

        # per-iteration seeds: if params.SEED is set, derive a distinct,
        # reproducible seed for each (combo, iteration) pair; if it's None,
        # pass None through so each worker draws its own fresh entropy
        if params.SEED is not None:
            iter_seeds = [params.SEED + combo_idx * params.n_iterations + i
                          for i in range(params.n_iterations)]
        else:
            iter_seeds = [None] * params.n_iterations

        tasks = [(r, q, n_patches, gamma, p_edge, seed) for seed in iter_seeds]

        ratio_social_learners_runs = []
        average_fitness_runs = []
        with Pool(processes=n_workers) as pool:
            for ratio_social_learners, average_fitness in pool.imap(run_single_iteration, tasks):
                ratio_social_learners_runs.append(ratio_social_learners)
                average_fitness_runs.append(average_fitness)

        # build this combo's rows as a DataFrame in one vectorized shot
        # (much faster than writer.writerow() per row - see benchmark notes)
        # rather than looping in Python, then append it to the CSV. compresslevel
        # is set low (gzip default is 9, which is ~3x slower for a small size
        # gain) since write speed matters more than a few % of file size here.
        n_iter = len(ratio_social_learners_runs)
        combo_df = pd.DataFrame({
            "r": r, "q": q, "n_patches": n_patches, "p_edge": p_edge, "gamma": gamma,
            "iteration": np.repeat(np.arange(n_iter), params.T),
            "t": np.tile(np.arange(1, params.T + 1), n_iter),
            "avg_fitness": np.concatenate(average_fitness_runs),
            "fraction_social": np.concatenate(ratio_social_learners_runs),
        })
        combo_df.to_csv(
            csv_path, index=False, columns=fieldnames,
            mode="w" if combo_idx == 0 else "a", header=(combo_idx == 0),
            compression={"method": "gzip", "compresslevel": 1})

    print(f"Done! Results saved in {save_dir}.")
    return save_dir

if '__main__' == __name__:
    run_simulation()