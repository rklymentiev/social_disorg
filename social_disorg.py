"""
Social vs. asocial learning among offenders ("Social Disorganization" model).

This is a Python port of Brantingham's Mathematica model
(LearningDisorgBasic_loops.nb, as presented at Penn State, 17-09-2026).
Each time step follows the notebook's order:

    1. replicator update of learning strategies (fitness-proportional
       resampling using w/W from the previous step)
    2. innovation: each agent flips its strategy with probability mu
    3. environmental disturbances, all applied BEFORE learning:
       temporal change of cues (q), residential mobility (r) and
       social-network rewiring (rho)
    4. learning: asocial learners perceive their site's cue with Gaussian
       error; social learners copy the behavior of a random member of
       {their social neighbors + themselves}. Agents update one after another
       in site order, so a social learner may copy a behavior that was
       already updated earlier in the same step (as in the notebook's loop)
    5. fitness: w = exp(-beta * (|b - s| + c * [agent is an asocial learner]))

The social network is the ring of spatial neighbors (the notebook's
WattsStrogatzGraphDistribution[n, 0, 1]); it only departs from space when
rho > 0 rewires it.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from datetime import datetime
from multiprocessing import Pool
import params
import os
import shutil


# ---------------------------------------------------------------------------
# fitness and strategy dynamics
# ---------------------------------------------------------------------------

def fitness_function(
        b_vals: np.ndarray,
        s_vals: np.ndarray,
        l_vals: np.ndarray,
        beta: float,
        c: float) -> np.ndarray:
    """
    Utility of each offender, w = exp(-beta * (|b - s| + c * 1(l = 0))).

    Matches w[b_, s_, l_] := Exp[-beta (Abs[b - s] + (1 - l) c)] in the
    notebook: the cost of asocial learning sits inside the exponent, so it
    scales fitness multiplicatively. Distance is the plain absolute
    difference (cues and behaviors are reals in [0, S], not a circle).

    Parameters
    ----------
    b_vals: np.ndarray
        Behavior of each offender.
    s_vals: np.ndarray
        Cue at each offender's site.
    l_vals: np.ndarray
        Learning strategy of each offender (0: asocial, 1: social).
    beta: float
        Strength of selection on behavioral mismatch.
    c: float
        Cost of asocial learning (in the same units as |b - s|).

    Returns
    -------
    np.ndarray
        Fitness values, all in (0, 1].
    """
    if not (b_vals.shape == s_vals.shape == l_vals.shape):
        raise ValueError("The shape of b_vals, s_vals, and l_vals do not match.")
    return np.exp(-beta * (np.abs(b_vals - s_vals) + (1 - l_vals) * c))


def replicator_update(
        l_vals: np.ndarray,
        w_vals: np.ndarray,
        rng: np.random.Generator) -> np.ndarray:
    """
    Fitness-based strategy copying (discrete replicator dynamics).

    Mirrors the notebook exactly: every site i is put into a pool
    Ceiling[w_i / W] times (W = mean fitness), n entries are drawn from the
    pool without replacement, the draws are sorted by the site they came
    from, and the k-th sorted draw gives site k its new strategy. In
    expectation this is f(l)_{t+1} = (w/W) f(l)_t; sorting by origin keeps
    copied strategies close to where they came from.

    Parameters
    ----------
    l_vals: np.ndarray
        Current learning strategies.
    w_vals: np.ndarray
        Fitness values from the previous time step.
    rng: np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        New learning strategies.
    """
    n = len(l_vals)
    copies = np.ceil(w_vals / w_vals.mean()).astype(int)
    pool = np.repeat(np.arange(n), copies)
    picked = rng.choice(pool.size, size=n, replace=False)
    origins = np.sort(pool[picked])
    return l_vals[origins]


def innovation(
        l_vals: np.ndarray,
        innovation_rate: float,
        rng: np.random.Generator) -> np.ndarray:
    """
    Each offender switches strategy (asocial <-> social) with probability
    innovation_rate (mu2 in the notebook).
    """
    flip = rng.random(len(l_vals)) < innovation_rate
    l_vals = l_vals.copy()
    l_vals[flip] ^= 1
    return l_vals


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def make_landscape(
        n: int,
        n_patches: int,
        s_max: float,
        rng: np.random.Generator,
        patch_sizes: str = "equal") -> tuple[np.ndarray, np.ndarray]:
    """
    Cue value at each site of the ring, organized in patches that share one
    value drawn uniformly from [0, s_max].

    Parameters
    ----------
    n: int
        Number of sites.
    n_patches: int
        Number of patches.
    s_max: float
        Upper bound of cue values.
    rng: np.random.Generator
        Random number generator.
    patch_sizes: str
        "equal": the notebook's Partition[..., UpTo[Ceiling[n/n_patches]]],
            i.e. consecutive blocks of Ceiling[n/n_patches] sites. When
            n_patches doesn't divide n this can give fewer patches than
            asked for (e.g. n=50, n_patches=15 -> 13 blocks), exactly as in
            Mathematica.
        "random": the notebook's random-size patches (used for the "40
            random landscapes" slide): n_patches change points are drawn
            from 1..n, which splits the ring into n_patches + 1 segments
            (the first one can be empty).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (s_vals, patch_id): cue per site and the patch each site belongs to.
    """
    if patch_sizes == "equal":
        size = int(np.ceil(n / n_patches))
        patch_id = np.arange(n) // size
    elif patch_sizes == "random":
        change = np.sort(rng.choice(np.arange(1, n + 1), size=n_patches, replace=False))
        change = np.append(change, n + 1)
        sizes = np.concatenate([[change[0] - 1], np.diff(change)])
        patch_id = np.repeat(np.arange(sizes.size), sizes)
    else:
        raise ValueError(f"patch_sizes must be 'equal' or 'random', got {patch_sizes!r}.")
    values = rng.uniform(0, s_max, size=patch_id.max() + 1)
    return values[patch_id], patch_id


def temporal_change(
        s_vals: np.ndarray,
        patch_id: np.ndarray,
        q: float,
        s_max: float,
        rng: np.random.Generator) -> np.ndarray:
    """
    Environmental change. Changes arrive as a Poisson process with rate q
    per time step; in the notebook the event times are truncated to whole
    steps, so a step sees a change with probability 1 - exp(-q). When a
    change happens every patch draws a new cue value (with a single patch
    this is the notebook's uniform jump of the whole environment).
    """
    if q > 0 and rng.random() < -np.expm1(-q):
        values = rng.uniform(0, s_max, size=patch_id.max() + 1)
        return values[patch_id]
    return s_vals


def residential_instability(
        b_vals: np.ndarray,
        l_vals: np.ndarray,
        r: float,
        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Residential mobility: round(r * n / 2) disjoint pairs of offenders swap
    places, taking their behavior and learning strategy with them (cues stay
    with the sites). Like the notebook, the 2 * n_pairs offenders are drawn
    without replacement, so nobody moves twice in one step. r = 2 * pairs / n,
    so r = 1 means everyone moves (25 pairs for n = 50).
    """
    n = len(b_vals)
    n_pairs = min(int(round(r * n / 2)), n // 2)
    if n_pairs == 0:
        return b_vals, l_vals
    movers = rng.choice(n, size=2 * n_pairs, replace=False)
    first, second = movers[0::2], movers[1::2]
    b_vals, l_vals = b_vals.copy(), l_vals.copy()
    b_vals[first], b_vals[second] = b_vals[second], b_vals[first].copy()
    l_vals[first], l_vals[second] = l_vals[second], l_vals[first].copy()
    return b_vals, l_vals


# ---------------------------------------------------------------------------
# social network
# ---------------------------------------------------------------------------

def ring_edges(n: int) -> np.ndarray:
    """
    Edge list of the ring: site i is tied to site i+1 (and the last site to
    the first), in the same order as Mathematica's EdgeList of the ring.
    """
    i = np.arange(n)
    return np.stack([i, (i + 1) % n], axis=1)


def rewire_edges(
        edges: np.ndarray,
        n: int,
        rho: float,
        rng: np.random.Generator) -> np.ndarray:
    """
    Social-network instability, following IGraph/M's IGRewireEdges[g, rho],
    i.e. igraph_rewire_edges without loops or multi-edges: EACH END of each
    edge is moved with probability rho, in two passes. Pass 1 goes through
    the edges in order and may move the smaller-index end (igraph stores an
    undirected edge as (larger, smaller)); pass 2 goes through the edges
    ordered by that end and may move the other one. A moving end goes to a
    vertex drawn uniformly among all vertices except the end that stays,
    redrawn while it is already a neighbor of the staying end (drawing the
    old vertex leaves the edge as it was). The notebook rewires the original
    ring afresh every step, so rewiring does not accumulate over time.
    """
    if rho <= 0 or len(edges) == 0:
        return edges
    # igraph's internal orientation: (from, to) = (larger, smaller)
    edges = np.sort(edges, axis=1)[:, ::-1].copy()
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    def move_end(e, moving):
        stay = edges[e, 1 - moving]
        old = edges[e, moving]
        while True:
            pot = rng.integers(n - 1)
            if pot >= stay:
                pot += 1  # uniform over every vertex except `stay`
            if pot == old or pot not in neighbors[stay]:
                break
        if pot != old:
            neighbors[old].discard(stay)
            neighbors[stay].discard(old)
            neighbors[pot].add(stay)
            neighbors[stay].add(pot)
            edges[e, moving] = pot

    for e in np.flatnonzero(rng.random(len(edges)) < rho):  # pass 1: "to" ends
        move_end(e, 1)
    order = np.argsort(edges[:, 1], kind="stable")
    for e in order[rng.random(len(edges)) < rho]:  # pass 2: "from" ends
        move_end(e, 0)
    return edges


def edges_to_csr(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compressed neighbor lists: neighbors of i are indices[indptr[i]:indptr[i+1]].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (indptr, indices, degrees)
    """
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    degrees = np.bincount(src, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(degrees)])
    return indptr, dst[order], degrees


# ---------------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------------

def learning(
        b_vals: np.ndarray,
        l_vals: np.ndarray,
        s_vals: np.ndarray,
        network: tuple[np.ndarray, np.ndarray, np.ndarray],
        sd_asocial: float,
        rng: np.random.Generator) -> np.ndarray:
    """
    Asocial and social learning.

    Asocial learners (l=0) perceive the cue at their own site with error:
    b ~ Normal(s, sd_asocial). Social learners (l=1) copy the behavior of
    one member of {their social neighbors + themselves}, chosen uniformly.

    The notebook updates agents one at a time in site order (For[j = 1..n]),
    so a social learner who copies an agent with a lower index gets that
    agent's behavior from THIS step, while copying a higher index (or
    itself) gets last step's behavior. This vectorized version reproduces
    that exactly by following chains of such copies back to an agent whose
    value is already known (pointer jumping).

    Parameters
    ----------
    b_vals: np.ndarray
        Behaviors from the previous step.
    l_vals: np.ndarray
        Learning strategies.
    s_vals: np.ndarray
        Cue at each site.
    network: tuple[np.ndarray, np.ndarray, np.ndarray]
        (indptr, indices, degrees) from edges_to_csr.
    sd_asocial: float
        Standard deviation of the asocial learners' perceptual error.
    rng: np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        New behaviors.
    """
    indptr, indices, degrees = network
    n = len(b_vals)
    idx = np.arange(n)

    # each agent's pick among (neighbors + self); drawn for everyone to keep
    # the code simple, only used for social learners
    k = (rng.random(n) * (degrees + 1)).astype(int)
    padded = np.append(indices, 0)  # k == degree means "self"; keeps indexing in bounds
    model = np.where(k == degrees, idx, padded[indptr[:-1] + k])

    asocial = l_vals == 0
    value = np.empty(n)
    value[asocial] = rng.normal(s_vals[asocial], sd_asocial)

    # social learners copying an earlier agent inherit that agent's new
    # value; everyone else is resolved directly
    chained = ~asocial & (model < idx)
    direct = ~asocial & ~chained
    value[direct] = b_vals[model[direct]]
    parent = np.where(chained, model, idx)
    while True:
        jumped = parent[parent]
        if np.array_equal(jumped, parent):
            break
        parent = jumped
    return value[parent]


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def simulate(
        T: int,
        n: int,
        s_max: float,
        beta: float,
        c: float,
        sd_asocial: float,
        mu: float,
        p_social_init: float,
        r: float,
        q: float,
        n_patches: int,
        rho: float,
        rng: np.random.Generator,
        patch_sizes: str = "equal",
        landscape_rng: np.random.Generator | None = None,
        environment_rng: np.random.Generator | None = None,
        s_init: np.ndarray | None = None,
        l_init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    One run of the model.

    Parameters
    ----------
    T, n, s_max, beta, c, sd_asocial, mu, p_social_init: model constants
        (see params.py).
    r: float
        Share of offenders relocating each step (2 * pairs / n).
    q: float
        Rate of environmental change.
    n_patches: int
        Number of cue patches.
    rho: float
        Probability that each social tie is rewired in a given step.
    rng: np.random.Generator
        Main random number generator.
    patch_sizes: str
        "equal" or "random" (see make_landscape).
    landscape_rng: np.random.Generator, optional
        Separate generator for the cue landscape. Passing a generator built
        from the same seed in every run keeps the landscape fixed across
        runs, like the notebook's BlockRandom[..., RandomSeeding->142].
    environment_rng: np.random.Generator, optional
        Separate generator for temporal changes (the notebook reuses one
        seeded change sequence for all runs).
    s_init, l_init: np.ndarray, optional
        Use this cue landscape / initial strategies instead of drawing them.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (fraction_social, average_fitness), each of length T, recorded at the
        end of every step.
    """
    landscape_rng = landscape_rng or rng
    environment_rng = environment_rng or rng

    if s_init is None:
        s_vals, patch_id = make_landscape(n, n_patches, s_max, landscape_rng, patch_sizes)
    else:
        s_vals = np.asarray(s_init, dtype=float).copy()
        _, patch_id = np.unique(s_vals, return_inverse=True)
    b_vals = rng.uniform(0, s_max, size=n)
    if l_init is None:
        l_vals = (rng.random(n) < p_social_init).astype(int)
    else:
        l_vals = np.asarray(l_init, dtype=int).copy()
    w_vals = fitness_function(b_vals, s_vals, l_vals, beta, c)

    base_edges = ring_edges(n)
    base_network = edges_to_csr(base_edges, n)

    fraction_social = np.empty(T)
    average_fitness = np.empty(T)
    for t in range(T):
        # 1. replicator update with last step's fitness, 2. innovation
        l_vals = replicator_update(l_vals, w_vals, rng)
        l_vals = innovation(l_vals, mu, rng)

        # 3. disturbances before learning
        s_vals = temporal_change(s_vals, patch_id, q, s_max, environment_rng)
        b_vals, l_vals = residential_instability(b_vals, l_vals, r, rng)
        if rho > 0:
            network = edges_to_csr(rewire_edges(base_edges, n, rho, rng), n)
        else:
            network = base_network

        # 4. learning, 5. fitness
        b_vals = learning(b_vals, l_vals, s_vals, network, sd_asocial, rng)
        w_vals = fitness_function(b_vals, s_vals, l_vals, beta, c)

        fraction_social[t] = l_vals.mean()
        average_fitness[t] = w_vals.mean()

    return fraction_social, average_fitness


def run_single_iteration(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one iteration for a given parameter combination with the constants
    in params.py. Takes a single tuple so it can be used with Pool.imap.

    Parameters
    ----------
    args: tuple
        (r, q, n_patches, rho, seed); seed=None draws fresh
        entropy for this run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (fraction_social, average_fitness) time series for this run.
    """
    r, q, n_patches, rho, seed = args
    landscape_seed = getattr(params, "landscape_seed", None)
    environment_seed = getattr(params, "environment_seed", None)
    return simulate(
        T=params.T, n=params.N, s_max=params.S, beta=params.beta, c=params.c,
        sd_asocial=params.sd_asocial, mu=params.mu, p_social_init=params.p_social_init,
        r=r, q=q, n_patches=int(n_patches), rho=rho,
        rng=np.random.default_rng(seed),
        patch_sizes=getattr(params, "patch_sizes", "equal"),
        landscape_rng=None if landscape_seed is None else np.random.default_rng(landscape_seed),
        environment_rng=None if environment_seed is None else np.random.default_rng(environment_seed),
    )


FREE_PARAMETERS = [  # (name in params.py, CSV column)
    ("r_vals", "r"), ("q_vals", "q"), ("n_patches_vals", "n_patches"),
    ("rho_vals", "rho"),
]


def run_simulation() -> str:
    """
    Run the full simulation sweep over every combination of free
    parameters, writing results incrementally to results.csv.gz.

    Returns
    -------
    str
        Path to the timestamped output folder this run wrote to - useful for
        chaining straight into visualizations.generate_visualizations().
    """
    free_values = {name: list(getattr(params, name, [0])) for name, _ in FREE_PARAMETERS}

    # catch duplicate (or near-duplicate) values within a free-parameter list
    # up front; product() would otherwise simulate that combination twice
    for name, vals in free_values.items():
        vals = np.asarray(vals, dtype=float)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if np.isclose(vals[i], vals[j]):
                    raise ValueError(
                        f"{name} in params.py has duplicate (or near-duplicate) "
                        f"values {vals[i]} and {vals[j]}. Each value should appear "
                        f"once - remove the duplicate before running the simulation."
                    )

    output_dir = getattr(params, "output_dir", None) or "../output/"
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"{current_time}_results")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy("params.py", os.path.join(save_dir, "params.py"))

    csv_path = os.path.join(save_dir, "results.csv.gz")
    fieldnames = [col for _, col in FREE_PARAMETERS] + ["iteration", "t", "avg_fitness", "fraction_social"]

    combos = list(product(*free_values.values()))
    with Pool(processes=params.n_cores) as pool:
        for combo_idx, (r, q, n_patches, rho) in enumerate(
                tqdm(combos, desc="Running simulations")):
            # distinct, reproducible seed per (combo, iteration) if SEED is set
            if params.SEED is not None:
                seeds = [params.SEED + combo_idx * params.n_iterations + i
                         for i in range(params.n_iterations)]
            else:
                seeds = [None] * params.n_iterations
            tasks = [(r, q, n_patches, rho, seed) for seed in seeds]
            runs = list(pool.imap(run_single_iteration, tasks))

            n_iter = len(runs)
            combo_df = pd.DataFrame({
                "r": r, "q": q, "n_patches": n_patches, "rho": rho,
                "iteration": np.repeat(np.arange(n_iter), params.T),
                "t": np.tile(np.arange(1, params.T + 1), n_iter),
                "avg_fitness": np.concatenate([fit for _, fit in runs]),
                "fraction_social": np.concatenate([soc for soc, _ in runs]),
            })
            combo_df.to_csv(
                csv_path, index=False, columns=fieldnames,
                mode="w" if combo_idx == 0 else "a", header=(combo_idx == 0),
                compression={"method": "gzip", "compresslevel": 1})

    print(f"Done! Results saved in {save_dir}.")
    return save_dir


if __name__ == "__main__":
    run_simulation()
