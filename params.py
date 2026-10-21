import numpy as np

# Defaults reproduce the baseline of Brantingham's Mathematica model
# (LearningDisorgBasic_loops.nb); the notebook's symbol is given in brackets.

# FIXED PARAMETERS
n_cores = 5  # number of cores to use for parallel processing
SEED = None  # base seed for reproducible runs (None = fresh randomness)
S = 1000  # cues and behaviors are real numbers in [0, S]
N = 50  # number of sites/offenders on the ring [n]
T = 500  # number of time steps [T]
beta = 0.01  # strength of selection on behavioral mismatch [β]
c = 20  # cost of asocial learning, paid inside the exponent of the fitness function [c]
sd_asocial = 20  # SD of asocial learners' perceptual error, b ~ Normal(s, sd_asocial)
mu = 0.02  # innovation rate: probability of switching strategy each step [μ2]
p_social_init = 0.08  # probability that an offender starts as a social learner [μ1]
n_iterations = 25  # number of runs for each unique combination of parameters [nLoops]

# cue landscape
patch_sizes = 'equal'  # 'equal': blocks of ceil(N / n_patches) sites; 'random': random patch sizes
landscape_seed = None  # int = same landscape in every run (the notebook uses RandomSeeding->142); None = new landscape per run
environment_seed = None  # int = same sequence of environmental changes in every run; None = new sequence per run

output_dir = '../02_output/'  # directory to save output files

img_format = 'png'  # format of plots to save, e.g., 'pdf', 'jpg', 'svg' (used in visualizations.py)
n_last = 400  # number of final time steps averaged in the summary plots (the notebook uses the last 400)

# FREE PARAMETERS
# all parameters have to be specified in a list or np.ndarray format, even if they only have one value,
# for example r_vals = [0], r_vals = [0, 0.1], or r_vals = np.linspace(0, 0.4, num=5)
# script will run for all combinations of parameter set
# for simplicity, use the range of values for one of the parameters while keeping the others fixed to a single value.

# share of offenders relocating each step, r = 2 * (number of swapping pairs) / N (residential instability).
# The notebook's 0..25 swapping pairs with N = 50 correspond to r = 0..1.
r_vals = [0]
# r_vals = np.linspace(0, 1, num=11)

# rate of environmental change: changes arrive as a Poisson process with rate q per step (temporal variation) [q]
q_vals = [0]
# q_vals = [1/10000, 1/200, 1/100, 1/50, 1/25, 1/10, 1/5, 1/4, 1/3, 1/2]

# number of patches (spatial heterogeneity) [np]
n_patches_vals = [1]
# n_patches_vals = [1, 2, 5, 10, 50]

# probability that each social tie is rewired at a given time step (social network instability) [ρ / rwProb]
rho_vals = [0]
# rho_vals = np.linspace(0, 1, num=11)

