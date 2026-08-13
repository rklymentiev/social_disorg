import numpy as np

# FIXED PARAMETERS
n_cores = 5  # number of cores to use for parallel processing
SEED = None
S = 1000  # number of cues
B = S  # number of behavioral responses
N = 500  # number of sites
T = 2000
learning_strategies = [0, 1]  # 0: asocial, 1: social
c = {0: .1, 1: 0}  # cost values
prob_success_asocial = 1  # probability of asocial learning being successful
mu = 0.0008  # innovation rate
h = 2  # penalty exponent
prop_asocial = 1  # proportion of asocial learners in the population at the initialization
n_iterations = 100  # number of iterations to run for each unique combination of parameters

output_dir = '../02_output/'  # directory to save output files

img_format = 'png'  # format of plots to save, e.g., 'pdf', 'jpg', 'svg' (used in visualizations.py)
n_last = 250  # number of time steps to use for visualizations (used in visualizations.py)

# FREE PARAMETERS
# all parameters have to be specified in a list or np.ndarray format, even if they only have one value,
# for example r_vals = [0], r_vals = [0, 0.1], or r_vals = np.linspace(0, 0.4, num=5)
# script will run for all combinations of parameter set
# for simplicity, use the range of values for one of the parameters while keeping the others fixed to a single value.

# proportion of offenders to shuffle (social disorganization)
# r_vals = [0]
r_vals = np.linspace(0, 0.2, num=25)

# probability to change cues' value (temporal variation)
# q_vals = np.linspace(0, 0.1, num=25)
q_vals = [0]

# number of patches (spatial heterogeneity)
n_patches_vals = [10]
# n_patches_vals = np.linspace(1, 120, num=25).astype(int)

# population turnover rate
gamma_vals = [0]  # proportion of population to replace at each time step

# probability of a random social-network edge between any pair of agents
# (social network), in addition to each agent's two spatial ring neighbors.
# p=0 disables the social network entirely (ring neighbors only, the
# original behavior); higher values add more long-range social ties.
# p_vals = np.linspace(0, 0.01, num=10)
p_vals = [0.1]