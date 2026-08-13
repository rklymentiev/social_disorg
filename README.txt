"Social Disorganization" simulation script

File structure:
├── ../output/                           # Directory for simulation outputs (generated automatically)
│   └── 20260813_110813_results/         # Example output directory (timestamped)
│       ├── results.csv.gz               # Gzip-compressed CSV with results for every run (see below)
│       └── params.py                    # Copy of the parameters used for this simulation run
├── params.py                            # Simulation parameters
├── README.txt                           # Project overview and usage instructions
├── requirements.txt                     # Python dependencies
├── social_disorg.py                     # Main simulation script
├── visualizations.py                    # Script for generating visualizations from simulation results
└── run_pipeline.py                      # Convenience script: runs social_disorg.py then visualizations.py in one go


How to use the code:

(1) Create a new Python virtual environment (optional but recommended):
$ python -m venv socialdisorg

(2) Activate the virtual environment:
$ source socialdisorg/bin/activate  # On Linux/Mac
$ socialdisorg\Scripts\activate     # On Windows

(3) Install the required dependencies:
$ pip install -r requirements.txt

(4) Set up the parameters for the simulation in `params.py` (see below for details).

(5) Run the simulation and generate visualizations. Either:

    - In one step (recommended):
      $ python run_pipeline.py
      This runs social_disorg.py, then automatically passes the results folder it just wrote
      straight into visualizations.py - no need to select a folder by hand.

    - Or as two separate steps:
      $ python social_disorg.py
      $ python visualizations.py
      Running visualizations.py on its own opens a folder-picker dialog asking you to select
      the results folder to visualize. You can skip the dialog by passing the folder directly:
      $ python visualizations.py path/to/results_folder

Parameters of the simulation can be adjusted in the `params.py` file.
Currently, there are five free parameters that can be varied to explore different scenarios:
(1) `r_vals`: The proportion of offenders to shuffle each time step (residential instability)
(2) `q_vals`: The probability of cues changing their value each time step (temporal variation)
(3) `n_patches_vals`: The number of patches of cues (spatial heterogeneity)
(4) `gamma_vals`: The proportion of the population replaced by new, naive individuals each
    time step (population turnover)
(5) `p_vals`: The probability of a random social-network edge between any pair of agents, in
    addition to each agent's two spatial ring neighbors (social network). p=0 disables the
    social network entirely, leaving only ring neighbors.
For simplicity, use a range of values for exactly one of these parameters while keeping the
others fixed to a single value - social_disorg.py will raise an error up front if it detects
a duplicate value within one of these lists, and visualizations.py will raise an error if it
detects more than one parameter varying at once, since both scripts assume only one parameter
sweeps at a time.

Other parameters worth knowing about:
- `output_dir`: where simulation output folders are created (default '../02_output/').
- `n_cores`: number of CPU cores used to parallelize the `n_iterations` runs of each parameter
  combination.
- `n_iterations`: number of independent iterations run for each unique combination of
  parameters.
See `params.py` for the full list of parameters, their defaults, and further details.

The script `social_disorg.py` will generate output in the directory set by `params.output_dir`,
with a timestamped subdirectory for each run. Each run's subdirectory contains a copy of the
parameters used (`params.py`) and a single `results.csv.gz` file: a gzip-compressed CSV with
one row per (parameter combination, iteration, time step), with the following columns:

    r | q | n_patches | p_edge | gamma | iteration | t | avg_fitness | fraction_social

`avg_fitness` and `fraction_social` are the average fitness and the fraction of social learners
in the population at time step `t`, for that iteration of that parameter combination.

The script `visualizations.py` can be used to create visualizations from the output data. It
reads `results.csv.gz` and the accompanying `params.py`, identifies which single parameter was
varied, and generates 4 plots in the same results folder:
(1) Average fitness value over time (averaged across `params.n_iterations` runs)
(2) Fraction of social learners over time (averaged across `params.n_iterations` runs)
(3) The average of the average fitness for the last `params.n_last` time steps (averaged across
    `params.n_iterations` runs)
(4) The average of the fraction of social learners for the last `params.n_last` time steps
    (averaged across `params.n_iterations` runs)