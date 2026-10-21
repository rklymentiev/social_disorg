"Social Disorganization" simulation script

File structure:
├── ../02_output/                        # Directory for simulation outputs (generated automatically;
│   │                                     # path is set by params.output_dir, default '../02_output/')
│   └── 20260813_110813_results/         # Example output directory (timestamped)
│       ├── results.csv.gz               # Gzip-compressed CSV with results for every run (see below)
│       └── params.py                    # Copy of the parameters used for this simulation run
├── params.py                            # Simulation parameters
├── README.txt                           # Project overview and usage instructions
├── requirements.txt                     # Python dependencies
├── social_disorg.py                     # Main simulation script
├── visualizations.py                    # Script for generating visualizations from simulation results
├── run_pipeline.py                      # Convenience script: runs social_disorg.py then visualizations.py in one go
└── gui.py                               # Optional graphical interface (live simulation + parameter sweeps)


The model:

N offenders live on the sites of a ring. Each site has a cue s in [0, S]; sites are grouped into
patches that share a cue value. Each offender has a behavior b in [0, S] and a learning strategy
l (0: asocial, 1: social). Fitness is

    w = exp(-beta * (|b - s| + c * 1(l = 0)))

so behaviors that match the local cue pay off and asocial learning carries a cost c.
Each time step:
(1) Replicator update: strategies are resampled in proportion to w / W (W = mean fitness),
    i.e. f(l)_{t+1} = (w / W) f(l)_t.
(2) Innovation: each offender switches strategy with probability mu.
(3) Disturbances, all before learning: environmental change (q), residential mobility (r)
    and rewiring of social ties (rho).
(4) Learning: asocial learners perceive their site's cue with error, b ~ Normal(s, sd_asocial);
    social learners copy the behavior of a random member of {their social neighbors + themselves}.
    The social network is the ring of spatial neighbors; it departs from space only when rho > 0.
(5) Fitness is recomputed and the share of social learners and the mean fitness are recorded.


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

    - Or with the graphical interface:
      $ python gui.py
      See "Graphical interface" below.

Parameters of the simulation can be adjusted in the `params.py` file. The defaults are the
notebook's baseline (N = 50, T = 500, beta = 0.01, c = 20, sd_asocial = 20, mu = 0.02,
8% social learners at the start, 25 runs per parameter combination).

There are four free parameters:
(1) `r_vals`: The share of offenders relocating each time step, r = 2 * (number of swapping
    pairs) / N (residential instability). Pairs swap places, taking their behavior and strategy
    with them.
(2) `q_vals`: The rate of environmental change (temporal variation). Changes arrive as a Poisson
    process with rate q per time step; at each change every patch gets a new cue value.
(3) `n_patches_vals`: The number of cue patches (spatial heterogeneity). With
    patch_sizes = 'equal' the ring is cut into blocks of ceil(N / n_patches) sites, as in the
    notebook; with 'random' the patches have random sizes.
(4) `rho_vals`: The probability that each social tie is rewired in a time step (social network
    instability). The ring is rewired afresh every step; rho = 0 keeps the ring.
For simplicity, use a range of values for exactly one of these parameters while keeping the
others fixed to a single value - social_disorg.py will raise an error up front if it detects
a duplicate value within one of these lists, and visualizations.py will raise an error if it
detects more than one parameter varying at once, since both scripts assume only one parameter
sweeps at a time.

Other parameters worth knowing about:
- `landscape_seed`: an integer keeps the same cue landscape in every run (the notebook fixes it
  with RandomSeeding -> 142); None draws a new landscape for every run.
- `environment_seed`: an integer reuses the same sequence of environmental changes in every run
  (as the notebook does); None draws a new sequence for every run.
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

    r | q | n_patches | rho | iteration | t | avg_fitness | fraction_social

`avg_fitness` and `fraction_social` are the average fitness and the fraction of social learners
in the population at the end of time step `t`, for that iteration of that parameter combination.

The script `visualizations.py` can be used to create visualizations from the output data. It
reads `results.csv.gz` and the accompanying `params.py`, identifies which single parameter was
varied, and generates 4 plots in the same results folder:
(1) Average fitness value over time (averaged across `params.n_iterations` runs)
(2) Fraction of social learners over time (averaged across `params.n_iterations` runs)
(3) The average of the average fitness for the last `params.n_last` time steps (averaged across
    `params.n_iterations` runs)
(4) The average of the fraction of social learners for the last `params.n_last` time steps
    (averaged across `params.n_iterations` runs). With n_last = 400 this is the statistic
    reported in the notebook and the presentation.


Graphical interface (gui.py):

gui.py works on top of social_disorg.py and visualizations.py without changing them; the scripts
above still run on their own exactly as before. It opens with the values from params.py.
- "Live simulation" tab: runs one simulation and draws it while it runs: the ring of offenders
  colored by learning strategy, their current social ties (long-range lines appear when rho > 0),
  the cue of each site as an outer band, cue vs. behavior at every site, and the share of social
  learners and mean fitness over time (dotted lines mark environmental changes). Set r, q, the
  number of patches and rho at the top, press Reset to apply changes, then Start/Pause or Step;
  the speed slider sets how many time steps are run between redraws. The live run uses exactly
  the same steps as social_disorg.simulate(), so it gives the same results for the same seeds.
- "Parameter sweep" tab: enter lists of values (or numpy expressions such as
  np.linspace(0, 1, 11)) and run settings, then "Run sweep". It runs
  social_disorg.run_simulation() and visualizations.generate_visualizations() in a separate
  process, shows the progress and the resulting plots, and can be cancelled. Your params.py is
  not modified: the settings go into a temporary params.py, which is copied into the results
  folder as usual.
- Text size: guessed from the screen resolution at start-up. Change it with the "Text size" box in
  the left panel or Ctrl + / Ctrl - / Ctrl 0 (back to 100%); it scales both the interface and the
  plots. To start at a fixed size, run e.g. `python gui.py --scale 1.5`, or set DEFAULT_SCALE at
  the top of gui.py.
- "Load params.py…" fills the form from a parameter file; "Save as params.py…" writes the form
  to a file, e.g. to make the settings the default for the command-line scripts.
- If Tk on Linux was built without Xft font support (common with conda's tk), it cannot draw
  Greek letters and symbols, so gui.py detects this and writes them out instead (beta, mu, rho).
  Installing Tk from conda-forge (`conda install -c conda-forge tk`) usually gives nicer,
  anti-aliased fonts with the symbols. Set PLAIN_TEXT at the top of gui.py to force either mode.
tkinter ships with most Python installations. On Linux it may need to be installed separately,
e.g. `sudo apt install python3-tk`, or `conda install tk` in a conda environment.
