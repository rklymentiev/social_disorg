"Social Disorganization" simulation script

File structure:
├── output/                              # Directory for simulation outputs (will be generated automatically)
│   └── 20260527_110416_results/         # Example output directory (timestamped)
│       ├── average_fitness.json         # JSON file with average fitness values over time across all runs
│       ├── params.py                    # Parameters used for this simulation run
│       └── ratio_social_learners.json   # JSON file with the proportion of social offenders over time across all runs
├── params.py                            # Simulation parameters
├── README.txt                           # Project overview and usage instructions
├── requirements.txt                     # Python dependencies
├── social_disorg.py                     # Main simulation script
└── visualizations.py                    # Script for generating visualizations from simulation results


How to use the code:

(1) Create a new Python virtual environment (optional but recommended):
$ python -m venv socialdisorg

(2) Activate the virtual environment:
$ source socialdisorg/bin/activate  # On Linux/Mac
$ socialdisorg\Scripts\activate     # On Windows

(3) Install the required dependencies:
$ pip install -r requirements.txt

(4) Set up the parameters for the simulation in `params.py` (see below for details).

(5) Run the simulation:
$ python social_disorg.py

(6) Create visualizations from the output data:
$ python visualizations.py

Parameters of the simulation can be adjusted in the `params.py` file.
Currently, there are three parameters of interest that can be varied to explore different scenarios:
(1) `r_vals`: The proportion of offenders to shuffle (social disorganization)
(2) `q_vals`: The probability of cues changing their value (temporal variation)
(3) `n_patches_vals`: The number of patches of cues (spatial heterogeneity)
For simplicity, use the range of values for one of the parameters while keeping the others fixed to a single value.
See `params.py` for more details on how to adjust these parameters and the default values used in the simulation.

The script `social_disorg.py` will generate output in the `output/` directory, with a timestamped subdirectory for each run.
Each run will include JSON files with results for the average fitness value and the proportion of social offenders over time,
as well as a copy of the parameters used for that simulation (`params.py`).

The script `visualizations.py` can be used to create visualizations from the output data.
The script will ask for a directory containing the results to visualize, and it will generate 4 plots in the same directory:
(1) Average fitness value over time (averaged across `params.n_iterations` runs)
(2) Fraction of social offenders over time (averaged across `params.n_iterations` runs)
(3) The average of the average fitness for the last `params.n_last` time steps (averaged across `params.n_iterations` runs)
(4) The average of the fraction of social offenders for the last `params.n_last` time steps (averaged across `params.n_iterations` runs)