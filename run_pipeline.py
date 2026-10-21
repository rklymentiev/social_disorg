"""
Runs the full pipeline end to end: simulate, then visualize.

Equivalent to running `python social_disorg.py` followed by
`python visualizations.py`, except the results folder social_disorg.py
just wrote is passed straight into visualizations.py automatically - no
folder-picker dialog, no manual copy-pasting of the output path.

Usage:
    python run_pipeline.py
"""
import social_disorg
import visualizations

if __name__ == "__main__":
    save_dir = social_disorg.run_simulation()
    visualizations.generate_visualizations(save_dir)