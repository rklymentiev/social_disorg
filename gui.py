"""
Graphical interface for the social disorganization model.

Works on top of social_disorg.py and visualizations.py without changing
them, so `python social_disorg.py`, `python visualizations.py` and
`python run_pipeline.py` keep working exactly as before.

Two tabs:
- Live simulation: one run of the model, drawn while it runs - the ring of
  offenders with their learning strategy and social ties, cues vs.
  behaviors at every site, and the share of social learners and mean
  fitness over time.
- Parameter sweep: runs social_disorg.run_simulation() and then
  visualizations.generate_visualizations() with the settings in the form,
  in a separate process, and shows the resulting plots. Your params.py is
  not touched; the settings are written to a temporary params.py that is
  copied into the results folder as usual. Use "Save as params.py..." if
  you want to keep them.

Usage:
    python gui.py               # text size guessed from the screen resolution
    python gui.py --scale 1.5   # or set it yourself (1 = 100%)
Change the text size while it runs with the "Text size" box or
Ctrl + / Ctrl - / Ctrl 0.
"""
import os
import re
import sys
import queue
import signal
import tempfile
import threading
import subprocess
import argparse
import importlib.util
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg

import params
import social_disorg as sd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# text size: None = guess from the screen resolution, or a factor such as 1.5
DEFAULT_SCALE = None
SCALE_STEPS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
NAMED_FONTS = ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont",
               "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont")
BASE_DPI = 100  # matplotlib dpi at 100%

# Tk builds without Xft (common with conda's tk on Linux) draw text with old
# X11 fonts that garble anything beyond Western European characters
# (β shows up as ∈″). None = detect; True/False = force plain text on/off.
PLAIN_TEXT = None
_PLAIN = False
_PLAIN_REPLACEMENTS = {"▶ ": "", "❚❚ ": "", "β": "beta", "μ": "mu", "ρ": "rho", "γ": "gamma",
                       "σ": "sigma", "—": "--", "…": "...", "−": "-", "≥": ">=", "≤": "<="}


def ui(text: str) -> str:
    """Text for a Tk widget: as is, or with plain replacements when Tk can't show it."""
    if not _PLAIN:
        return text
    for char, plain in _PLAIN_REPLACEMENTS.items():
        text = text.replace(char, plain)
    return "".join(ch if ord(ch) < 256 else "?" for ch in text)


def tk_needs_plain_text(root) -> bool:
    if PLAIN_TEXT is not None:
        return PLAIN_TEXT
    if root.tk.call("tk", "windowingsystem") != "x11":
        return False  # Windows and macOS fonts handle Unicode
    try:
        return root.tk.call("::tk::pkgconfig", "get", "fontsystem") != "xft"
    except tk.TclError:
        return True

# colors: categorical slots for the two strategies + a third series,
# muted ink for text, grid and ties
SOCIAL, ASOCIAL, FITNESS = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#22211f", "#6b6a64", "#e6e5e0", "#fcfcfb"

# (name in params.py, label, kind); kinds: int, float, optint (blank = None), choice:<a|b>
MODEL_FIELDS = [
    ("N", "Offenders (N)", "int"),
    ("T", "Time steps (T)", "int"),
    ("S", "Cue range (S)", "float"),
    ("beta", "Selection strength (β)", "float"),
    ("c", "Asocial cost (c)", "float"),
    ("sd_asocial", "Asocial error (SD)", "float"),
    ("mu", "Innovation rate (μ)", "float"),
    ("p_social_init", "Initial share social", "float"),
    ("patch_sizes", "Patch sizes", "choice:equal|random"),
    ("landscape_seed", "Landscape seed (blank = new)", "optint"),
    ("environment_seed", "Change seed (blank = new)", "optint"),
    ("SEED", "Seed (blank = random)", "optint"),
]
# free parameters: (single-value name in the live tab, list name in params.py, label, type)
FREE_FIELDS = [
    ("r", "r_vals", "Relocating share (r)", float),
    ("q", "q_vals", "Rate of environmental change (q)", float),
    ("n_patches", "n_patches_vals", "Number of patches", int),
    ("rho", "rho_vals", "Rewiring probability (ρ)", float),
]
SWEEP_FIELDS = [
    ("n_iterations", "Runs per combination", "int"),
    ("n_cores", "CPU cores", "int"),
    ("n_last", "Final steps averaged in plots", "int"),
    ("img_format", "Plot format", "choice:png|pdf|svg|jpg"),
]


def fmt_number(v) -> str:
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)


def fmt_list(vals) -> str:
    return ", ".join(fmt_number(v) for v in np.atleast_1d(vals))


def parse_list(text: str, cast) -> list:
    """Parse '0, 0.1, 0.2' or an expression such as 'np.linspace(0, 1, 11)'."""
    try:
        value = eval(text, {"np": np, "__builtins__": {}}, {})
    except Exception as exc:
        raise ValueError(f"could not read {text!r}: {exc}") from None
    values = np.atleast_1d(np.asarray(value, dtype=float))
    if values.size == 0:
        raise ValueError("the list is empty")
    if cast is int:
        if not np.allclose(values, np.round(values)):
            raise ValueError(f"{text!r} must contain whole numbers")
        return [int(round(v)) for v in values]
    return [float(v) for v in values]


class LiveModel:
    """
    One run of the model, advanced one step at a time. The step is the same
    sequence of calls as social_disorg.simulate() (checked: with the same
    seeds both give identical results); it is repeated here only so the GUI
    can draw the state between steps.
    """

    def __init__(self, m: dict, r: float, q: float, n_patches: int, rho: float):
        self.m, self.r, self.q, self.rho = m, r, q, rho
        self.n = m["N"]
        self.rng = np.random.default_rng(m["SEED"])
        landscape_rng = self.rng if m["landscape_seed"] is None else np.random.default_rng(m["landscape_seed"])
        self.env_rng = self.rng if m["environment_seed"] is None else np.random.default_rng(m["environment_seed"])

        self.s, self.patch_id = sd.make_landscape(self.n, n_patches, m["S"], landscape_rng, m["patch_sizes"])
        self.b = self.rng.uniform(0, m["S"], size=self.n)
        self.l = (self.rng.random(self.n) < m["p_social_init"]).astype(int)
        self.w = sd.fitness_function(self.b, self.s, self.l, m["beta"], m["c"])
        self.base_edges = sd.ring_edges(self.n)
        self.base_network = sd.edges_to_csr(self.base_edges, self.n)
        self.edges = self.base_edges

        self.t = 0
        self.share_social, self.mean_fitness, self.change_times = [], [], []

    def step(self):
        m, rng = self.m, self.rng
        l_vals = sd.replicator_update(self.l, self.w, rng)
        l_vals = sd.innovation(l_vals, m["mu"], rng)

        s_vals = sd.temporal_change(self.s, self.patch_id, self.q, m["S"], self.env_rng)
        if s_vals is not self.s:
            self.change_times.append(self.t + 1)
        b_vals, l_vals = sd.residential_instability(self.b, l_vals, self.r, rng)
        if self.rho > 0:
            self.edges = sd.rewire_edges(self.base_edges, self.n, self.rho, rng)
            network = sd.edges_to_csr(self.edges, self.n)
        else:
            network = self.base_network

        b_vals = sd.learning(b_vals, l_vals, s_vals, network, m["sd_asocial"], rng)
        self.w = sd.fitness_function(b_vals, s_vals, l_vals, m["beta"], m["c"])
        self.s, self.b, self.l = s_vals, b_vals, l_vals
        self.t += 1
        self.share_social.append(l_vals.mean())
        self.mean_fitness.append(self.w.mean())


class App(tk.Tk):
    def __init__(self, scale=None):
        super().__init__()
        global _PLAIN
        _PLAIN = tk_needs_plain_text(self)
        self.title("Social disorganization model")
        self.base_fonts = {}
        for name in NAMED_FONTS:
            try:
                self.base_fonts[name] = tkfont.nametofont(name).cget("size")
            except tk.TclError:
                pass
        if scale is None:
            scale = DEFAULT_SCALE if DEFAULT_SCALE is not None else self.guess_scale()
        self.scale = scale
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.vars = {}
        self.model = None
        self.running = False
        self.after_id = None
        self.proc = None
        self.out_queue = queue.Queue()
        self.results_dir = None

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        self.paned = paned
        self.model_panel = self._build_model_panel(paned)
        paned.add(self.model_panel, weight=0)
        tabs = ttk.Notebook(paned)
        tabs.add(self._build_live_tab(tabs), text="Live simulation")
        tabs.add(self._build_sweep_tab(tabs), text="Parameter sweep")
        paned.add(tabs, weight=1)

        self.load_params_module(params)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        for keys, action in ((("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"), lambda: self.zoom(+1)),
                             (("<Control-minus>", "<Control-KP_Subtract>"), lambda: self.zoom(-1)),
                             (("<Control-0>", "<Control-KP_0>"), lambda: self.set_scale(1.0))):
            for key in keys:
                self.bind_all(key, lambda event, a=action: a())

        self.set_scale(self.scale, resize_window=False)
        for fig, canvas in ((self.live_fig, self.live_canvas), (self.result_fig, self.result_canvas)):
            # runs after matplotlib's own <Map> handler, which resets the dpi
            canvas.get_tk_widget().bind("<Map>", lambda e, f=fig, c=canvas: self._scale_figure(f, c), add="+")
        self.fit_window()
        self.minsize(min(1000, self.winfo_screenwidth() - 40), min(640, self.winfo_screenheight() - 80))
        self.after(100, self.reset_live)

    # ----------------------------------------------------------- text size
    def guess_scale(self) -> float:
        """Rough guess for high-resolution laptop screens, rounded to 25%."""
        pixel_fonts = self.base_fonts.get("TkDefaultFont", -1) < 0  # point sizes already follow the dpi
        by_dpi = self.winfo_fpixels("1i") / 96 if pixel_fonts else 1.0
        by_height = self.winfo_screenheight() / 1080
        guess = max(1.0, by_dpi, by_height if by_height > 1.15 else 1.0)
        return float(min(2.5, round(guess * 4) / 4))

    def fit_window(self):
        """Size the window for the current text size, within the screen."""
        width = min(int(1360 * self.scale), self.winfo_screenwidth() - 40)
        height = min(int(860 * self.scale), self.winfo_screenheight() - 80)
        self.geometry(f"{width}x{height}")

    def set_scale(self, factor: float, resize_window: bool = True):
        """Scale every interface font and both plot areas by `factor`."""
        factor = float(min(max(factor, SCALE_STEPS[0]), SCALE_STEPS[-1]))
        self.scale = factor
        for name, size in self.base_fonts.items():
            if size:
                new = max(1, round(abs(size) * factor))
                tkfont.nametofont(name).configure(size=new if size > 0 else -new)
        self.speed_scale.configure(length=int(180 * factor))
        for fig, canvas in ((self.live_fig, self.live_canvas), (self.result_fig, self.result_canvas)):
            self._scale_figure(fig, canvas)
        self.scale_text.set(f"{round(factor * 100)}%")
        # widen the left panel to fit the resized text
        self.update_idletasks()
        try:
            self.paned.sashpos(0, self.model_panel.winfo_reqwidth())
        except tk.TclError:
            pass
        if resize_window:
            self.fit_window()

    def _scale_figure(self, fig, canvas):
        """
        Matplotlib sets a figure's dpi to (original dpi x the screen's pixel
        ratio) whenever the canvas is mapped, so scale the original dpi; the
        division keeps plot text in step with the (pixel-sized) interface fonts.
        """
        dpi = BASE_DPI * self.scale
        ratio = getattr(canvas, "device_pixel_ratio", 1) or 1
        fig._original_dpi = dpi / ratio
        if hasattr(fig, "_set_dpi"):
            fig._set_dpi(dpi, forward=False)
        else:
            fig.dpi = dpi
        widget = canvas.get_tk_widget()
        if widget.winfo_width() > 1:
            fig.set_size_inches(widget.winfo_width() / dpi, widget.winfo_height() / dpi, forward=False)
        canvas.draw_idle()

    def zoom(self, direction: int):
        steps = SCALE_STEPS
        if direction > 0:
            self.set_scale(next((v for v in steps if v > self.scale + 1e-9), steps[-1]))
        else:
            self.set_scale(next((v for v in reversed(steps) if v < self.scale - 1e-9), steps[0]))

    # ------------------------------------------------------------------ form
    def _add_field(self, parent, row, name, label, kind, width=8):
        ttk.Label(parent, text=ui(label)).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        var = tk.StringVar()
        if kind.startswith("choice:"):
            widget = ttk.Combobox(parent, textvariable=var, values=kind.split(":", 1)[1].split("|"),
                                  state="readonly", width=width - 2)
        else:
            widget = ttk.Entry(parent, textvariable=var, width=width)
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        self.vars[name] = (var, kind)

    def _build_model_panel(self, parent):
        frame = ttk.Frame(parent, padding=(4, 4, 10, 4))
        box = ttk.LabelFrame(frame, text="Model", padding=8)
        box.pack(fill="x")
        for row, (name, label, kind) in enumerate(MODEL_FIELDS):
            self._add_field(box, row, name, label, kind)
        box.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Shared by both tabs. In the live tab,\nchanges apply when you press Reset.",
                  foreground=MUTED).pack(anchor="w", pady=(6, 10))
        ttk.Button(frame, text=ui("Load params.py…"), command=self.ask_load_params).pack(fill="x", pady=2)
        ttk.Button(frame, text=ui("Save as params.py…"), command=self.ask_save_params).pack(fill="x", pady=2)
        size = ttk.Frame(frame)
        size.pack(fill="x", pady=(14, 0))
        ttk.Label(size, text="Text size").pack(side="left")
        self.scale_text = tk.StringVar()
        box = ttk.Combobox(size, textvariable=self.scale_text, width=6, state="readonly",
                           values=[f"{round(v * 100)}%" for v in SCALE_STEPS])
        box.pack(side="left", padx=6)
        box.bind("<<ComboboxSelected>>", lambda event: self.set_scale(int(self.scale_text.get()[:-1]) / 100))
        ttk.Label(size, text=ui("Ctrl + / −"), foreground=MUTED).pack(side="left")
        return frame

    def _read(self, name):
        var, kind = self.vars[name]
        text = var.get().strip()
        try:
            if kind == "int":
                return int(text)
            if kind == "float":
                return float(text)
            if kind == "optint":
                return None if text in ("", "None") else int(text)
            return text  # choice / free text
        except ValueError:
            raise ValueError(f"'{text}' is not a valid value for {name}") from None

    def read_model(self) -> dict:
        m = {name: self._read(name) for name, _, _ in MODEL_FIELDS}
        if m["N"] < 3 or m["T"] < 1:
            raise ValueError("N must be at least 3 and T at least 1")
        return m

    def load_params_module(self, module):
        for name, _, kind in MODEL_FIELDS + SWEEP_FIELDS:
            value = getattr(module, name, None)
            self.vars[name][0].set("" if value is None else fmt_number(value))
        self.output_dir.set(getattr(module, "output_dir", "../02_output/"))
        for single, plural, _, cast in FREE_FIELDS:
            vals = list(np.atleast_1d(getattr(module, plural, [0])))
            self.vars[plural][0].set(fmt_list(vals))
            self.vars[single][0].set(fmt_number(cast(vals[0])))

    def ask_load_params(self):
        path = filedialog.askopenfilename(initialdir=CODE_DIR, title="Load parameters",
                                          filetypes=[("Python files", "*.py")])
        if not path:
            return
        spec = importlib.util.spec_from_file_location("params_loaded", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            messagebox.showerror("Could not load", ui(str(exc)))
            return
        self.load_params_module(module)

    def params_source(self, absolute_output: bool) -> str:
        """Contents of a params.py with every setting in the form."""
        m = self.read_model()
        values = dict(m)
        for name, _, _ in SWEEP_FIELDS:
            values[name] = self._read(name)
        out = self.output_dir.get().strip() or "../02_output/"
        if absolute_output and not os.path.isabs(out):
            out = os.path.normpath(os.path.join(CODE_DIR, out))
        values["output_dir"] = out
        for _, plural, label, cast in FREE_FIELDS:
            values[plural] = parse_list(self.vars[plural][0].get(), cast)
        lines = ['"""Parameters written by gui.py."""', "import numpy as np", ""]
        for key, value in values.items():
            lines.append(f"{key} = {value!r}")
        return "\n".join(lines) + "\n"

    def ask_save_params(self):
        try:
            source = self.params_source(absolute_output=False)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", ui(str(exc)))
            return
        path = filedialog.asksaveasfilename(initialdir=CODE_DIR, initialfile="params.py",
                                            defaultextension=".py", filetypes=[("Python files", "*.py")])
        if path:
            with open(path, "w") as fh:
                fh.write(source)

    # ------------------------------------------------------------- live tab
    def _build_live_tab(self, parent):
        frame = ttk.Frame(parent, padding=6)
        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        short = {"r": "Relocating share r", "q": "Change rate q", "n_patches": "Patches", "rho": "Rewiring ρ"}
        for col, (single, _, label, _) in enumerate(FREE_FIELDS):
            ttk.Label(bar, text=ui(short[single])).grid(row=0, column=2 * col, sticky="w", padx=(0, 4))
            var = tk.StringVar()
            ttk.Entry(bar, textvariable=var, width=8).grid(row=0, column=2 * col + 1, padx=(0, 14))
            self.vars[single] = (var, "float" if single != "n_patches" else "int")

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(8, 4))
        ttk.Button(controls, text="Reset", command=self.reset_live).pack(side="left")
        self.play_button = ttk.Button(controls, text=ui("▶ Start"), width=10, command=self.toggle_live)
        self.play_button.pack(side="left", padx=6)
        ttk.Button(controls, text="Step", command=self.step_once).pack(side="left")
        ttk.Label(controls, text="Speed (steps per frame)").pack(side="left", padx=(20, 4))
        self.speed = tk.IntVar(value=1)
        self.speed_scale = ttk.Scale(controls, from_=1, to=50, variable=self.speed, length=180,
                                     command=lambda v: self.speed.set(int(float(v))))
        self.speed_scale.pack(side="left")
        ttk.Label(controls, textvariable=self.speed, width=3).pack(side="left", padx=4)
        self.live_status = tk.StringVar()
        ttk.Label(frame, textvariable=self.live_status, foreground=INK).pack(anchor="w", pady=(2, 0))

        self.live_fig = Figure(figsize=(10, 6.4), facecolor=SURFACE, layout="constrained")
        grid = self.live_fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
        self.ax_ring = self.live_fig.add_subplot(grid[:, 0])
        self.ax_cue = self.live_fig.add_subplot(grid[0, 1])
        self.ax_time = self.live_fig.add_subplot(grid[1, 1])
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=frame)
        self.live_canvas.get_tk_widget().pack(fill="both", expand=True)
        return frame

    def _style(self, ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    def reset_live(self):
        self.pause_live()
        try:
            m = self.read_model()
            r, q, n_patches, rho = (self._read(s) for s, _, _, _ in FREE_FIELDS)
            if not 1 <= n_patches <= m["N"]:
                raise ValueError("the number of patches must be between 1 and N")
            if not (0 <= r <= 1 and 0 <= rho <= 1 and q >= 0):
                raise ValueError("r and ρ must be in [0, 1] and q must be ≥ 0")
            self.model = LiveModel(m, r, q, n_patches, rho)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", ui(str(exc)))
            return
        self._setup_live_plot()
        self._update_live_plot()

    def _setup_live_plot(self):
        mod, n, S, T = self.model, self.model.n, self.model.m["S"], self.model.m["T"]
        for ax in (self.ax_ring, self.ax_cue, self.ax_time):
            ax.clear()

        # ring: outer band = cue value, dots = offenders, lines = social ties
        theta = np.pi / 2 - 2 * np.pi * np.arange(n) / n
        self.pos = np.column_stack([np.cos(theta), np.sin(theta)])
        ax = self.ax_ring
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.42)
        size = float(np.clip(4000 / n, 6, 90))
        self.ties = LineCollection([], colors=MUTED, linewidths=0.8, alpha=0.6, zorder=1)
        ax.add_collection(self.ties)
        self.cue_band = ax.scatter(*(1.16 * self.pos.T), c=mod.s, cmap="Greys", vmin=-0.25 * S, vmax=S,
                                   marker="s", s=size * 0.9, linewidths=0, zorder=1)
        self.agents = ax.scatter(*self.pos.T, s=size, edgecolors=SURFACE, linewidths=0.8, zorder=2)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SOCIAL, label="social learner"),
                           Line2D([], [], marker="o", ls="", color=ASOCIAL, label="asocial learner"),
                           Line2D([], [], marker="s", ls="", color="#8a8984", label="cue (darker = higher)")],
                  loc="lower left", ncol=1, frameon=False, fontsize=8, bbox_to_anchor=(-1.25, 1.22),
                  bbox_transform=ax.transData, handletextpad=0.2)

        # cue vs behavior at every site
        ax = self.ax_cue
        self._style(ax)
        x = np.arange(n)
        self.cue_line, = ax.step(x, mod.s, where="mid", color=INK, lw=1.5, label="cue s")
        self.behaviors = ax.scatter(x, mod.b, s=float(np.clip(1500 / n, 4, 30)), linewidths=0, zorder=3)
        ax.set(xlim=(-0.5, n - 0.5), ylim=(-0.05 * S, 1.05 * S))
        ax.set_xlabel("site", color=INK, fontsize=9)
        ax.set_ylabel("cue / behavior", color=INK, fontsize=9)
        ax.set_title("Cue and behavior at each site", fontsize=10, color=INK, loc="left")

        # share of social learners and mean fitness over time (both 0-1)
        ax = self.ax_time
        self._style(ax)
        self.line_social, = ax.plot([], [], color=SOCIAL, lw=2, label="share of social learners")
        self.line_fitness, = ax.plot([], [], color=FITNESS, lw=2, label="mean fitness")
        ax.set(xlim=(0, T), ylim=(0, 1.05))
        ax.set_xlabel("time step", color=INK, fontsize=9)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=1, frameon=False, fontsize=8,
                  borderaxespad=0.2)
        self.drawn_changes = 0

    def _update_live_plot(self):
        mod = self.model
        colors = np.where(mod.l == 1, SOCIAL, ASOCIAL)
        self.agents.set_facecolors(colors)
        self.cue_band.set_array(mod.s)
        self.ties.set_segments(self.pos[mod.edges])
        self.cue_line.set_ydata(mod.s)
        self.behaviors.set_offsets(np.column_stack([np.arange(mod.n), mod.b]))
        self.behaviors.set_facecolors(colors)
        t = np.arange(1, mod.t + 1)
        self.line_social.set_data(t, mod.share_social)
        self.line_fitness.set_data(t, mod.mean_fitness)
        for when in mod.change_times[self.drawn_changes:]:
            self.ax_time.axvline(when, color=MUTED, lw=0.8, ls=":", zorder=0)
        self.drawn_changes = len(mod.change_times)
        if mod.t:
            self.live_status.set(f"t = {mod.t} / {mod.m['T']}   ·   social learners {mod.share_social[-1]:.0%}"
                                 f"   ·   mean fitness {mod.mean_fitness[-1]:.2f}")
        else:
            self.live_status.set(f"t = 0 / {mod.m['T']}   ·   social learners {mod.l.mean():.0%}")
        self.live_canvas.draw_idle()

    def toggle_live(self):
        if self.running:
            self.pause_live()
        else:
            if self.model is None or self.model.t >= self.model.m["T"]:
                self.reset_live()
                if self.model is None:
                    return
            self.running = True
            self.play_button.config(text=ui("❚❚ Pause"))
            self._tick()

    def pause_live(self):
        self.running = False
        self.play_button.config(text=ui("▶ Start"))
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def step_once(self):
        if self.model is None:
            return
        self.pause_live()
        if self.model.t < self.model.m["T"]:
            self.model.step()
            self._update_live_plot()

    def _tick(self):
        if not self.running:
            return
        for _ in range(max(1, self.speed.get())):
            if self.model.t >= self.model.m["T"]:
                break
            self.model.step()
        self._update_live_plot()
        if self.model.t >= self.model.m["T"]:
            self.pause_live()
            return
        self.after_id = self.after(15, self._tick)

    # ------------------------------------------------------------ sweep tab
    def _build_sweep_tab(self, parent):
        frame = ttk.Frame(parent, padding=6)
        top = ttk.Frame(frame)
        top.pack(fill="x")

        free = ttk.LabelFrame(top, text="Free parameters (vary one at a time)", padding=8)
        free.pack(side="left", fill="both", expand=True)
        for row, (_, plural, label, _) in enumerate(FREE_FIELDS):
            self._add_field(free, row, plural, label, "text", width=24)
        ttk.Label(free, text="Comma-separated values or numpy,\ne.g. np.linspace(0, 1, 11)",
                  foreground=MUTED).grid(row=len(FREE_FIELDS), column=0, columnspan=2, sticky="w", pady=(4, 0))
        free.columnconfigure(1, weight=1)

        run = ttk.LabelFrame(top, text="Run", padding=8)
        run.pack(side="left", fill="both", padx=(8, 0))
        for row, (name, label, kind) in enumerate(SWEEP_FIELDS):
            self._add_field(run, row, name, label, kind, width=7)
        row = len(SWEEP_FIELDS)
        ttk.Label(run, text="Output folder").grid(row=row, column=0, sticky="w", pady=2)
        self.output_dir = tk.StringVar()
        out = ttk.Frame(run)
        out.grid(row=row, column=1, sticky="ew")
        ttk.Entry(out, textvariable=self.output_dir, width=14).pack(side="left", fill="x", expand=True)
        ttk.Button(out, text=ui("…"), width=3, command=self.ask_output_dir).pack(side="left")
        self.make_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(run, text="Make plots (visualizations.py)", variable=self.make_plots).grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=8)
        self.run_button = ttk.Button(buttons, text="Run sweep", command=self.run_sweep)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.cancel_sweep, state="disabled")
        self.cancel_button.pack(side="left", padx=6)
        self.open_button = ttk.Button(buttons, text="Open results folder", command=self.open_results,
                                      state="disabled")
        self.open_button.pack(side="left")
        self.progress = ttk.Progressbar(buttons, length=260, mode="determinate")
        self.progress.pack(side="left", padx=(16, 6))
        self.progress_text = tk.StringVar()
        ttk.Label(buttons, textvariable=self.progress_text).pack(side="left")

        self.log = tk.Text(frame, height=5, wrap="word", relief="flat", background="#f3f2ee",
                           foreground=INK, font="TkFixedFont")
        self.log.pack(fill="x")
        self.log.configure(state="disabled")

        self.result_fig = Figure(figsize=(10, 5), facecolor=SURFACE, layout="constrained")
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=frame)
        self.result_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        return frame

    def ask_output_dir(self):
        path = filedialog.askdirectory(initialdir=CODE_DIR, title="Folder for simulation results")
        if path:
            self.output_dir.set(path)

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", ui(text))
        self.log.see("end")
        self.log.configure(state="disabled")

    def run_sweep(self):
        try:
            source = self.params_source(absolute_output=True)
            for _, plural, _, cast in FREE_FIELDS:
                parse_list(self.vars[plural][0].get(), cast)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", ui(str(exc)))
            return
        varying = [label for _, plural, label, cast in FREE_FIELDS
                   if len(parse_list(self.vars[plural][0].get(), cast)) > 1]
        plots = self.make_plots.get()
        if plots and len(varying) > 1:
            if not messagebox.askyesno(
                    "More than one free parameter",
                    ui("visualizations.py plots one free parameter at a time, but these vary:\n\n"
                       + "\n".join(varying) + "\n\nRun the simulation without plots?")):
                return
            plots = False
        if plots and not varying:
            self.write_log("No free parameter varies, so there is nothing to plot; running the simulation only.\n")
            plots = False

        stage = tempfile.mkdtemp(prefix="socialdisorg_gui_")
        with open(os.path.join(stage, "params.py"), "w") as fh:
            fh.write(source)
        # import order: the staged params.py first, then this folder's scripts
        boot = (f"import sys; sys.path[:0] = [{stage!r}, {CODE_DIR!r}]\n"
                "import social_disorg, visualizations\n"
                "d = social_disorg.run_simulation()\n"
                "print('RESULTS_DIR=' + d, flush=True)\n")
        if plots:
            boot += "visualizations.generate_visualizations(d)\n"
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
        kwargs = dict(cwd=stage, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.proc = subprocess.Popen([sys.executable, "-c", boot], **kwargs)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

        self.results_dir = None
        self.progress["value"] = 0
        self.progress_text.set(ui("starting…"))
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.write_log("— Running sweep —\n")
        self.after(100, self._poll_sweep)

    def _reader(self, proc):
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            self.out_queue.put(chunk.decode(errors="replace"))
        self.out_queue.put(None)

    def _poll_sweep(self):
        finished = False
        buffer = ""
        while True:
            try:
                item = self.out_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                finished = True
                break
            buffer += item
        for segment in re.split(r"[\r\n]", buffer):
            if not segment.strip():
                continue
            bar = re.search(r"Running simulations:.*?(\d+)/(\d+)", segment)
            if bar:
                done, total = int(bar.group(1)), int(bar.group(2))
                self.progress["maximum"] = total
                self.progress["value"] = done
                self.progress_text.set(f"{done} / {total} parameter combinations")
            elif segment.startswith("RESULTS_DIR="):
                self.results_dir = segment.split("=", 1)[1].strip()
                self.open_button.config(state="normal")
            else:
                self.write_log(segment + "\n")
        if finished:
            code = self.proc.wait()
            self.proc = None
            self.run_button.config(state="normal")
            self.cancel_button.config(state="disabled")
            if code == 0:
                self.progress_text.set("done")
                self.show_results()
            else:
                self.progress_text.set("stopped" if code < 0 or code in (1, 15) else f"failed ({code})")
                self.write_log(f"Process ended with code {code}.\n")
            return
        self.after(100, self._poll_sweep)

    def cancel_sweep(self):
        if self.proc is None:
            return
        try:
            if os.name == "nt":
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        self.write_log("Cancelled. The results folder may contain the combinations finished so far.\n")

    def show_results(self):
        self.result_fig.clear()
        if not self.results_dir:
            return
        images = sorted(f for f in os.listdir(self.results_dir) if f.lower().endswith(".png"))
        order = ["fraction_social_by_", "avg_fitness_by_", "average_fitness_by_"]
        images.sort(key=lambda f: (not f.endswith("_t.png"), next((i for i, p in enumerate(order) if f.startswith(p)), 9)))
        if not images:
            ax = self.result_fig.add_subplot()
            ax.axis("off")
            ax.text(0.5, 0.5, f"Results saved in\n{self.results_dir}", ha="center", va="center", color=INK)
        for i, name in enumerate(images[:4]):
            ax = self.result_fig.add_subplot(2, 2, i + 1) if len(images) > 1 else self.result_fig.add_subplot()
            ax.imshow(mpimg.imread(os.path.join(self.results_dir, name)))
            ax.axis("off")
        self.result_canvas.draw_idle()
        self.write_log(f"Results saved in {self.results_dir}\n")

    def open_results(self):
        if not self.results_dir:
            return
        if sys.platform.startswith("win"):
            os.startfile(self.results_dir)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", self.results_dir])

    def on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("Sweep running", "A sweep is still running. Stop it and quit?"):
                return
            self.cancel_sweep()
        self.pause_live()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description="Graphical interface for the social disorganization model.")
    parser.add_argument("--scale", type=float, default=None,
                        help="text size factor, e.g. 1.5 for 150%% (default: guessed from the screen)")
    App(scale=parser.parse_args().scale).mainloop()


if __name__ == "__main__":
    main()
