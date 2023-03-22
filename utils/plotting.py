"""Note to self: These StackOverflow articles made it seem like getting the plot to be non-blocking
and continually update, live, was going to be very hard. Strangely enough, once I followed the MPL
docs and simply turned on plt.ion(), everything just worked. I won't rule out nasty surprises in the
future if I change something, though.

https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
"""

import os
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import epiboly_globals as g
import config as cfg
import utils.epiboly_utils as epu
import utils.tf_utils as tfu

_phi: list[float] = []
_timesteps: list[int] = []
_timestep: int = 0
_fig: Optional[Figure] = None
_ax: Optional[Axes] = None
_plot_path: str

# in case sim is ended and restarted from exported data, output a new plot, going back all the way
# to the beginning, spanning all parts of the composite sim. But for now, number it and retain the
# earlier partial plot image as well.
_plot_num: int = 1

def _init_graph() -> None:
    """Initialize matplotlib and also a subdirectory in which to put the saved plots
    
    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _fig, _ax, _plot_path
    
    _fig, _ax = plt.subplots()
    _ax.set_ylabel(r"Leading edge  $\bar{\phi}$  (radians)")
    
    _plot_path = os.path.join(tfu.export_path(), "Plots")
    os.makedirs(_plot_path, exist_ok=True)

def show_graph(timer_override: bool = False) -> None:
    global _timestep
    
    if g.LeadingEdge.items()[0].frozen_z:
        # During the z-frozen phase of equilibration, don't even increment _timestep, so that once
        # we start graphing, it will start at time 0, representing the moment when the leading edge
        # becomes free to move.
        return

    if not _fig:
        # if init hasn't been run yet, run it
        _init_graph()

    # Don't need to add to the graph every timestep.
    if _timestep % 100 == 0 or timer_override:
        phi: float = round(epu.leading_edge_mean_phi(), 4)
        print(f"Appending: {_timestep}, {phi}")
        _timesteps.append(_timestep)
        _phi.append(phi)
        
        # ToDo? In windowless, technically we don't need to do this until once, at the end, just before
        #  saving the plot. Test for that? Would that improve performance, since it would avoid rendering?
        #  (In HPC? When executing manually?) Of course, need this for windowed mode, for live-updating plot.
        _ax.plot(_timesteps, _phi, "bo")

    _timestep += 1
    
def save_graph(end: Optional[bool] = None) -> None:
    if _fig:
        # i.e., only if init_graph() was ever run
        if end:
            # Final save, plot one final data point
            show_graph(timer_override=True)
            
        total_evl_cells: int = len(g.Little.items()) + len(g.LeadingEdge.items())
        filename: str = f"{_plot_num}. "
        if end is not None:
            filename += f"End. Timestep = {_timestep - 1}; " if end else "Start. "
        filename += f"Num cells = {total_evl_cells}; radius = {round(g.Little.radius, 2)}"
        filename += f" ({cfg.num_spherical_positions} + {cfg.num_leading_edge_points})"
        filename += f", external = {cfg.yolk_cortical_tension} + {cfg.external_force}"
        filename += ".png"
        filepath: str = os.path.join(_plot_path, filename)
        _fig.savefig(filepath, transparent=False, bbox_inches="tight")
        
def get_state() -> dict:
    """In composite runs, produce multiple plots, each numbered - but cumulative, all back to 0
    
    Each run saves its own plot, but the data is saved as part of the state, so the next run
    can import it and graph all the way from Timestep 0. Thus you get separate plots showing
    what was the state at the end of each run, but the final plot contains everything.
    (Note to self, once I'm confident of this, I can get rid of _plot_num; then each run will
    use the same filename and there will only be ONE plot.)
    """
    return {"plotnum": _plot_num,
            "timestep": _timestep,
            "phi": _phi,
            "timesteps": _timesteps,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved.
    
    Increment _plot_num with each run to generate a new filename, hence separate plot
    """
    global _plot_num, _timestep, _phi, _timesteps
    _plot_num = d["plotnum"] + 1
    _timestep = d["timestep"]
    _phi = d["phi"]
    _timesteps = d["timesteps"]
    
# At module import: set to interactive mode ("ion" = "interactive on") so that plot display isn't blocking.
# Note to self: do I need to make sure interactive is off, when I'm in windowless mode? That would be
# necessary for true automation, but would be nice to run windowless manually and still see the plots.
# However, it seems like TF is suppressing that; in windowless only, the plots aren't showing up once
# I've called this function.
plt.ion()
