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

from epiboly_init import LeadingEdge, Little
import config as cfg
import utils.epiboly_utils as epu
import utils.video_export as vx

_phi: list[float] = []
_timesteps: list[int] = []
_timestep: int = 0
_fig: Optional[Figure] = None
_ax: Optional[Axes] = None

def _init_graph() -> None:
    global _fig, _ax
    
    _fig, _ax = plt.subplots()
    _ax.set_ylabel(r"Leading edge  $\bar{\phi}$  (radians)")

def show_graph() -> None:
    global _timestep
    
    if LeadingEdge.items()[0].frozen_z:
        # During the z-frozen phase of equilibration, don't even increment _timestep, so that once
        # we start graphing, it will start at time 0, representing the moment when the leading edge
        # becomes free to move.
        return

    if not _fig:
        # if init hasn't been run yet, run it
        _init_graph()

    # Don't need to add to the graph every timestep.
    if _timestep % 100 == 0:
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
        total_evl_cells: int = len(Little.items()) + len(LeadingEdge.items())
        filename: str = ""
        if end is not None:
            filename += "End. " if end else "Start. "
        filename += f"Num cells = {total_evl_cells}; radius = {round(Little.radius, 2)}"
        filename += f" ({cfg.num_spherical_positions} + {cfg.num_leading_edge_points})"
        filename += f", external = {cfg.yolk_cortical_tension}"
        filename += ".png"
        filepath: str = os.path.join(vx.sim_root(), filename)
        _fig.savefig(filepath, transparent=False, bbox_inches="tight")
        
# At import time: set to interactive mode ("ion" = "interactive on") so that plot display isn't blocking.
# Note to self: do I need to make sure interactive is off, when I'm in windowless mode? That would be
# necessary for true automation, but would be nice to run windowless manually and still see the plots.
# However, it seems like TF is suppressing that; in windowless only, the plots aren't showing up once
# I've called this function.
plt.ion()
