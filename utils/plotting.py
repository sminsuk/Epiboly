import os
from typing import Optional

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from epiboly_init import LeadingEdge
import utils.epiboly_utils as epu
import utils.video_export as vx

_phi: list[float] = []
_timesteps: list[int] = []
_timestep: int = 0
_fig: Optional[Figure] = None
def show_graph() -> None:
    """Not really viable, but just as a test and learning experience. New graph each timestep."""
    global _timestep, _fig
    
    if LeadingEdge.items()[0].frozen_z:
        # During the z-frozen phase of equilibration, don't even increment _timestep, so that once
        # we start graphing, it will start at time 0, representing the moment when the leading edge
        # becomes free to move.
        return
    
    # Until I have proper animation figured out, just do this now and then
    if _timestep % 1000 == 0:
        _fig, ax = plt.subplots()
    
        phi: float = round(epu.leading_edge_mean_phi(), 4)
        print(f"Appending: {_timestep}, {phi}")
        _timesteps.append(_timestep)
        _phi.append(phi)
    
        ax.plot(_timesteps, _phi, "bo")
        ax.set_ylabel("mean phi of leading edge (radians)")
        plt.show()

    _timestep += 1
    
def save_graph() -> None:
    if _fig:
        # i.e., only if show_graph() was ever run
        filepath: str = os.path.join(vx.sim_root(), "Plot.png")
        _fig.savefig(filepath, transparent=False, bbox_inches="tight")
