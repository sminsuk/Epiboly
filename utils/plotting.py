"""plotting.py - matplotlib plots

Note to self: These StackOverflow articles made it seem like getting the plot to be non-blocking
and continually update, live, was going to be very hard. Strangely enough, once I followed the MPL
docs and simply turned on plt.ion(), everything just worked. I won't rule out nasty surprises in the
future if I change something, though.

https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
"""

import os
from statistics import fmean
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import neighbors as nbrs
import utils.epiboly_utils as epu
import utils.tf_utils as tfu

_phi: list[float] = []
_bonds_per_particle: list[float] = []
_timesteps: list[int] = []
_timestep: int = 0
_progress_fig: Optional[Figure] = None
_progress_ax: Optional[Axes] = None
_energy_fig: Optional[Figure] = None
_energy_ax: Optional[Axes] = None
_potentials_fig: Optional[Figure] = None
_potentials_ax: Optional[Axes] = None
_bond_lengths_fig: Optional[Figure] = None
_bond_lengths_ax: Optional[Axes] = None
_bond_count_fig: Optional[Figure] = None
_bond_count_ax: Optional[Axes] = None
_plot_path: str

# in case sim is ended and restarted from exported data, output a new plot, going back all the way
# to the beginning, spanning all parts of the composite sim. But for now, number it and retain the
# earlier partial plot image as well.
_plot_num: int = 1

def _init_graphs() -> None:
    """Initialize matplotlib and also a subdirectory in which to put the saved plots
    
    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _progress_fig, _progress_ax, _plot_path

    _progress_fig, _progress_ax = plt.subplots()
    _progress_ax.set_ylabel(r"Leading edge  $\bar{\phi}$  (radians)")
    
    # _init_test_energy_v_distance()
    _init_test_bondlengths_v_phi()
    _init_bond_counts()
    
    _plot_path = os.path.join(tfu.export_path(), "Plots")
    os.makedirs(_plot_path, exist_ok=True)

def _init_test_energy_v_distance() -> None:
    global _energy_fig, _energy_ax, _potentials_fig, _potentials_ax
    
    _energy_fig, _energy_ax = plt.subplots()
    _energy_ax.set_xlabel("particle distance")
    _energy_ax.set_ylabel("bond energy")
    _potentials_fig, _potentials_ax = plt.subplots()
    _potentials_ax.set_xlabel("particle distance")
    _potentials_ax.set_ylabel("bond potential")

def _show_test_energy_v_distance() -> None:
    """Plot energy vs length, & potential vs. length (this is to see if they are the same)"""
    energy: list[float] = []
    potentials: list[float] = []
    distance: list[float] = []
    bhandle: tf.BondHandle
    
    for bhandle in tf.BondHandle.items()[:100]:
        distance.append(bhandle.length)
        energy.append(bhandle.energy)
        potentials.append(bhandle.potential(bhandle.length))

    _energy_ax.plot(distance, energy, "bo")
    _potentials_ax.plot(distance, potentials, "ro")

def _save_graph_energy_v_distance() -> None:
    energypath: str = os.path.join(_plot_path, "Energy vs. bond distance.png")
    _energy_fig.savefig(energypath, transparent=False, bbox_inches="tight")
    potentialpath: str = os.path.join(_plot_path, "Potential vs. bond distance.png")
    _potentials_fig.savefig(potentialpath, transparent=False, bbox_inches="tight")

def _init_test_bondlengths_v_phi() -> None:
    global _bond_lengths_fig, _bond_lengths_ax

    _bond_lengths_fig, _bond_lengths_ax = plt.subplots()
    _bond_lengths_ax.set_xlabel("particle phi")
    _bond_lengths_ax.set_ylabel("mean bond length")

def _show_test_bondlengths_v_phi() -> None:
    """Plot mean bond length of all bonds on a particle, vs. phi of the particle"""
    bhandle: tf.BondHandle
    phandle: tf.ParticleHandle
    neighbor: tf.ParticleHandle
    mean_length: list[float] = []
    particle_phi: list[float] = []
    for phandle in g.Little.items():
        mean_length.append(fmean([bhandle.length for bhandle in nbrs.bonds(phandle)]))
        particle_phi.append(epu.embryo_phi(phandle))
    
    _bond_lengths_ax.plot(particle_phi, mean_length, "bo")

def _save_graph_bondlengths_v_phi() -> None:
    if _timestep % 1000 == 0:
        bond_lengths_path: str = os.path.join(_plot_path, f"Particle mean bond lengths vs. phi, T {_timestep}.png")
        _bond_lengths_fig.savefig(bond_lengths_path, transparent=False, bbox_inches="tight")

def _init_bond_counts() -> None:
    global _bond_count_fig, _bond_count_ax

    _bond_count_fig, _bond_count_ax = plt.subplots()
    _bond_count_ax.set_ylabel("Mean bonded neighbors per particle")
    
def _show_bond_counts() -> None:
    phandle: tf.ParticleHandle
    
    # logical: it's the mean of how many neighbors each particle has:
    mean_bonds_per_particle: float = fmean([len(phandle.bonded_neighbors) for phandle in g.Little.items()])
    
    # better & faster: it's twice the ratio of bonds to particles. (Have to include leading edge if doing it this way.)
    mean_bonds_per_particle: float = (2 * len(tf.BondHandle.items()) /
                                      (len(g.Little.items()) + len(g.LeadingEdge.items())))
    _bonds_per_particle.append(mean_bonds_per_particle)
    
    _bond_count_ax.plot(_timesteps, _bonds_per_particle, "bo")
    
def _save_bond_counts() -> None:
    bond_count_path: str = os.path.join(_plot_path, "Mean bonds per particle")
    _bond_count_fig.savefig(bond_count_path, transparent=False, bbox_inches="tight")

def show_graphs(end: bool = False) -> None:
    global _timestep
    
    if g.LeadingEdge.items()[0].frozen_z:
        # During the z-frozen phase of equilibration, don't even increment _timestep, so that once
        # we start graphing, it will start at time 0, representing the moment when the leading edge
        # becomes free to move.
        return

    if not _progress_fig:
        # if init hasn't been run yet, run it
        _init_graphs()

    # Don't need to add to the graph every timestep.
    if _timestep % 100 == 0 or end:
        phi: float = round(epu.leading_edge_mean_phi(), 4)
        print(f"Appending: {_timestep}, {phi}")
        _timesteps.append(_timestep)
        _phi.append(phi)
        
        # ToDo? In windowless, technically we don't need to do this until once, at the end, just before
        #  saving the plot. Test for that? Would that improve performance, since it would avoid rendering?
        #  (In HPC? When executing manually?) Of course, need this for windowed mode, for live-updating plot.
        _progress_ax.plot(_timesteps, _phi, "bo")
        
        # _show_test_energy_v_distance()
        _show_test_bondlengths_v_phi()
        _show_bond_counts()
        
        # Go ahead and save every time we add to the plots. That way even in windowless mode, we can
        # monitor the plots as they update.
        _save_graphs(end)

    _timestep += 1
    
def _save_graphs(end: bool = False) -> None:
    if _progress_fig:
        # i.e., only if init_graph() was ever run
        filename: str = f"{_plot_num}. "
        if end:
            filename += f"End. Timestep = {_timestep - 1}; "
        filename += f"Cortical tension = {cfg.yolk_cortical_tension}; external force = {cfg.external_force}"
        filename += ".png"
        filepath: str = os.path.join(_plot_path, filename)
        _progress_fig.savefig(filepath, transparent=False, bbox_inches="tight")
        
        # _save_graph_energy_v_distance()
        _save_graph_bondlengths_v_phi()
        _save_bond_counts()
        
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
            "bond_counts": _bonds_per_particle,
            "phi": _phi,
            "timesteps": _timesteps,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved.
    
    Increment _plot_num with each run to generate a new filename, hence separate plot
    """
    global _plot_num, _timestep, _bonds_per_particle, _phi, _timesteps
    _plot_num = d["plotnum"] + 1
    _timestep = d["timestep"]
    _bonds_per_particle = d["bond_counts"]
    _phi = d["phi"]
    _timesteps = d["timesteps"]
    
# At module import: set to interactive mode ("ion" = "interactive on") so that plot display isn't blocking.
# Note to self: do I need to make sure interactive is off, when I'm in windowless mode? That would be
# necessary for true automation, but would be nice to run windowless manually and still see the plots.
# However, it seems like TF is suppressing that; in windowless only, the plots aren't showing up once
# I've called this function.
plt.ion()
