"""plotting.py - matplotlib plots

Note to self: These StackOverflow articles made it seem like getting the plot to be non-blocking
and continually update, live, was going to be very hard. Strangely enough, once I followed the MPL
docs and simply turned on plt.ion(), everything just worked. I won't rule out nasty surprises in the
future if I change something, though.

https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib

On the other hand, after increasing the number of things I'm potting, it just made a mess (in windowed mode).
So, turning it off for now. Turns out it's easy enough to track them "live" as they plot – since I save frequently –
just by opening the files they save to. (Plus it doesn't seem to work in TF windowless mode anyway; TF
seems to suppress it.)
"""

import numpy as np
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

_leading_edge_phi: list[float] = []
_bonds_per_particle: list[float] = []

# Note: I considered trying to share bin_axis and timestep histories over all quantities being binned over phi. I think
# I could do it, minimize code duplication, and save on memory and disk space. However, it would also lock me into
# graphing all such quantities over the SAME plotting interval, and I might regret that. So, for now, I'm duplicating
# these structures (and the code that generates them) for each graph even though the values will be the same.
_combo_tension_bin_axis_history: list[list[float]] = []
_combo_median_tensions_history: list[list] = []
_combo_tension_timestep_history: list[int] = []
_combo_speeds_bin_axis_history: list[list[float]] = []
_combo_median_speeds_history: list[list] = []
_combo_speeds_timestep_history: list[int] = []

_timesteps: list[int] = []
_timestep: int = 0
_progress_fig: Optional[Figure] = None
_progress_ax: Optional[Axes] = None
_energy_fig: Optional[Figure] = None
_energy_ax: Optional[Axes] = None
_potentials_fig: Optional[Figure] = None
_potentials_ax: Optional[Axes] = None
_bond_count_fig: Optional[Figure] = None
_bond_count_ax: Optional[Axes] = None
_plot_path: str = ""

def _init_graphs() -> None:
    """Initialize matplotlib and also a subdirectory in which to put the saved plots
    
    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _plot_path
    
    _plot_path = os.path.join(tfu.export_path(), "Plots")
    os.makedirs(_plot_path, exist_ok=True)

def _show_test_energy_v_distance() -> None:
    """Plot energy vs length, & potential vs. length (this is to see if they are the same)"""
    global _energy_fig, _energy_ax, _potentials_fig, _potentials_ax
    
    if not _energy_fig:
        # Init only once
        _energy_fig, _energy_ax = plt.subplots()
        _energy_ax.set_xlabel("particle distance")
        _energy_ax.set_ylabel("bond energy")
        _potentials_fig, _potentials_ax = plt.subplots()
        _potentials_ax.set_xlabel("particle distance")
        _potentials_ax.set_ylabel("bond potential")

    energy: list[float] = []
    potentials: list[float] = []
    distance: list[float] = []
    bhandle: tf.BondHandle
    
    for bhandle in tf.BondHandle.items()[:100]:
        distance.append(bhandle.length)
        energy.append(bhandle.energy)
        potentials.append(bhandle.potential(bhandle.length))

    # plot
    _energy_ax.plot(distance, energy, "b.")
    _potentials_ax.plot(distance, potentials, "r.")

    # save
    energypath: str = os.path.join(_plot_path, "Energy vs. bond distance.png")
    _energy_fig.savefig(energypath, transparent=False, bbox_inches="tight")
    potentialpath: str = os.path.join(_plot_path, "Potential vs. bond distance.png")
    _potentials_fig.savefig(potentialpath, transparent=False, bbox_inches="tight")

def _show_test_tension_v_phi(end: bool) -> None:
    """Plot mean tension of all bonds on a particle, vs. phi of the particle;
    
    and then bin the values and plot the median tension for each bin.
    """
    tensions_fig: Figure
    tensions_ax: Axes
    combo_tensions_binned_fig: Figure
    combo_tensions_binned_ax: Axes
    
    # Init the plots from scratch every single time
    # This one is just a single-timestep plot
    tensions_fig, tensions_ax = plt.subplots()
    tensions_ax.set_xlabel("Particle phi")
    tensions_ax.set_xlim(0, np.pi)
    tensions_ax.set_ylabel("Particle tension\n(mean bond displacement from equilibrium)")
    if not end:
        # Final timestep will go way beyond this ylim value, so don't constrain it.
        tensions_ax.set_ylim(0.0, 0.35)
    tensions_ax.text(0.02, 0.97, f"T={_timestep}", transform=tensions_ax.transAxes,
                     verticalalignment="top", horizontalalignment="left",
                     fontsize=28, fontweight="bold")

    # This one is all the timesteps on one plot, but all of them re-plotted from scratch each time
    combo_tensions_binned_fig, combo_tensions_binned_ax = plt.subplots()
    combo_tensions_binned_ax.set_xlabel(r"Particle position $\phi$")
    combo_tensions_binned_ax.set_xlim(0, np.pi)
    combo_tensions_binned_ax.set_xticks([0, np.pi / 2, np.pi], labels=["0", "π/2", "π"])
    combo_tensions_binned_ax.set_ylabel("Median particle tension")
    if not end:
        # Final timestep will go way beyond this ylim value, so don't constrain it.
        combo_tensions_binned_ax.set_ylim(0.0, 0.25)
    
    bhandle: tf.BondHandle
    phandle: tf.ParticleHandle
    neighbor: tf.ParticleHandle
    tensions: list[float] = []
    particle_phi: list[float] = []
    for phandle in g.Little.items():
        tensions.append(fmean([max(0, bhandle.length - bhandle.potential.r0)
                              for bhandle in nbrs.bonds(phandle)]))
        particle_phi.append(epu.embryo_phi(phandle))
    
    # plot
    tensions_ax.plot(particle_phi, tensions, "b.")

    # save
    tensions_path: str = os.path.join(_plot_path, f"Particle tensions vs. phi, T {_timestep}.png")
    tensions_fig.savefig(tensions_path, transparent=False, bbox_inches="tight")
    plt.close(tensions_fig)

    # That was the raw data, now let's bin it and plot its median
    np_tensions = np.array(tensions)
    np_particle_phi = np.array(particle_phi)
    
    # How many bins? A constant bin size resulted in a partially full final bin, depending on epiboly progress
    # at each timestep. To ensure the final bin has a large enough sample of particles to generate a valid median,
    # calculate a bin size that fits an integer number of times into the range of the data. But also, to use roughly
    # the same size bins at each time step, let the number of bins vary each timestep, accordingly.
    max_phi: float = epu.internal_evl_max_phi()
    approximate_bin_size = np.pi / 20
    num_bins: int = round(max_phi / approximate_bin_size)
    bin_edges: np.ndarray = np.linspace(0.0, max_phi, num_bins + 1)
    bin_indices: np.ndarray = np.digitize(np_particle_phi, bin_edges)

    # Note: numpy ufunc equality and masking!
    # https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
    bins: list[np.ndarray] = [np_tensions[bin_indices == i] for i in range(1, bin_edges.size)]
    binn: np.ndarray
    medians = []
    bin_axis: list[float] = []
    for i, binn in enumerate(bins):
        if binn.size > 0:
            medians.append(np.median(binn))     # np.median() returns ndarray but is really float because binn is 1d
            bin_axis.append(bin_edges[i])
            
    # Add to history so we will re-plot the ENTIRE history.
    _combo_median_tensions_history.append(medians)
    _combo_tension_bin_axis_history.append(bin_axis)
    _combo_tension_timestep_history.append(_timestep)
    
    # plot
    for i, medians in enumerate(_combo_median_tensions_history):
        bin_axis = _combo_tension_bin_axis_history[i]
        timestep: int = _combo_tension_timestep_history[i]
        combo_tensions_binned_ax.plot(bin_axis, medians, "-", label=f"T = {timestep}")
    combo_tensions_binned_ax.legend(loc="upper left")
    
    # Then plot the T=0 line again, without a legend this time since the legend is already there. That way its
    # plot can be in front, since it tends to get covered over by all the other lines when it's in back (when cell
    # division is enabled). Must specify color 0 in the color cycle so it matches the legend for the first plot!
    combo_tensions_binned_ax.plot(_combo_tension_bin_axis_history[0], _combo_median_tensions_history[0], "C0-")
    
    # save
    # On final timestep, use a different filename, so I get two saved versions: with and without the final plot
    suffix: str = " (with final timestep)" if end else ""
    combo_path: str = os.path.join(_plot_path, f"Aggregate tension vs. phi, multiple timepoints{suffix}.png")
    combo_tensions_binned_fig.savefig(combo_path, transparent=False, bbox_inches="tight")
    plt.close(combo_tensions_binned_fig)

def _show_piv_speed_v_phi(end: bool) -> None:
    """Particle Image Velocimetry - or the one aspect of it that's relevant in this context
    
    Embryo is cylindrically symmetrical. We just want to know the magnitude of the vegetally-pointing
    component of the velocity, as a function of phi and time.
    """
    combo_speeds_binned_fig: Figure
    combo_speeds_binned_ax: Axes

    # All the timesteps on one plot, but all of them re-plotted from scratch each time
    combo_speeds_binned_fig, combo_speeds_binned_ax = plt.subplots()
    combo_speeds_binned_ax.set_xlabel(r"Particle position $\phi$")
    combo_speeds_binned_ax.set_xlim(0, np.pi)
    combo_speeds_binned_ax.set_xticks([0, np.pi / 2, np.pi], labels=["0", "π/2", "π"])
    # magnitude (double vertical bar) of the vector v-sub-veg, the vegetal component of velocity
    combo_speeds_binned_ax.set_ylabel(r"Median $\|\mathbf{v_{veg}}\|$")
    if not end:
        # Final timestep will go way beyond this ylim value, so don't constrain it.
        combo_speeds_binned_ax.set_ylim(0.0, 0.15)
    # ToDo: But actually, this should be time in the x axis, and phi in the colors! Or maybe even, phi vs time,
    #  with velocity displayed as a heatmap???

    def phi_and_vegetal_speed(phandle: tf.ParticleHandle) -> tuple[float, float]:
        theta, particle_position_phi = epu.embryo_coords(phandle)
        tangent_phi: float = particle_position_phi + np.pi/2
        tangent_vec: tf.fVector3 = tfu.cartesian_from_spherical([1, theta, tangent_phi])
        velocity: tf.fVector3 = phandle.velocity
        # Note: why does TF have both .projected(), and this? Is this faster?
        veg_component: tf.fVector3 = velocity.projectedOntoNormalized(tangent_vec)
        return particle_position_phi, veg_component.length()
        
    phandle: tf.ParticleHandle
    speeds: list[float] = []
    particle_phi: list[float] = []
    for phandle in g.Little.items():
        particle_position_phi, speed = phi_and_vegetal_speed(phandle)
        speeds.append(speed)
        particle_phi.append(particle_position_phi)
    for phandle in g.LeadingEdge.items():
        particle_position_phi, speed = phi_and_vegetal_speed(phandle)
        speeds.append(speed)
        particle_phi.append(particle_position_phi)

    # bin it and plot its median
    np_speeds = np.array(speeds)
    np_particle_phi = np.array(particle_phi)

    # How many bins? See explanation in _show_test_tension_v_phi(). (But in this case, include LeadingEdge particles.)
    # ToDo: Should bin size be based on number of particles rather than on phi??? Or for a quicker fix,
    #  just make them bigger?
    max_phi: float = epu.leading_edge_max_phi()
    approximate_bin_size = np.pi / 10
    num_bins: int = round(max_phi / approximate_bin_size)
    bin_edges: np.ndarray = np.linspace(0.0, max_phi, num_bins + 1)
    bin_indices: np.ndarray = np.digitize(np_particle_phi, bin_edges)

    # Note: numpy ufunc equality and masking!
    # https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
    bins: list[np.ndarray] = [np_speeds[bin_indices == i] for i in range(1, bin_edges.size)]
    binn: np.ndarray
    medians = []
    bin_axis: list[float] = []
    for i, binn in enumerate(bins):
        if binn.size > 0:
            medians.append(np.median(binn))  # np.median() returns ndarray but is really float because binn is 1d
            bin_axis.append(bin_edges[i])

    # Add to history so we will re-plot the ENTIRE history.
    _combo_median_speeds_history.append(medians)
    _combo_speeds_bin_axis_history.append(bin_axis)
    _combo_speeds_timestep_history.append(_timestep)

    # plot
    for i, medians in enumerate(_combo_median_speeds_history):
        bin_axis = _combo_speeds_bin_axis_history[i]
        timestep: int = _combo_speeds_timestep_history[i]
        combo_speeds_binned_ax.plot(bin_axis, medians, "-", label=f"T = {timestep}")
    combo_speeds_binned_ax.legend(loc="upper left")

    # save
    # On final timestep, use a different filename, so I get two saved versions: with and without the final plot
    suffix: str = " (with final timestep)" if end else ""
    combo_path: str = os.path.join(_plot_path, f"PIV - speed vs. phi, multiple timepoints{suffix}.png")
    combo_speeds_binned_fig.savefig(combo_path, transparent=False, bbox_inches="tight")
    plt.close(combo_speeds_binned_fig)

def _show_bond_counts() -> None:
    global _bond_count_fig, _bond_count_ax
    
    if not _bond_count_fig:
        # Init only once
        _bond_count_fig, _bond_count_ax = plt.subplots()
        _bond_count_ax.set_ylabel("Mean bonded neighbors per particle")

    phandle: tf.ParticleHandle
    
    # logical: it's the mean of how many neighbors each particle has:
    mean_bonds_per_particle: float = fmean([len(phandle.bonded_neighbors) for phandle in g.Little.items()])
    
    # better & faster: it's twice the ratio of bonds to particles. (Have to include leading edge if doing it this way.)
    # On the other hand, I have not tested to be sure BondHandle.items() isn't affected by the phantom-bond bug,
    # something I probably need ToDo.
    # So save this and maybe use it later:
    # mean_bonds_per_particle: float = (2 * len(tf.BondHandle.items()) /
    #                                   (len(g.Little.items()) + len(g.LeadingEdge.items())))
    
    _bonds_per_particle.append(mean_bonds_per_particle)
    
    # plot
    _bond_count_ax.plot(_timesteps, _bonds_per_particle, "b.")
    
    # save
    bond_count_path: str = os.path.join(_plot_path, "Mean bond count per particle")
    _bond_count_fig.savefig(bond_count_path, transparent=False, bbox_inches="tight")


def _show_progress_graph(end: bool) -> None:
    global _progress_fig, _progress_ax
    
    # Init only once
    if not _progress_fig:
        _progress_fig, _progress_ax = plt.subplots()
        _progress_ax.set_ylabel(r"Leading edge  $\bar{\phi}$  (radians)")

    phi: float = round(epu.leading_edge_mean_phi(), 4)
    print(f"Appending: {_timestep}, {phi}")
    _timesteps.append(_timestep)
    _leading_edge_phi.append(phi)

    # ToDo? In windowless, technically we don't need to do this until once, at the end, just before
    #  saving the plot. Test for that? Would that improve performance, since it would avoid rendering?
    #  (In HPC? When executing manually?) Of course, need this for windowed mode, for live-updating plot.
    # Plot
    _progress_ax.plot(_timesteps, _leading_edge_phi, "b.")

    # Go ahead and save every time we add to the plot. That way even in windowless mode, we can
    # monitor the plot as it updates.
    filename: str = f"Leading edge phi"
    filepath: str = os.path.join(_plot_path, filename + ".png")
    _progress_fig.savefig(filepath, transparent=False, bbox_inches="tight")

    if end:
        suffix: str = f"; Timestep = {_timestep}"
        newfilepath: str = os.path.join(_plot_path, filename + suffix + ".png")
        os.rename(filepath, newfilepath)

def show_graphs(end: bool = False) -> None:
    global _timestep
    
    if g.LeadingEdge.items()[0].frozen_z:
        # During the z-frozen phase of equilibration, don't even increment _timestep, so that once
        # we start graphing, it will start at time 0, representing the moment when the leading edge
        # becomes free to move.
        return

    if not _plot_path:
        # if init hasn't been run yet, run it
        _init_graphs()

    # Don't need to add to the graphs every timestep.
    if _timestep % 100 == 0 or end:
        _show_progress_graph(end)
        # _show_test_energy_v_distance()
        _show_bond_counts()
        
        # Call these less frequently when cell division disabled; otherwise so many lines get drawn (because
        # the no-division sim lasts twice as many timesteps) that the legends get too tall for the graph.
        plot_interval: int = 1000 if cfg.cell_division_enabled else 2000
        if _timestep % plot_interval == 0 or end:
            _show_test_tension_v_phi(end)
            _show_piv_speed_v_phi(end)
        
    _timestep += 1
    
def get_state() -> dict:
    """In composite runs, save incomplete plot data so those plots can be completed with cumulative data, all back to 0
    
    This applies to plots that accumulate over the life of the sim, like the progress plot and the median
    tensions combo plot. Each run saves the plot image to disk, but the data is saved as part of the state,
    so the next run can import it and overwrite the saved image, all the way from Timestep 0.
    """
    return {"timestep": _timestep,
            "bond_counts": _bonds_per_particle,
            "leading_edge_phi": _leading_edge_phi,
            "timesteps": _timesteps,
            "combo_tension_bin_axis_history": _combo_tension_bin_axis_history,
            "combo_median_tensions_history": _combo_median_tensions_history,
            "combo_tension_timestep_history": _combo_tension_timestep_history,
            "combo_speeds_bin_axis_history": _combo_speeds_bin_axis_history,
            "combo_median_speeds_history": _combo_median_speeds_history,
            "combo_speeds_timestep_history": _combo_speeds_timestep_history,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global _timestep, _bonds_per_particle, _leading_edge_phi, _timesteps
    global _combo_tension_bin_axis_history, _combo_median_tensions_history, _combo_tension_timestep_history
    global _combo_speeds_bin_axis_history, _combo_median_speeds_history, _combo_speeds_timestep_history
    _timestep = d["timestep"]
    _bonds_per_particle = d["bond_counts"]
    _leading_edge_phi = d["leading_edge_phi"]
    _timesteps = d["timesteps"]
    _combo_tension_bin_axis_history = d["combo_tension_bin_axis_history"]
    _combo_median_tensions_history = d["combo_median_tensions_history"]
    _combo_tension_timestep_history = d["combo_tension_timestep_history"]
    _combo_speeds_bin_axis_history = d["combo_speeds_bin_axis_history"]
    _combo_median_speeds_history = d["combo_median_speeds_history"]
    _combo_speeds_timestep_history = d["combo_speeds_timestep_history"]
    
# At module import: set to interactive mode ("ion" = "interactive on") so that plot display isn't blocking.
# (Disabled: see module docstring.)
# plt.ion()
