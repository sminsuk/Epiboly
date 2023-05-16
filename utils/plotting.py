"""plotting.py - matplotlib plots

Note to self: These StackOverflow articles made it seem like getting the plot to be non-blocking
and continually update, live, was going to be very hard. Strangely enough, once I followed the MPL
docs and simply turned on plt.ion(), everything just worked. I won't rule out nasty surprises in the
future if I change something, though.

https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib

On the other hand, after increasing the number of things I'm potting, it just made a mess (in windowed mode).
So, turned it off. (Plus it doesn't seem to work in TF windowless mode anyway; TF seems to suppress it.)
Furthermore, once I switched all my plots to local and properly closed each one after local use, ion() doesn't
really work anymore anyway, even in windowed mode. (I could set any I want to update live, back to global
Figure/Axes as they were, and leave those select ones open, but it's not really worth it.) Turns out it's easy
enough to track them "live" as they plot – since I save frequently – just by opening the files they save to.
"""

import numpy as np
import os
from statistics import fmean

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
# graphing all such quantities over the SAME plotting interval, which I would have regretted. So, duplicate
# these structures (and the code that generates them) for each graph.
_combo_tension_bin_axis_history: list[list[float]] = []
_combo_median_tensions_history: list[list[float]] = []
_combo_tension_timestep_history: list[int] = []

_combo_speeds_bin_axis_history: list[list[float]] = []
_combo_median_speeds_history: list[list[float]] = []
# Name this strain rate variable after the algorithm it's using, which is the difference between the speed bin values.
# We'll also calculate it elsewhere by a different algorithm.
_combo_strain_rates_by_speed_diffs_history: list[list[float]] = []
_combo_speeds_timestep_history: list[int] = []
_speeds: list[float] = []
_speeds_particle_phi: list[float] = []

_combo_strain_rate_bin_axis_history: list[list[float]] = []
_combo_median_strain_rates_history: list[list[float]] = []
_combo_strain_rate_timestep_history: list[int] = []
_strain_rates: list[float] = []
_strain_rate_bond_phi: list[float] = []

_timesteps: list[int] = []
_timestep: int = 0
_plot_path: str = ""

def _init_graphs() -> None:
    """Initialize matplotlib and also a subdirectory in which to put the saved plots
    
    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _plot_path
    
    _plot_path = os.path.join(tfu.export_path(), "Plots")
    os.makedirs(_plot_path, exist_ok=True)

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
    median_tensions: list[float] = []
    bin_axis: list[float] = []
    for i, binn in enumerate(bins):
        if binn.size > 0:
            # np.median() returns ndarray but is really float because binn is 1d
            median_tensions.append(np.median(binn).item())
            bin_axis.append(bin_edges[i])
            
    # Add to history so we will re-plot the ENTIRE history.
    _combo_median_tensions_history.append(median_tensions)
    _combo_tension_bin_axis_history.append(bin_axis)
    _combo_tension_timestep_history.append(_timestep)
    
    # plot
    for i, median_tensions in enumerate(_combo_median_tensions_history):
        bin_axis = _combo_tension_bin_axis_history[i]
        timestep: int = _combo_tension_timestep_history[i]
        combo_tensions_binned_ax.plot(bin_axis, median_tensions, "-", label=f"T = {timestep}")
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

def _show_piv_speed_v_phi(finished_accumulating: bool, end: bool) -> None:
    """Particle Image Velocimetry - or the one aspect of it that's relevant in this context
    
    Embryo is cylindrically symmetrical. We just want to know the magnitude of the vegetally-pointing
    component of the velocity, as a function of phi and time.
    
    Further, generate strain rates from these values, so plot those here as well.
    """
    def phi_and_vegetal_speed(phandle: tf.ParticleHandle) -> tuple[float, float]:
        theta, particle_position_phi = epu.embryo_coords(phandle)
        tangent_phi: float = particle_position_phi + np.pi / 2
        tangent_vec: tf.fVector3 = tfu.cartesian_from_spherical([1, theta, tangent_phi])
        velocity: tf.fVector3 = phandle.velocity
        veg_component: tf.fVector3 = velocity.projectedOntoNormalized(tangent_vec)
        return particle_position_phi, veg_component.length()

    if end:
        # Normally we've been accumulating into these lists over multiple timesteps, so we just continue to add to them.
        # But if end, we'll keep things simple by dumping earlier data (if any) and gathering just the current
        # data and plotting it (so, not time averaged as usual).
        _speeds.clear()
        _speeds_particle_phi.clear()

    phandle: tf.ParticleHandle
    for phandle in g.Little.items():
        particle_position_phi, speed = phi_and_vegetal_speed(phandle)
        _speeds.append(speed)
        _speeds_particle_phi.append(particle_position_phi)
    for phandle in g.LeadingEdge.items():
        particle_position_phi, speed = phi_and_vegetal_speed(phandle)
        _speeds.append(speed)
        _speeds_particle_phi.append(particle_position_phi)

    if not finished_accumulating and not end:
        # accumulate more timesteps before plotting
        return

    combo_speeds_binned_fig: Figure
    combo_speeds_binned_ax: Axes

    # All the timesteps on one plot, but all of them re-plotted from scratch each time
    combo_speeds_binned_fig, combo_speeds_binned_ax = plt.subplots()
    combo_speeds_binned_ax.set_xlabel(r"Particle position $\phi$")
    combo_speeds_binned_ax.set_xlim(0, np.pi)
    combo_speeds_binned_ax.set_xticks([0, np.pi / 2, np.pi], labels=["0", "π/2", "π"])
    # magnitude (double vertical bar) of the vector v-sub-veg, the vegetal component of velocity
    combo_speeds_binned_ax.set_ylabel(r"Median $\Vert\mathbf{v_{veg}}\Vert$")
    if not end:
        # Final timestep will go way beyond this ylim value, so don't constrain it.
        combo_speeds_binned_ax.set_ylim(0.0, 0.15)
    # ToDo: But actually, this should be time in the x axis, and phi in the colors! Or maybe even, phi vs time,
    #  with velocity displayed as a heatmap???

    # bin it and plot its median
    np_speeds = np.array(_speeds)
    np_particle_phi = np.array(_speeds_particle_phi)

    # How many bins? See explanation in _show_test_tension_v_phi(). (But in this case, include LeadingEdge particles.)
    # ToDo: Should bin size be based on number of particles rather than on phi???
    max_phi: float = epu.leading_edge_max_phi()
    approximate_bin_size = np.pi / 10
    num_bins: int = round(max_phi / approximate_bin_size)
    bin_edges: np.ndarray = np.linspace(0.0, max_phi, num_bins + 1)
    actual_bin_size: float = bin_edges[1] - bin_edges[0]
    bin_indices: np.ndarray = np.digitize(np_particle_phi, bin_edges)

    # Note: numpy ufunc equality and masking!
    # https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
    bins: list[np.ndarray] = [np_speeds[bin_indices == i] for i in range(1, bin_edges.size)]
    binn: np.ndarray
    median_speeds: list[float] = []
    bin_axis: list[float] = []
    for i, binn in enumerate(bins):
        if binn.size > 0:
            # np.median() returns ndarray but is really float because binn is 1d
            median_speeds.append(np.median(binn).item())
            bin_axis.append(bin_edges[i])

    # Add to history so we will re-plot the ENTIRE history.
    _combo_median_speeds_history.append(median_speeds)
    _combo_speeds_bin_axis_history.append(bin_axis)
    _combo_speeds_timestep_history.append(_timestep)

    # And now we can delete the raw data so that the lists can be reused later
    _speeds.clear()
    _speeds_particle_phi.clear()

    # plot
    for i, history_median_speeds in enumerate(_combo_median_speeds_history):
        bin_axis = _combo_speeds_bin_axis_history[i]
        timestep: int = _combo_speeds_timestep_history[i]
        combo_speeds_binned_ax.plot(bin_axis, history_median_speeds, "-", label=f"T = {timestep}")
    combo_speeds_binned_ax.legend(loc="upper left")

    # save
    # On final timestep, use a different filename, so I get two saved versions: with and without the final plot
    suffix: str = " (with final timestep)" if end else ""
    combo_path: str = os.path.join(_plot_path, f"PIV - speed vs. phi, multiple timepoints{suffix}.png")
    combo_speeds_binned_fig.savefig(combo_path, transparent=False, bbox_inches="tight")
    plt.close(combo_speeds_binned_fig)
    
    # Now generate another plot, for strain rates based on these median values
    
    combo_strain_rates_binned_fig: Figure
    combo_strain_rates_binned_ax: Axes

    combo_strain_rates_binned_fig, combo_strain_rates_binned_ax = plt.subplots()
    combo_strain_rates_binned_ax.set_xlabel(r"Particle position $\phi$")
    combo_strain_rates_binned_ax.axvline(x=np.pi/2, linestyle=":", color="k", linewidth=0.5)  # equator
    combo_strain_rates_binned_ax.set_xlim(0, np.pi)
    combo_strain_rates_binned_ax.set_xticks([0, np.pi / 2, np.pi], labels=["0", "π/2", "π"])
    combo_strain_rates_binned_ax.set_ylabel("Strain rate (speed-bin differences)")
    combo_strain_rates_binned_ax.axhline(y=0, linestyle=":", color="k", linewidth=0.5)  # stretch/compression boundary
    if not end:
        combo_strain_rates_binned_ax.set_ylim(-0.011, 0.032)

    # From the aggregate speed of each bin, calculate strain rate = difference from previous bin, which we'll plot
    # separately. (And divide by actual_bin_size, which is constant within the time point, but not between time points;
    # so they need to be made comparable.) For the first bin, strain rate is not really valid, so don't plot it at all.)
    previous_speed = None
    strain_rates: list[float] = []
    for speed in median_speeds:
        if previous_speed is not None:
            # Skip calculating strain rate for the first bin, because there's nothing valid to subtract.
            # And normalize for the bin size so the values will be comparable between time points.
            strain_rates.append((speed - previous_speed) * approximate_bin_size / actual_bin_size)
        previous_speed = speed

    # Add this to history as well
    _combo_strain_rates_by_speed_diffs_history.append(strain_rates)

    # plot the ENTIRE history
    for i, strain_rates in enumerate(_combo_strain_rates_by_speed_diffs_history):
        # From each bin axis, take all but the first item, because we omitted strain rate for that bin
        bin_axis: list[float] = _combo_speeds_bin_axis_history[i][1:]
        timestep: int = _combo_speeds_timestep_history[i]
        combo_strain_rates_binned_ax.plot(bin_axis, strain_rates, "-", label=f"T = {timestep}")
    combo_strain_rates_binned_ax.legend(loc="upper left" if end else "lower right")
    
    combo_path = os.path.join(_plot_path, f"Strain rates by speed bin diffs{suffix}.png")
    combo_strain_rates_binned_fig.savefig(combo_path, transparent=False, bbox_inches="tight")
    plt.close(combo_strain_rates_binned_fig)

def _show_strain_rates_v_phi(finished_accumulating: bool, end: bool) -> None:
    """Plot binned median strain-rates for all bonded particle pairs, vs. phi

    phi of the bond determined as mean of the phi of the two bonded particles
    """
    def signed_scalar_from_vector_projection(vector: tf.fVector3, direction: tf.fVector3) -> float:
        """Return the magnitude of the vector projection, but preserve the sign"""
        projection: tf.fVector3 = vector.projected(direction)
        return projection.length() * np.sign(vector.dot(direction))

    def phi_and_strain_rate(bhandle: tf.BondHandle) -> tuple[float, float]:
        p1: tf.ParticleHandle
        p2: tf.ParticleHandle
        p1, p2 = bhandle.parts
        strain_rate_vec: tf.fVector3 = p1.velocity - p2.velocity
        bond_direction: tf.fVector3 = p1.position - p2.position
        normal_strain_rate: float = signed_scalar_from_vector_projection(strain_rate_vec, direction=bond_direction)
        phi: float = (epu.embryo_phi(p1) + epu.embryo_phi(p2)) / 2
        return phi, normal_strain_rate

    if end:
        # Normally we've been accumulating into these lists over multiple timesteps, so we just continue to add to them.
        # But if end, we'll keep things simple by dumping earlier data (if any) and gathering just the current
        # data and plotting it (so, not time averaged as usual).
        _strain_rates.clear()
        _strain_rate_bond_phi.clear()

    # Calculate strain rate for each bonded particle pair, along with its position
    bhandle: tf.BondHandle
    for bhandle in tf.BondHandle.items():
        bond_position_phi, strain_rate = phi_and_strain_rate(bhandle)
        _strain_rates.append(strain_rate)
        _strain_rate_bond_phi.append(bond_position_phi)

    if not finished_accumulating and not end:
        # accumulate more timesteps before plotting
        return

    combo_strain_rates_binned_fig: Figure
    combo_strain_rates_binned_ax: Axes
    
    # ##### For debugging, get the resulting values and sort them in both directions
    # so I can more easily observe the larger values
    sorted_strain_rates = sorted(_strain_rates)
    reversed_strain_rates = reversed(sorted_strain_rates)
    
    # All the timesteps on one plot, but all of them re-plotted from scratch each time
    combo_strain_rates_binned_fig, combo_strain_rates_binned_ax = plt.subplots()
    combo_strain_rates_binned_ax.set_xlabel(r"Particle position $\phi$")
    combo_strain_rates_binned_ax.axvline(x=np.pi/2, linestyle=":", color="k", linewidth=0.5)  # equator
    combo_strain_rates_binned_ax.set_xlim(0, np.pi)
    combo_strain_rates_binned_ax.set_xticks([0, np.pi / 2, np.pi], labels=["0", "π/2", "π"])
    combo_strain_rates_binned_ax.set_ylabel("Median strain rate (bonded particle-pairs)")
    combo_strain_rates_binned_ax.axhline(y=0, linestyle=":", color="k", linewidth=0.5)  # stretch/compression boundary
    if not end:
        # Final timestep will go way beyond this ylim value, so don't constrain it.
        combo_strain_rates_binned_ax.set_ylim(-0.015, 0.01)

    # bin it and plot its median
    np_strain_rates = np.array(_strain_rates)
    np_bond_phi = np.array(_strain_rate_bond_phi)

    # How many bins? See explanation in _show_test_tension_v_phi().
    max_phi: float = epu.leading_edge_max_phi()
    approximate_bin_size = np.pi / 20
    num_bins: int = round(max_phi / approximate_bin_size)
    bin_edges: np.ndarray = np.linspace(0.0, max_phi, num_bins + 1)
    bin_indices: np.ndarray = np.digitize(np_bond_phi, bin_edges)

    bins: list[np.ndarray] = [np_strain_rates[bin_indices == i] for i in range(1, bin_edges.size)]
    binn: np.ndarray
    median_strain_rates: list[float] = []
    bin_axis: list[float] = []
    for i, binn in enumerate(bins):
        if binn.size > 0:
            # np.median() returns ndarray but is really float because binn is 1d
            median_strain_rates.append(np.median(binn).item())
            bin_axis.append(bin_edges[i])

    # Add to history so we will re-plot the ENTIRE history.
    _combo_median_strain_rates_history.append(median_strain_rates)
    _combo_strain_rate_bin_axis_history.append(bin_axis)
    _combo_strain_rate_timestep_history.append(_timestep)
    
    # And now we can delete the raw data so that the lists can be reused later
    _strain_rates.clear()
    _strain_rate_bond_phi.clear()

    # plot
    for i, median_strain_rates in enumerate(_combo_median_strain_rates_history):
        bin_axis: list[float] = _combo_strain_rate_bin_axis_history[i]
        timestep: int = _combo_strain_rate_timestep_history[i]
        combo_strain_rates_binned_ax.plot(bin_axis, median_strain_rates, "-", label=f"T = {timestep}")
    if end:
        combo_strain_rates_binned_ax.legend()
    else:
        combo_strain_rates_binned_ax.legend(loc="upper right")

    # save
    # On final timestep, use a different filename, so I get two saved versions: with and without the final plot
    suffix: str = " (with final timestep)" if end else ""
    combo_path = os.path.join(_plot_path, f"Strain rates by particle pair{suffix}.png")
    combo_strain_rates_binned_fig.savefig(combo_path, transparent=False, bbox_inches="tight")
    plt.close(combo_strain_rates_binned_fig)

def _show_bond_counts() -> None:
    bond_count_fig: Figure
    bond_count_ax: Axes
    
    bond_count_fig, bond_count_ax = plt.subplots()
    bond_count_ax.set_ylabel("Mean bonded neighbors per particle")

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
    bond_count_ax.plot(_timesteps, _bonds_per_particle, "b.")
    
    # save
    bond_count_path: str = os.path.join(_plot_path, "Mean bond count per particle")
    bond_count_fig.savefig(bond_count_path, transparent=False, bbox_inches="tight")
    plt.close(bond_count_fig)

def _show_progress_graph(end: bool) -> None:
    progress_fig: Figure
    progress_ax: Axes
    
    progress_fig, progress_ax = plt.subplots()
    progress_ax.set_ylabel(r"Leading edge  $\bar{\phi}$  (radians)")

    phi: float = round(epu.leading_edge_mean_phi(), 4)
    print(f"Appending: {_timestep}, {phi}")
    _timesteps.append(_timestep)
    _leading_edge_phi.append(phi)

    # ToDo? In windowless, technically we don't need to do this until once, at the end, just before
    #  saving the plot. Test for that? Would that improve performance, since it would avoid rendering?
    #  (In HPC? When executing manually?) Of course, need this for windowed mode, for live-updating plot.
    # Plot
    progress_ax.plot(_timesteps, _leading_edge_phi, "b.")

    # Go ahead and save every time we add to the plot. That way even in windowless mode, we can
    # monitor the plot as it updates.
    filename: str = f"Leading edge phi"
    filepath: str = os.path.join(_plot_path, filename + ".png")
    progress_fig.savefig(filepath, transparent=False, bbox_inches="tight")
    plt.close(progress_fig)

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
        _show_bond_counts()

    # Call the aggregate plots less frequently when cell division disabled; otherwise so many lines get drawn (because
    # the no-division sim lasts twice as many timesteps) that the legends get too tall for the graph.
    plot_interval: int = 1000 if cfg.cell_division_enabled else 2000
    
    if _timestep % plot_interval == 0 or end:
        # These aggregate graphs don't need to be time-averaged, so just call them exactly on the interval (including 0)
        _show_test_tension_v_phi(end)

    if _timestep > 0:
        # These aggregate graphs need to be time-averaged. Don't do them at exactly 0, because we need to
        # average the timesteps AFTER that. And the logic below needs to skip over 0 to work. We want
        # remainder == 0 at the end of the accumulation period, not at the beginning. (Later, we time-average
        # BEFORE the time point, e.g., get all the data from steps (5000 - [num steps]) through 5000.)
        
        time_avg_accumulation_steps: int = 200
        if _timestep <= time_avg_accumulation_steps:
            # Special case so that at the beginning of the sim, we time-average AFTER T=0,
            # i.e, get all the data from steps 1 through [num steps].
            plot_interval = time_avg_accumulation_steps
        
        remainder: int = _timestep % plot_interval
        # If within accumulation_steps of the time point to be plotted, go accumulate data but don't plot anything yet;
        # when the time to plot arrives, go accumulate that final time point's data, time-average it all, and plot.
        if (remainder == 0
                or remainder > plot_interval - time_avg_accumulation_steps
                or end):
            _show_piv_speed_v_phi(remainder == 0, end)
            _show_strain_rates_v_phi(remainder == 0, end)
        
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
            "combo_strain_rates_by_speed_diffs_history": _combo_strain_rates_by_speed_diffs_history,
            "combo_speeds_timestep_history": _combo_speeds_timestep_history,
            "speeds": _speeds,
            "speeds_particle_phi": _speeds_particle_phi,
            
            "combo_strain_rate_bin_axis_history": _combo_strain_rate_bin_axis_history,
            "combo_median_strain_rates_history": _combo_median_strain_rates_history,
            "combo_strain_rate_timestep_history": _combo_strain_rate_timestep_history,
            "strain_rates": _strain_rates,
            "strain_rate_bond_phi": _strain_rate_bond_phi,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global _timestep, _bonds_per_particle, _leading_edge_phi, _timesteps
    global _combo_tension_bin_axis_history, _combo_median_tensions_history, _combo_tension_timestep_history
    global _combo_speeds_bin_axis_history, _combo_median_speeds_history, _combo_speeds_timestep_history
    global _combo_strain_rates_by_speed_diffs_history
    global _speeds, _speeds_particle_phi
    global _combo_strain_rate_bin_axis_history, _combo_median_strain_rates_history, _combo_strain_rate_timestep_history
    global _strain_rates, _strain_rate_bond_phi
    _timestep = d["timestep"]
    _bonds_per_particle = d["bond_counts"]
    _leading_edge_phi = d["leading_edge_phi"]
    _timesteps = d["timesteps"]
    
    _combo_tension_bin_axis_history = d["combo_tension_bin_axis_history"]
    _combo_median_tensions_history = d["combo_median_tensions_history"]
    _combo_tension_timestep_history = d["combo_tension_timestep_history"]
    
    _combo_speeds_bin_axis_history = d["combo_speeds_bin_axis_history"]
    _combo_median_speeds_history = d["combo_median_speeds_history"]
    _combo_strain_rates_by_speed_diffs_history = d["combo_strain_rates_by_speed_diffs_history"]
    _combo_speeds_timestep_history = d["combo_speeds_timestep_history"]
    _speeds = d["speeds"]
    _speeds_particle_phi = d["speeds_particle_phi"]
    
    _combo_strain_rate_bin_axis_history = d["combo_strain_rate_bin_axis_history"]
    _combo_median_strain_rates_history = d["combo_median_strain_rates_history"]
    _combo_strain_rate_timestep_history = d["combo_strain_rate_timestep_history"]
    _strain_rates = d["strain_rates"]
    _strain_rate_bond_phi = d["strain_rate_bond_phi"]
