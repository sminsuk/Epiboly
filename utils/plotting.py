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

from itertools import chain
import numpy as np
import os
from statistics import fmean

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import tissue_forge as tf
import epiboly_globals as g

import biology.microtubules as mt
import config as cfg
import utils.epiboly_utils as epu
import utils.tf_utils as tfu

_leading_edge_phi: list[float] = []
_bonds_per_particle: list[float] = []
_forces: list[float] = []
_straightness: list[float] = []
_margin_count: list[int] = []
_margin_cum_in: list[int] = []
_margin_cum_out: list[int] = []
_margin_cum_divide: list[int] = []

# Note: I considered trying to share bin_axis and timestep histories over all quantities being binned over phi. I think
# I could do it, minimize code duplication, and save on memory and disk space. However, it would also lock me into
# graphing all such quantities over the SAME plotting interval, which I would have regretted. So, duplicate
# these structures (and the code that generates them) for each graph.
_tension_bin_axis_history: list[list[float]] = []
_median_tensions_history: list[list[float]] = []
_tension_timestep_history: list[int] = []

_undivided_tensions_bin_axis_history: list[list[float]] = []
_undivided_tensions_history: list[list[float]] = []
_undivided_tensions_timestep_history: list[int] = []
_divided_tensions_bin_axis_history: list[list[float]] = []
_divided_tensions_history: list[list[float]] = []
_divided_tensions_timestep_history: list[int] = []

_speeds_bin_axis_history: list[list[float]] = []
_median_speeds_history: list[list[float]] = []
# Name this strain rate variable after the algorithm it's using, which is the difference between the speed bin values.
# We'll also calculate it elsewhere by a different algorithm.
_strain_rates_by_speed_diffs_history: list[list[float]] = []
_speeds_timestep_history: list[int] = []
_speeds: list[float] = []
_speeds_particle_phi: list[float] = []

_strain_rate_bin_axis_history: list[list[float]] = []
_median_normal_strain_rates_history: list[list[float]] = []
_median_polar_strain_rates_history: list[list[float]] = []
_median_circumf_strain_rates_history: list[list[float]] = []
_strain_rate_timestep_history: list[int] = []
_normal_strain_rates: list[float] = []
_polar_strain_rates: list[float] = []
_circumf_strain_rates: list[float] = []
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

def _expand_limits_if_needed(limits: tuple[float, float], data: list) -> tuple[float, float]:
    """Test whether data exceeds the plotting limits, and expand the limits to accommodate. But never shrink them.
    
    limits: ylim values for the plot when well behaved
    data: list[float], or list[list[float]], representing all points that will be plotted.
    return: revised ylim values for the plot
    
    Final timestep may be extreme, gets saved as a separate plot, and doesn't need its scale to be consistent
    between different sim configurations (e.g. with vs without cell division), so don't constrain it
    from expanding from what's normally used. But do constrain it from shrinking.
    """
    low_lim, high_lim = limits

    # flatten 2-d data to 1-d so we can take max, min
    # (Seems to be best and fastest way:
    # https://www.leocon.dev/blog/2021/09/how-to-flatten-a-python-list-array-and-which-one-should-you-use/
    # Note that input array can be ragged, making numpy approaches more difficult.)
    flat_data: list
    if isinstance(data[0], list):
        flat_data = list(chain.from_iterable(data))
    else:
        flat_data = data
        
    data_min: float = min(flat_data)
    data_max: float = max(flat_data)
    
    # On each end (top and bottom), use the passed-in limit unless the data exceed it
    plot_min: float = min(low_lim, data_min)
    plot_max: float = max(high_lim, data_max)
    
    # Prevent expanded plot from touching the bounding box, so that it's obvious the data is entirely contained.
    margin: float = (plot_max - plot_min) / 50
    
    if plot_min < low_lim:
        plot_min -= margin
    if plot_max > high_lim:
        plot_max += margin
        
    return plot_min, plot_max

def _plot_data_history(values_history: list[list[float]],
                       bin_axis_history: list[list[float]],
                       timestep_history: list[int],
                       filename: str,
                       secondary_values_history: list[list[float]] = None,
                       secondary_bin_axis_history: list[list[float]] = None,
                       xlabel: str = None,
                       ylabel: str = None,
                       ylim: tuple[float, float] = None,
                       axvline: float = None,
                       axhline: float = None,
                       legend_loc: str = None,
                       end_legend_loc: str = None,
                       end: bool = False) -> None:
    """Plot the history of binned data over multiple time points. If a secondary set is passed, plot those dotted.

    Required parameters:
    history lists: Global storage, preserving all data drawn over multiple time points.
    filename: where to save the plot image. Do not include file extension.

    Optional parameters:
    secondary history lists: to plot two datasets together, the secondary with dots. Both sets use the same
        set of timepoints, so no need to pass a secondary timestep history. But bin_axis is needed because
        primary and secondary can differ. Both secondary history lists must be present if either is.
    xlabel, ylabel: x & y axis labels
    ylim: lower and upper y axis bounds. (xlim is assumed to be (0, pi).)
          y axis scale is left unconstrained if ylim is None, or if end == True.
    axvline, axhline: x and y positions of optional vertical and horizontal lines, respectively.
    legend locations: if None, legend will be unconstrained, i.e. will use the default, which is "best" location.
    end: whether this represents the final plot at the last timestep, which is treated specially.
    """
    binned_values_fig: Figure
    binned_values_ax: Axes
    
    # All the timesteps on one plot, but all of them re-plotted from scratch each time
    binned_values_fig, binned_values_ax = plt.subplots()
    
    secondary: bool = (secondary_values_history is not None and
                       secondary_bin_axis_history is not None)
    if secondary:
        # We will want to double each color in the color cycle so that the primary and secondary values
        # at a given timestep will have the same color.
        # Also rotate the first color to the back because we'll be skipping T=0.
        # (The latter really ought to be controlled by the caller with a flag parameter rather than here. In
        # the current usage, secondary will be empty at T=0 and cause problems, so better to skip it. But that
        # shouldn't be part of this general function, but of the specific call.)
        default_cycle = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        rotated_cycle = default_cycle[1:]
        rotated_cycle.append(default_cycle[0])
        doubled_cycle = [color for color in rotated_cycle for _ in range(2)]
        binned_values_ax.set_prop_cycle(color=doubled_cycle)
    
    if xlabel is not None:
        binned_values_ax.set_xlabel(xlabel)
    if axvline is not None:
        binned_values_ax.axvline(x=axvline, linestyle=":", color="k", linewidth=0.5)
    binned_values_ax.set_xlim(0, np.pi)
    binned_values_ax.set_xticks([0, np.pi / 2, np.pi], labels=["0", "π/2", "π"])
    if ylabel is not None:
        binned_values_ax.set_ylabel(ylabel)
    if axhline is not None:
        binned_values_ax.axhline(y=axhline, linestyle=":", color="k", linewidth=0.5)
    if ylim is not None:
        if end:
            if secondary:
                ylim = _expand_limits_if_needed(ylim, values_history + secondary_values_history)
            else:
                ylim = _expand_limits_if_needed(ylim, values_history)
        binned_values_ax.set_ylim(*ylim)
    
    # plot the entire history
    for i, median_values in enumerate(values_history):
        bin_axis: list[float] = bin_axis_history[i]
        timestep: int = timestep_history[i]
        binned_values_ax.plot(bin_axis, median_values, "-", label=f"T = {timestep}")
        if secondary:
            # Dotted this time, should be same color. Skip the label - too much legend, makes too busy.
            # While the bin boundaries are the same, the two bin_axis variables can be different because
            # the content is different and each variable excludes empty bins, which can occur.
            secondary_median_values: list[float] = secondary_values_history[i]
            secondary_bin_axis: list[float] = secondary_bin_axis_history[i]
            binned_values_ax.plot(secondary_bin_axis, secondary_median_values, ":")

    if end:
        if end_legend_loc is None:
            binned_values_ax.legend()  # unconstrained
        else:
            binned_values_ax.legend(loc=end_legend_loc)
    else:
        if legend_loc is None:
            binned_values_ax.legend()
        else:
            binned_values_ax.legend(loc=legend_loc)
    
    # save
    # On final timestep, use a different filename, so I get two saved versions: with and without the final plot
    suffix: str = " (with final timestep)" if end else ""
    path = os.path.join(_plot_path, f"{filename}{suffix}.png")
    binned_values_fig.savefig(path, transparent=False, bbox_inches="tight")
    plt.close(binned_values_fig)

def _add_binned_medians_to_history(values: list[float],
                                   positions: list[float],
                                   values_history: list[list[float]],
                                   bin_axis_history: list[list[float]],
                                   timestep_history: list[int],
                                   first: bool = True,
                                   last: bool = True,
                                   approx_bin_size: float = np.pi / 6) -> float:
    """From list of data points and positions, generate bins and median values, and add to history
    
    values, positions: data for a single plotted line on the graph. May be only from the current timestep,
        or global storage with data accumulated over multiple timesteps for time averaging.
    history lists: Global storage, preserving lines previously drawn. Current timestep data will be added to it.
    approx_bin_size: width of bins on the x axis (delta phi); actual width will be adjusted to fit evenly into the data.
    first, last: These flags are used to indicate that this function is being called multiple times with related data
        sets, which all share the same positions (and hence the same bin_axis_history). When making a singular
        call, use the defaults of both = True (it's the only call, so it's both the first and last call).
        When making a series of calls for related data, set first = True only on the first call, and False
        everywhere else; and set last = True only on the last call, and False everywhere else. The bin_axis_history
        and timestep_history will only be appended to history on the first call. The raw accumulated positions
        data for time averaging, which needs to be reset for future use, will only be cleared on the last call.
    
    returns: actual bin size used for the current timepoint, as adjusted from approx_bin_size
    """
    assert values and positions, "Attempting to bin an empty dataset!"
    
    # bin the data and calculate the median for each bin
    np_values = np.array(values)
    np_positions = np.array(positions)
    
    # How many bins? A constant bin size resulted in a partially full final bin, depending on epiboly progress
    # at each timestep. To ensure the final bin has a large enough sample of particles to generate a valid median,
    # calculate a bin size that fits an integer number of times into the range of the data. But also, to use roughly
    # the same size bins at each time step, let the number of bins vary each timestep, accordingly.
    max_phi: float = epu.leading_edge_max_phi()
    num_bins: int = round(max_phi / approx_bin_size)
    bin_edges: np.ndarray = np.linspace(0.0, max_phi, num_bins + 1)
    actual_bin_size: float = bin_edges[1] - bin_edges[0]
    bin_indices: np.ndarray = np.digitize(np_positions, bin_edges)
    
    # Note: numpy ufunc equality and masking!
    # https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
    bins: list[np.ndarray] = [np_values[bin_indices == i] for i in range(1, bin_edges.size)]
    binn: np.ndarray
    median_values: list[float] = []
    bin_axis: list[float] = []
    for i, binn in enumerate(bins):
        if binn.size > 0:
            # np.median() returns ndarray but is really float because binn is 1d
            median_values.append(np.median(binn).item())
            bin_axis.append(bin_edges[i])
    
    # Add to history. We will re-plot the entire thing.
    values_history.append(median_values)
    if first:
        bin_axis_history.append(bin_axis)
        timestep_history.append(_timestep)
    
    # Now delete the raw data (multi-timestep data accumulation) so that these global lists can be reused
    # later. In a series of multiple calls, values is a different list each time, so always clear it.
    values.clear()
    if last:
        # In a series of multiple calls, positions is reused each time, so only clear it after the last call.
        positions.clear()

    return actual_bin_size

def _show_test_tension_v_phi(end: bool) -> None:
    """Plot mean tension of all bonds on a particle, vs. phi of the particle;
    
    and then bin the values and plot the median tension for each bin.
    """
    tensions_fig: Figure
    tensions_ax: Axes
    
    # Init the plots from scratch every single time
    # This one is just a single-timestep plot
    tensions_fig, tensions_ax = plt.subplots()
    tensions_ax.set_xlabel("Particle phi")
    tensions_ax.set_xlim(0, np.pi)
    tensions_ax.set_ylabel("Particle tension\n(mean bond displacement from equilibrium)")
    tensions_ax.axhline(y=0, linestyle=":", color="k", linewidth=0.5)  # tension/compression boundary
    tensions_ax.text(0.02, 0.97, f"T={_timestep}", transform=tensions_ax.transAxes,
                     verticalalignment="top", horizontalalignment="left",
                     fontsize=28, fontweight="bold")
    
    phandle: tf.ParticleHandle
    tensions: list[float] = []
    particle_phi: list[float] = []
    for phandle in g.Little.items():
        tensions.append(tfu.strain(phandle))
        particle_phi.append(epu.embryo_phi(phandle))
    
    # plot
    ylim: tuple[float, float] = (-0.05, 0.25)
    if end:
        ylim = _expand_limits_if_needed(ylim, tensions)
    tensions_ax.set_ylim(*ylim)
    tensions_ax.plot(particle_phi, tensions, "b.")

    # save
    tensions_path: str = os.path.join(_plot_path, f"Particle tensions vs. phi, T {_timestep}.png")
    tensions_fig.savefig(tensions_path, transparent=False, bbox_inches="tight")
    plt.close(tensions_fig)

    # That was the raw data, now let's bin it and plot its median
    _add_binned_medians_to_history(tensions,
                                   particle_phi,
                                   _median_tensions_history,
                                   _tension_bin_axis_history,
                                   _tension_timestep_history)
    
    _plot_data_history(_median_tensions_history,
                       _tension_bin_axis_history,
                       _tension_timestep_history,
                       filename="Aggregate tension vs. phi, multiple timepoints",
                       xlabel=r"Particle position $\phi$",
                       ylabel="Median particle tension",
                       ylim=(-0.05, 0.20),
                       axhline=0,  # compression/tension boundary
                       # legend_loc="lower right" if cfg.force_algorithm is cfg.ForceAlgorithm.CONSTANT else None,
                       # end_legend_loc="upper left" if cfg.force_algorithm is cfg.ForceAlgorithm.CONSTANT else None,
                       end=end)

def _show_tension_by_cell_size(end: bool) -> None:
    """Plot median tensions of undivided and divided cells separately"""
    if not cfg.cell_division_enabled:
        # Plot this only if cell division is happening
        return
    
    phandle: tf.ParticleHandle
    undivided_tensions: list[float] = []
    undivided_particle_phi: list[float] = []
    divided_tensions: list[float] = []
    divided_particle_phi: list[float] = []
    for phandle in g.Little.items():
        if epu.is_undivided(phandle):
            undivided_tensions.append(tfu.strain(phandle))
            undivided_particle_phi.append(epu.embryo_phi(phandle))
        else:
            divided_tensions.append(tfu.strain(phandle))
            divided_particle_phi.append(epu.embryo_phi(phandle))
            
    if len(divided_tensions) == 0:
        # If there are no divided cells yet, just skip this. Avoid handling that case.
        # So at T=0, no plot at all; at later times, this plot won't have a T=0 line.
        return
        
    # Bin the raw data and plot its median
    _add_binned_medians_to_history(undivided_tensions,
                                   undivided_particle_phi,
                                   _undivided_tensions_history,
                                   _undivided_tensions_bin_axis_history,
                                   _undivided_tensions_timestep_history)
    
    _add_binned_medians_to_history(divided_tensions,
                                   divided_particle_phi,
                                   _divided_tensions_history,
                                   _divided_tensions_bin_axis_history,
                                   _divided_tensions_timestep_history)
    
    _plot_data_history(_undivided_tensions_history,
                       _undivided_tensions_bin_axis_history,
                       _undivided_tensions_timestep_history,
                       secondary_values_history=_divided_tensions_history,
                       secondary_bin_axis_history=_divided_tensions_bin_axis_history,
                       filename="Tensions by cell size",
                       xlabel=r"Particle position $\phi$",
                       ylabel="Median particle tension",
                       ylim=(-0.05, 0.20),
                       axhline=0,  # compression/tension boundary
                       # legend_loc="lower right" if cfg.force_algorithm is cfg.ForceAlgorithm.CONSTANT else None,
                       # end_legend_loc="upper left" if cfg.force_algorithm is cfg.ForceAlgorithm.CONSTANT else None,
                       end=end)

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
        return particle_position_phi, tfu.signed_scalar_from_vector_projection(velocity, tangent_vec)

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

    approximate_bin_size: float = np.pi / 6
    actual_bin_size: float = _add_binned_medians_to_history(_speeds,
                                                            _speeds_particle_phi,
                                                            _median_speeds_history,
                                                            _speeds_bin_axis_history,
                                                            _speeds_timestep_history,
                                                            approx_bin_size=approximate_bin_size)
    # ToDo? Alternative ways to plot? Time in the x axis, and phi as the different colored lines!
    #  Or maybe even, phi vs time, with velocity displayed as a heatmap???
    # ToDo? Should bin size be based on number of particles (or height, which is proportional to
    #  surface area) rather than on phi???

    # Latex: magnitude (double vertical bar) of the vector v-sub-veg, the vegetal component of velocity
    ylabel: str = r"Median $\Vert\mathbf{v_{veg}}\Vert$"

    _plot_data_history(_median_speeds_history,
                       _speeds_bin_axis_history,
                       _speeds_timestep_history,
                       filename="PIV - speed vs. phi, multiple timepoints",
                       xlabel=r"Particle position $\phi$",
                       ylabel=ylabel,
                       ylim=(-0.006, 0.09),
                       axhline=0,        # vegetalward/animalward boundary
                       legend_loc="upper left",
                       end_legend_loc="upper left",
                       end=end)
    
    # Now generate another plot, for strain rates based on the median speed values, which were appended to the
    # speeds history when the speed plot was generated:
    median_speeds: list[float] = _median_speeds_history[-1]

    # From the aggregate speed of each bin, calculate strain rate = difference from previous bin, which we'll plot
    # separately. (And correct for actual_bin_size, which is constant within a time point, but not between time points;
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
    _strain_rates_by_speed_diffs_history.append(strain_rates)

    # Since we skipped the first bin on the values, we have to do the same on the positions.
    # This is true for every timepoint in the history.
    truncated_bin_axis_history: list[list[float]] = [bin_axis[1:] for bin_axis in _speeds_bin_axis_history]
    
    _plot_data_history(_strain_rates_by_speed_diffs_history,
                       truncated_bin_axis_history,
                       _speeds_timestep_history,
                       filename="Strain rates by speed bin diffs",
                       xlabel=r"Particle position $\phi$",
                       ylabel="Strain rate (speed-bin differences)",
                       ylim=(-0.005, 0.04),
                       axvline=np.pi/2,  # equator
                       axhline=0,        # stretch/compression boundary
                       # legend_loc="lower right" if cfg.force_algorithm is cfg.ForceAlgorithm.CONSTANT else None,
                       # end_legend_loc="upper left" if cfg.force_algorithm is cfg.ForceAlgorithm.CONSTANT else None,
                       end=end)

def _show_strain_rates_v_phi(finished_accumulating: bool, end: bool) -> None:
    """Plot binned median strain-rates for all bonded particle pairs, vs. phi

    phi of the bond determined as mean of the phi of the two bonded particles
    """
    def phi_and_strain_rates(bhandle: tf.BondHandle) -> tuple[float, float, float, float]:
        p1: tf.ParticleHandle
        p2: tf.ParticleHandle
        p1, p2 = bhandle.parts
        theta1, phi1 = epu.embryo_coords(p1)
        theta2, phi2 = epu.embryo_coords(p2)
        
        # The average theta is easier to get this way than by just averaging the two thetas,
        # because then you'd have to deal with the case of two thetas at opposite ends of the
        # range, i.e. across a discontinuous numerical boundary.
        midpoint: tf.fVector3 = (p1.position + p2.position) / 2
        theta, phi = epu.embryo_coords(midpoint)
        
        vegetalward_phi: float = phi + np.pi / 2
        eastward_theta: float = theta + np.pi / 2

        strain_rate_vec: tf.fVector3 = p2.velocity - p1.velocity
        normal_direction: tf.fVector3 = p2.position - p1.position
        vegetalward: tf.fVector3 = tfu.cartesian_from_spherical([1, theta, vegetalward_phi])
        eastward: tf.fVector3 = tfu.cartesian_from_spherical([1, eastward_theta, np.pi/2])
        
        # Particle positions could theoretically be identical, so normal_direction would be the zero vector,
        # and projecting onto it would give NaN. But in practice that's never going to happen.
        normal_strain_rate: float = tfu.signed_scalar_from_vector_projection(strain_rate_vec,
                                                                             direction=normal_direction)
        
        # In contrast, fairly frequently, the difference between the 2 phi values or the 2 theta values is zero,
        # which would result in the usual calculation of polar_direction and circumf_direction being the zero vector,
        # and projecting onto those would give NaN. So we need to handle those special cases. But in those cases
        # the strain rate is definitely nonnegative (can't compress when distance along those respective directions
        # is already zero), so we just need the simpler calculation that ignores the sign of the vector projection.
        if phi1 == phi2:
            polar_strain_rate: float = tfu.unsigned_scalar_from_vector_projection(strain_rate_vec,
                                                                                  direction=vegetalward)
        else:
            polar_direction: tf.fVector3 = vegetalward * np.sign(phi2 - phi1)
            polar_strain_rate: float = tfu.signed_scalar_from_vector_projection(strain_rate_vec,
                                                                                direction=polar_direction)
        if theta1 == theta2:
            circumf_strain_rate: float = tfu.unsigned_scalar_from_vector_projection(strain_rate_vec,
                                                                                    direction=eastward)
        else:
            circumf_direction: tf.fVector3 = eastward * np.sign(tfu.corrected_theta(theta2 - theta1))
            circumf_strain_rate: float = tfu.signed_scalar_from_vector_projection(strain_rate_vec,
                                                                                  direction=circumf_direction)
        
        return phi, normal_strain_rate, polar_strain_rate, circumf_strain_rate

    if end:
        # Normally we've been accumulating into these lists over multiple timesteps, so we just continue to add to them.
        # But if end, we'll keep things simple by dumping earlier data (if any) and gathering just the current
        # data and plotting it (so, not time averaged as usual).
        _normal_strain_rates.clear()
        _polar_strain_rates.clear()
        _circumf_strain_rates.clear()
        _strain_rate_bond_phi.clear()

    # Calculate strain rate for each bonded particle pair, along with its position
    bhandle: tf.BondHandle
    for bhandle in tf.BondHandle.items():
        bond_position_phi, normal_strain_rate, polar_strain_rate, circumf_strain_rate = phi_and_strain_rates(bhandle)
        _normal_strain_rates.append(normal_strain_rate)
        _polar_strain_rates.append(polar_strain_rate)
        _circumf_strain_rates.append(circumf_strain_rate)
        _strain_rate_bond_phi.append(bond_position_phi)

    if not finished_accumulating and not end:
        # accumulate more timesteps before plotting
        return

    # For all three sets of data (normal, polar, circumferential), generate binned median values and store to history,
    # storing the results from each data set in its respective history list. The bin_axis_history and timestep_history
    # will only be generated once, in the first of the three calls (first=True).
    arg_sets = [(_normal_strain_rates, _median_normal_strain_rates_history, True, False),
                (_polar_strain_rates, _median_polar_strain_rates_history, False, False),
                (_circumf_strain_rates, _median_circumf_strain_rates_history, False, True)]
    for values, values_history, first, last in arg_sets:
        _add_binned_medians_to_history(values, _strain_rate_bond_phi,
                                       values_history,
                                       _strain_rate_bin_axis_history,
                                       _strain_rate_timestep_history,
                                       first=first,
                                       last=last)
    
    # Now plot all three graphs, the same way.
    arg_sets = [("normal", _median_normal_strain_rates_history, (-0.02, 0.01), "upper right"),
                ("polar", _median_polar_strain_rates_history, (-0.009, 0.01), None),
                ("circumferential", _median_circumf_strain_rates_history, (-0.018, 0.006), "upper right")]
    for direction, values_history, ylim, legend_loc in arg_sets:
        _plot_data_history(values_history,
                           _strain_rate_bin_axis_history,
                           _strain_rate_timestep_history,
                           filename=f"{direction.capitalize()} strain rates",
                           xlabel=r"Particle position $\phi$",
                           ylabel=f"Median {direction} strain rate",
                           ylim=ylim,
                           axvline=np.pi/2,  # equator
                           axhline=0,        # stretch/compression boundary
                           # legend_loc=legend_loc if cfg.force_algorithm is cfg.ForceAlgorithm.CONSTANT else None,
                           end=end)

def _show_straightness() -> None:
    """Plot straightness index of the leading edge

    Straightness index = "D/L, where D is the beeline distance between the first and last points in the trajectory,
    and L is the path length travelled. It is a value between 0 (infinitely tortuous) to 1 (a straight line)."
    https://search.r-project.org/CRAN/refmans/trajr/html/TrajStraightness.html
    """
    straightness_fig: Figure
    straightness_ax: Axes
    
    straightness_fig, straightness_ax = plt.subplots()
    straightness_ax.set_ylabel("Straightness Index")
    
    # Sort all the edge particles on theta, into a new list (copy, not live)
    sorted_particles: list[tf.ParticleHandle] = sorted(g.LeadingEdge.items(), key=epu.embryo_theta)
    
    # Calculate the path distance from particle to particle
    p: tf.ParticleHandle
    previous_p: tf.ParticleHandle = sorted_particles[-1]  # last one
    path_length: float = 0
    for p in sorted_particles:
        path_length += p.distance(previous_p)
        previous_p = p
    
    # Beeline is not the circumference, it's a polygon. Project each particle along a longitude line
    # to a circle located at the mean phi position. That's the polygon to measure
    r: float = epu.embryo_radius()
    beeline_phi: float = epu.leading_edge_mean_phi()
    beeline_positions: list[tf.fVector3] = [tfu.cartesian_from_spherical([r, epu.embryo_theta(p), beeline_phi])
                                            for p in sorted_particles]
    
    # Calculate the path distance from position to position in the beeline
    pos: tf.fVector3
    previous_pos: tf.fVector3 = beeline_positions[-1]
    beeline_path_length: float = 0
    for pos in beeline_positions:
        vector_diff: tf.fVector3 = pos - previous_pos
        beeline_path_length += vector_diff.length()
        previous_pos = pos
    
    straightness: float = beeline_path_length / path_length
    _straightness.append(straightness)
    
    # ToDo: do this vs phi (or better, %ep) so that it's easier to see "when" it gets straight. Epiboly position = time!
    # 0.90 is usually good enough for bottom, but if it dips below that, get the whole plot in frame
    limits: tuple[float, float] = _expand_limits_if_needed(limits=(0.90, 1.001), data=_straightness)
    straightness_ax.set_ylim(limits)
    straightness_ax.plot(_timesteps, _straightness, ".-b")
    
    # save
    straightness_path: str = os.path.join(_plot_path, "Straightness Index of EVL margin.png")
    straightness_fig.savefig(straightness_path, transparent=False, bbox_inches="tight")
    plt.close(straightness_fig)
    
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
    bond_count_path: str = os.path.join(_plot_path, "Mean bond count per particle.png")
    bond_count_fig.savefig(bond_count_path, transparent=False, bbox_inches="tight")
    plt.close(bond_count_fig)

def _show_margin_population() -> None:
    # ToDo: this needs to be with respect to phi, not time, because especially for non-cell-division,
    #  the simulation slows down which distorts the curve.
    margin_fig: Figure
    margin_ax: Axes
    
    margin_fig, margin_ax = plt.subplots()
    margin_ax.set_ylabel("Number of margin cells")
    
    _margin_count.append(len(g.LeadingEdge.items()))
    _margin_cum_in.append(epu.cumulative_to_edge)
    _margin_cum_out.append(epu.cumulative_from_edge)
    if cfg.cell_division_enabled:
        _margin_cum_divide.append(epu.cumulative_edge_divisions)
    
    # plot just the total
    margin_ax.plot(_timesteps, _margin_count, ".b", label="Total margin cell count")

    # save
    margin_path: str = os.path.join(_plot_path, "Margin cell rearrangement.png")
    margin_fig.savefig(margin_path, transparent=False, bbox_inches="tight")
    
    # add additional lines to the plot and resave under a different name
    margin_ax.plot(_timesteps, _margin_cum_in, "--b", label="Cumulative in")
    margin_ax.plot(_timesteps, _margin_cum_out, ":b", label="Cumulative out")
    if cfg.cell_division_enabled:
        margin_ax.plot(_timesteps, _margin_cum_divide, "-b", label="Cumulative divisions")
    margin_ax.legend(loc="center right")

    margin_path = os.path.join(_plot_path, "Margin cell rearrangement, plus cumulative.png")
    margin_fig.savefig(margin_path, transparent=False, bbox_inches="tight")
    plt.close(margin_fig)

def _show_forces() -> None:
    if cfg.force_algorithm is cfg.ForceAlgorithm.CONSTANT:
        # Nothing terribly interesting to plot!
        return
    
    forces_fig: Figure
    forces_ax: Axes

    forces_fig, forces_ax = plt.subplots()
    forces_ax.set_ylabel("Forces")
    forces_ax.axhline(y=cfg.yolk_cortical_tension + cfg.external_force, linestyle=":", color="k", linewidth=0.5)

    _forces.append(mt.current_total_force())
    
    # plot
    forces_ax.plot(_timesteps, _forces, "b.")
    
    # save
    forces_path: str = os.path.join(_plot_path, "Forces on leading edge.png")
    forces_fig.savefig(forces_path, transparent=False, bbox_inches="tight")
    plt.close(forces_fig)

def _show_progress_graph() -> None:
    progress_fig: Figure
    progress_ax: Axes
    
    progress_fig, progress_ax = plt.subplots()
    progress_ax.set_ylabel(r"Leading edge  $\bar{\phi}$  (radians)")

    phi: float = round(epu.leading_edge_mean_phi(), 4)
    print(f"Appending: {_timestep}, {phi}")
    _timesteps.append(_timestep)
    _leading_edge_phi.append(phi)

    # Plot
    progress_ax.plot(_timesteps, _leading_edge_phi, "b.")

    # Go ahead and save every time we add to the plot. That way even in windowless mode, we can
    # monitor the plot as it updates.
    filename: str = f"Leading edge phi"
    filepath: str = os.path.join(_plot_path, filename + ".png")
    progress_fig.savefig(filepath, transparent=False, bbox_inches="tight")
    plt.close(progress_fig)

def show_graphs(end: bool = False) -> None:
    global _timestep
    
    if not g.LeadingEdge.items() or g.LeadingEdge.items()[0].frozen_z:
        # Prior to instantiating any LeadingEdge particles, or during the z-frozen phase of equilibration,
        # don't even increment _timestep, so that once we start graphing, it will start at time 0, representing the
        # moment when the leading edge is both present, and free to move. (Frozen/unfrozen state, and pre-/post-
        # instantiation, refer to the initialization methods by config and by graph boundary, respectively.)
        return

    if not _plot_path:
        # if init hasn't been run yet, run it
        _init_graphs()

    # Don't need to add to the graphs every timestep.
    simtime_interval: float = 4
    timestep_interval: int = round(simtime_interval / cfg.dt)
    if _timestep % timestep_interval == 0 or end:
        _show_progress_graph()
        _show_bond_counts()
        _show_forces()
        _show_straightness()
        _show_margin_population()

    plot_interval: int = cfg.plotting_interval_timesteps
    
    if _timestep % plot_interval == 0 or end:
        # These aggregate graphs don't need to be time-averaged, so just call them exactly on the interval (including 0)
        _show_test_tension_v_phi(end)
        _show_tension_by_cell_size(end)

    if not cfg.plot_time_averages:
        if _timestep % plot_interval == 0 or end:
            _show_piv_speed_v_phi(True, end)
            _show_strain_rates_v_phi(True, end)
    else:
        if _timestep > 0 or cfg.plot_t0_as_single_timestep:
            # These aggregate graphs need to be time-averaged. Averaging at simulation start is optional,
            # according to the config flag. If doing that, don't plot at exactly 0, because we need to average
            # the timesteps AFTER that. And the logic below needs to skip over 0 to work. We want remainder == 0
            # at the end of the accumulation period, not at the beginning. (At later timesteps, we time-average
            # BEFORE the time point, e.g., get all the data from steps (5000 - [num steps]) through 5000.)
            # If not averaging at simulation start, plot T = 0 as a single timestep.
            
            if _timestep <= cfg.time_avg_accumulation_steps and not cfg.plot_t0_as_single_timestep:
                # Special case, when time-averaging the beginning of the simulation, so that we time-average AFTER T=0,
                # i.e, get all the data from steps 1 through [num steps].
                plot_interval = cfg.time_avg_accumulation_steps
            
            remainder: int = _timestep % plot_interval
            # During accumulation phase for the time point, go accumulate data but don't plot anything yet;
            # when the time to plot arrives, go accumulate that final time point's data, time-average it all, and plot.
            if (remainder == 0
                    or remainder > plot_interval - cfg.time_avg_accumulation_steps
                    or end):
                _show_piv_speed_v_phi(remainder == 0, end)
                _show_strain_rates_v_phi(remainder == 0, end)
        
    _timestep += 1
    
def get_state() -> dict:
    """In composite runs, save incomplete plot data so those plots can be completed with cumulative data, all back to 0
    
    This applies to plots that accumulate over the life of the sim, like the progress plot and the median
    tensions plot. Each run saves the plot image to disk, but the data is saved as part of the state,
    so the next run can import it and overwrite the saved image, all the way from Timestep 0.
    """
    return {"timestep": _timestep,
            "bond_counts": _bonds_per_particle,
            "leading_edge_phi": _leading_edge_phi,
            "forces": _forces,
            "straightness": _straightness,
            "margin_count": _margin_count,
            "margin_cum_in": _margin_cum_in,
            "margin_cum_out": _margin_cum_out,
            "margin_cum_divide": _margin_cum_divide,
            "timesteps": _timesteps,
            
            "tension_bin_axis_history": _tension_bin_axis_history,
            "median_tensions_history": _median_tensions_history,
            "tension_timestep_history": _tension_timestep_history,
            
            "undivided_tensions_bin_axis_history": _undivided_tensions_bin_axis_history,
            "undivided_tensions_history": _undivided_tensions_history,
            "undivided_tensions_timestep_history": _undivided_tensions_timestep_history,
            "divided_tensions_bin_axis_history": _divided_tensions_bin_axis_history,
            "divided_tensions_history": _divided_tensions_history,
            "divided_tensions_timestep_history": _divided_tensions_timestep_history,
            
            "speeds_bin_axis_history": _speeds_bin_axis_history,
            "median_speeds_history": _median_speeds_history,
            "strain_rates_by_speed_diffs_history": _strain_rates_by_speed_diffs_history,
            "speeds_timestep_history": _speeds_timestep_history,
            "speeds": _speeds,
            "speeds_particle_phi": _speeds_particle_phi,
            
            "strain_rate_bin_axis_history": _strain_rate_bin_axis_history,
            "median_normal_strain_rates_history": _median_normal_strain_rates_history,
            "median_polar_strain_rates_history": _median_polar_strain_rates_history,
            "median_circumf_strain_rates_history": _median_circumf_strain_rates_history,
            "strain_rate_timestep_history": _strain_rate_timestep_history,
            "normal_strain_rates": _normal_strain_rates,
            "polar_strain_rates": _polar_strain_rates,
            "circumf_strain_rates": _circumf_strain_rates,
            "strain_rate_bond_phi": _strain_rate_bond_phi,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global _timestep, _bonds_per_particle, _leading_edge_phi, _forces, _straightness, _timesteps
    global _margin_count, _margin_cum_in, _margin_cum_out, _margin_cum_divide
    global _tension_bin_axis_history, _median_tensions_history, _tension_timestep_history
    global _undivided_tensions_bin_axis_history, _undivided_tensions_history, _undivided_tensions_timestep_history
    global _divided_tensions_bin_axis_history, _divided_tensions_history, _divided_tensions_timestep_history
    global _speeds_bin_axis_history, _median_speeds_history, _speeds_timestep_history
    global _strain_rates_by_speed_diffs_history
    global _speeds, _speeds_particle_phi
    global _strain_rate_bin_axis_history, _median_normal_strain_rates_history, _strain_rate_timestep_history
    global _median_polar_strain_rates_history, _median_circumf_strain_rates_history
    global _normal_strain_rates, _polar_strain_rates, _circumf_strain_rates, _strain_rate_bond_phi
    _timestep = d["timestep"]
    _bonds_per_particle = d["bond_counts"]
    _leading_edge_phi = d["leading_edge_phi"]
    _forces = d["forces"]
    _straightness = d["straightness"]
    _margin_count = d["margin_count"]
    _margin_cum_in = d["margin_cum_in"]
    _margin_cum_out = d["margin_cum_out"]
    _margin_cum_divide = d["margin_cum_divide"]
    _timesteps = d["timesteps"]
    
    _tension_bin_axis_history = d["tension_bin_axis_history"]
    _median_tensions_history = d["median_tensions_history"]
    _tension_timestep_history = d["tension_timestep_history"]
    
    _undivided_tensions_bin_axis_history = d["undivided_tensions_bin_axis_history"]
    _undivided_tensions_history = d["undivided_tensions_history"]
    _undivided_tensions_timestep_history = d["undivided_tensions_timestep_history"]
    _divided_tensions_bin_axis_history = d["divided_tensions_bin_axis_history"]
    _divided_tensions_history = d["divided_tensions_history"]
    _divided_tensions_timestep_history = d["divided_tensions_timestep_history"]
    
    _speeds_bin_axis_history = d["speeds_bin_axis_history"]
    _median_speeds_history = d["median_speeds_history"]
    _strain_rates_by_speed_diffs_history = d["strain_rates_by_speed_diffs_history"]
    _speeds_timestep_history = d["speeds_timestep_history"]
    _speeds = d["speeds"]
    _speeds_particle_phi = d["speeds_particle_phi"]
    
    _strain_rate_bin_axis_history = d["strain_rate_bin_axis_history"]
    _median_normal_strain_rates_history = d["median_normal_strain_rates_history"]
    _median_polar_strain_rates_history = d["median_polar_strain_rates_history"]
    _median_circumf_strain_rates_history = d["median_circumf_strain_rates_history"]
    _strain_rate_timestep_history = d["strain_rate_timestep_history"]
    _normal_strain_rates = d["normal_strain_rates"]
    _polar_strain_rates = d["polar_strain_rates"]
    _circumf_strain_rates = d["circumf_strain_rates"]
    _strain_rate_bond_phi = d["strain_rate_bond_phi"]
