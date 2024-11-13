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
from typing import TypedDict

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
_straightness_cyl: list[float] = []
_margin_lopsidedness: list[float] = []
_margin_count: list[int] = []
_margin_cum_in: list[int] = []
_margin_cum_out: list[int] = []
_margin_cum_divide: list[int] = []
_median_tension_leading_edge: list[float] = []
_median_tension_all: list[float] = []

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

class PlotData(TypedDict, total=False):
    data: list[float] | list[int]   # required
    phi: list[float]                # x axis (any of these 3) required in post-process; ignored in real-time plotting
    timesteps: list[int]
    norm_times: list[float]
    fmt: str                        # not required
    label: object                   # not required; can be str or anything that can be turned into str (float, int...)

def _init_graphs() -> None:
    """Initialize matplotlib and also a subdirectory in which to put the saved plots
    
    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _plot_path
    
    _plot_path = os.path.join(tfu.export_path(), "Plots")
    os.makedirs(_plot_path, exist_ok=True)

def _add_time_axis(axes: Axes, plot_v_time: bool = False, normalize_time: bool = False) -> None:
    """Use leading edge mean phi (i.e. embryonic stage) as a proxy for time (when possible)
    
    Add standard x-axis labeling for epiboly progress (leading edge mean phi) as a proxy for time
    (in other words, essentially plotting vs. embryonic stage). Will be used in multiple plots.
    """
    if plot_v_time:
        # Time axis needs no special labeling
        axes.set_xlabel("Normalized time" if normalize_time else "Timesteps")
        return
    
    axes.set_xlabel(r"Leading edge  $\bar{\phi}$  (radians)")
    axes.set_xlim(np.pi * 7 / 16, np.pi)
    axes.set_xticks([np.pi / 2, np.pi * 5/8, np.pi * 3/4, np.pi * 7/8, np.pi],
                    labels=[r"$\pi$/2", "", r"3$\pi$/4", "", r"$\pi$"])

def _plot_datasets_v_time(datadicts: list[PlotData],
                          filename: str,
                          limits: tuple[float, float] = None,
                          ylabel: str = None,
                          plot_formats: str = None,
                          axvline: float = None,
                          legend_loc: str = None,
                          yticks: dict = None,
                          plot_v_time: bool = None,
                          normalize_time: bool = False,
                          post_process: bool = False) -> None:
    """Plot one or more datasets on a single set of Figure/Axes
    
    When plotting in real time during simulation run, plot_v_time will typically be None, and we'll (at least for now)
    use the balanced-force-control config to decide what x is. (In the control, cannot plot vs. epiboly progress for
    most plots, because EVL will not advance, so we plot vs. time.) But this can be overridden by passing a bool
    value for plot_v_time, and then we'll ignore the config (e.g., for plotting the epiboly progress plot, which is
    always vs. time). When plotting multiple datasets in post-process, basically the same, but we won't want to check
    the flag for each dataset separately, as they all need the same x-axis; so interpret None as False, i.e. caller
    must decide, and pass the x data with each dataset.

    :param datadicts: one or more PlotData, each containing a dataset to plot, plus optional legend label, format
        string, and x axis data (timesteps, or leading_edge_phi). If format string is present, use it and ignore
        plot_formats parameter; if absent, use plot_formats. In real-time plotting, ignore x from the PlotData,
        and get it from the real-time global variables; in post-process plotting, each dataset needs its own
        independent x axis, which is passed in each PlotData.
    :param limits: y-axis limits
    :param ylabel: y-axis label
    :param filename: for the saved image of the plot
    :param plot_formats: the format str for plotting. If a normal format string is passed, use it for all the plots
        (they'll all be the same). If None is passed, then just use "-" and let matplotlib select the colors
        (all different).
    :param axvline: position of vertical line if one is needed. Note that, if done post-process, it should work
        fine on a plot vs. phi as long as the mark is always in the same place (e.g., for cell division cessation,
        phi just reflects initial edge position); but it won't work on a plot v. timesteps, because the position
        will vary among the datasets. If necessary to display that, it would have to by a different means, like
        a special marker on each line in the plot.
    :param legend_loc: optional, specify where legend will go if there is one.
    :param yticks: Currently using this in only 2 spots, ad hoc, not anticipating more. But if I start using this
        more generally, then... ToDo: define a proper typed dict and do parameter validation
    :param plot_v_time: whether to plot vs. time instead of vs. phi. In real-time plotting only, a value of None
        means to decide based on cfg.run_balanced_force_control. In post-process plotting, None is treated as False.
    :param normalize_time: In real-time plotting, ignored. In post-process plotting vs. phi, ignored. In post-process
        plotting vs. time, whether to normalize (time axis goes from 0 to 1).
    :param post_process: plotting post-process (presumably with multiple datasets per plot) as opposed to during
        the real-time simulation (which may have one or more datasets per plot).
    """
    if plot_v_time is None:
        plot_v_time = False if post_process else cfg.run_balanced_force_control
        
    fig: Figure
    ax: Axes
    
    fig, ax = plt.subplots()
    if ylabel:
        ax.set_ylabel(ylabel)
    _add_time_axis(ax, plot_v_time, post_process and normalize_time)
    if axvline is not None:
        ax.axvline(x=axvline, linestyle=":", color="k", linewidth=0.5)
    
    if limits:
        ax.set_ylim(limits)
    if yticks:
        # For now, assuming yticks dict has all the correct content and format, since I'm only passing this ad hoc.
        ax.set_yticks(yticks["major_range"], labels=yticks["labels"])
        ax.set_yticks(yticks["minor_range"], minor=True)
        
    data: list[float]
    legend_needed: bool = False

    x: list[float] | list[int] = []
    if not post_process:
        x = _timesteps if plot_v_time else _leading_edge_phi
    
    for datadict in datadicts:
        if post_process:
            x = (datadict["phi"] if not plot_v_time else
                 datadict["norm_times"] if normalize_time else
                 datadict["timesteps"])
        data: list[float] = datadict["data"]
        plot_format: str = datadict["fmt"] if "fmt" in datadict else plot_formats if plot_formats else "-"
        label: object = None if "label" not in datadict else datadict["label"]
        if label is not None:
            legend_needed = True
        ax.plot(x, data, plot_format, label=label)
    if legend_needed:
        ax.legend(loc=legend_loc)
    
    # save
    savepath: str = os.path.join(_plot_path, filename + ".png")
    fig.savefig(savepath, transparent=False, bbox_inches="tight")
    plt.close(fig)

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
    binned_values_ax.set_xticks([0, np.pi / 2, np.pi], labels=["0", r"$\pi$/2", r"$\pi$"])
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
        tensions.append(epu.tension(phandle))
        particle_phi.append(epu.embryo_phi(phandle))
    
    # plot
    ylim: tuple[float, float] = (-0.05, 0.15)
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
                       ylim=(-0.01, 0.10),
                       axhline=0,  # compression/tension boundary
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
            undivided_tensions.append(epu.tension(phandle))
            undivided_particle_phi.append(epu.embryo_phi(phandle))
        else:
            divided_tensions.append(epu.tension(phandle))
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
                       ylim=(-0.01, 0.10),
                       axhline=0,  # compression/tension boundary
                       end=end)

def _show_avg_tensions_v_microtime() -> None:
    """Show tension on a much finer timescale; median tension of all cells, and of leading edge cells"""
    axvline: float | None = None
    if cfg.run_balanced_force_control:
        # Plotting vs. time; display axvline for cessation time once we know it.
        # (However, note that it should never happen, if force parameters are correct, since
        # EVL should not expand, and should never reach the cessation position. And of course,
        # there should be little if any cell division, either.)
        if epu.cell_division_cessation_timestep > 0:
            # cell division is enabled and has already ceased, at that time
            axvline = epu.cell_division_cessation_timestep
    else:
        # Plotting vs. epiboly progress (phi); we know by definition, at what phi that will happen
        if epu.cell_division_cessation_phi > 0:
            # cell division is enabled, and that is where it will or did cease
            axvline = epu.cell_division_cessation_phi

    # calculate tensions
    leading_edge_cells: list = []
    leading_edge_cells.extend(g.LeadingEdge.items())
    all_cells: list = []
    all_cells.extend(g.LeadingEdge.items())
    all_cells.extend(g.Little.items())

    leading_edge_tensions: list[float] = [epu.tension(phandle) for phandle in leading_edge_cells]
    all_tensions: list[float] = [epu.tension(phandle) for phandle in all_cells]
    leading_edge_tension: float = np.median(leading_edge_tensions).item()
    all_tension: float = np.median(all_tensions).item()
    _median_tension_leading_edge.append(leading_edge_tension)
    _median_tension_all.append(all_tension)

    # Plot
    leading_edge_data: PlotData = {"data": _median_tension_leading_edge,
                                   "fmt": "-m",
                                   "label": "Leading edge cells"}
    all_cells_data: PlotData = {"data": _median_tension_all,
                                "fmt": "-b",
                                "label": "All cells"}
    time_axis: str = "time" if cfg.run_balanced_force_control else "leading edge progress (phi)"
    _plot_datasets_v_time([leading_edge_data, all_cells_data],
                          filename=f"Median tension v. {time_axis}",
                          limits=(-0.01, 0.2),
                          ylabel="Median particle tension",
                          axvline=axvline)

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
                           end=end)

def _show_cylindrical_straightness() -> None:
    """Plot straightness index of the leading edge, based on a cylindrical projection

    Straightness index = "D/L, where D is the beeline distance between the first and last points in the trajectory,
    and L is the path length travelled. It is a value between 0 (infinitely tortuous) to 1 (a straight line)."
    https://search.r-project.org/CRAN/refmans/trajr/html/TrajStraightness.html. This is an adaptation of that, to
    make sense on the spherical surface.

    In this version, the sphere is projected onto a cylinder, and then unrolled to a flat plane: a Cartesion
    map of simply phi vs. theta. In that configuration, the beeline path is simply a horizontal line across the
    field, connecting a particle to itself (it is on both edges). The length of the beeline is always simply
    2 * pi, regardless of which particle is selected as a starting point and regardless of its position on the
    sphere. This removes any ambiguity about which path is the beeline, and any anomalies that a simple spherical
    approach encounters near the pole; and there is no longer any possibility of getting a result greater than 1
    (actual path length is always 2 * pi or greater). This solved ALL the problems I was having.
    
    Note that the path distance used between two particles is now not the actual distance between those particles,
    nor the shortest surface-bound path between them. It is the distance on the flattened field:
    sqrt(delta_phi ** 2 + delta_theta ** 2)
    
    Moreover, correct for lopsidedness.
    Calculate a new axis based on the best-fit plane to all the edge particles. And then, rather than trying to
    figure out phi and theta of all the particles relative to that new axis directly, instead rotate the entire
    coordinate system so that the new axis becomes vertical; then phi and theta can be read from the particles
    in the usual way.
    """
    def cylindrical_distance(p1: tf.ParticleHandle, p2: tf.ParticleHandle, rotation: np.ndarray = None) -> float:
        if rotation is None:
            rotation = np.identity(3)
            
        phi1: float
        phi2: float
        theta1: float
        theta2: float
        theta1, phi1 = epu.embryo_coords(p1, rotation)
        theta2, phi2 = epu.embryo_coords(p2, rotation)
        d_phi: float = phi2 - phi1
        d_theta: float = abs(theta2 - theta1)
        if d_theta > np.pi:
            # crossing the 0-to-2pi boundary
            d_theta = 2 * np.pi - d_theta
        return np.sqrt(d_phi ** 2 + d_theta ** 2)
        
    ordered_particles: list[tf.ParticleHandle] = epu.get_leading_edge_ordered_particles()
    
    # Find the appropriate axis for the leading edge, in case it's lopsided
    _, normal_vec = epu.leading_edge_best_fit_plane()
    rotation_matrix: np.ndarray = epu.rotation_matrix(axis=normal_vec)
    
    # Separately, plot phi of normal_vec as a measure of lopsidedness
    _show_margin_lopsidedness(normal_vec)
    
    # Calculate the path distance from particle to particle
    p: tf.ParticleHandle
    previous_p: tf.ParticleHandle = ordered_particles[-1]  # last one
    path_length: float = 0
    for p in ordered_particles:
        path_length += cylindrical_distance(p, previous_p, rotation_matrix)
        previous_p = p
    
    beeline_path_length: float = 2 * np.pi
    
    straightness_cyl: float = beeline_path_length / path_length
    _straightness_cyl.append(straightness_cyl)
    
    # 0.90 is usually good enough for bottom, but if it dips below that, get the whole plot in frame
    limits: tuple[float, float] = _expand_limits_if_needed(limits=(0.9, 1.001), data=_straightness_cyl)
    straightness_data: PlotData = {"data": _straightness_cyl}
    _plot_datasets_v_time([straightness_data],
                          filename="Straightness Index",
                          limits=limits,
                          ylabel="Straightness Index (SI)",
                          plot_formats=".-b")

def _show_margin_lopsidedness(normal_vec: tf.fVector3) -> None:
    """Plot the angle of the margin axis - i.e., phi of normal_vec generated by the straightness calculation

    Measure lopsided / off-center / non-synchronous epiboly.
    This should be close to 0 for the non-lopsided case, increasing toward max of pi/2 (but should never get that big).
    """
    _, _, phi = tfu.spherical_from_cartesian(normal_vec)
    _margin_lopsidedness.append(phi)

    # ToDo: need to implement code to take data from multiple runs and plot them together on a single Axes.
    yticks = {"major_range": np.arange(0, 0.102 * np.pi, 0.05 * np.pi),
              "minor_range": np.arange(0, 0.102 * np.pi, 0.01 * np.pi),
              "labels": ["0", r"0.05$\pi$", r"0.10$\pi$"]}
    
    lopsidedness_data: PlotData = {"data": _margin_lopsidedness}
    _plot_datasets_v_time([lopsidedness_data],
                          filename="Margin lopsidedness",
                          limits=(-0.002 * np.pi, 0.102 * np.pi),
                          ylabel="Margin lopsidedness (angle of axis)",
                          plot_formats=".-b",
                          yticks=yticks)

def _show_bond_counts() -> None:
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
    
    bond_data: PlotData = {"data": _bonds_per_particle}
    _plot_datasets_v_time([bond_data],
                          filename="Mean bond count per particle",
                          ylabel="Mean bonded neighbors per particle",
                          plot_formats="b.")

def _show_margin_population() -> None:
    _margin_count.append(len(g.LeadingEdge.items()))
    _margin_cum_in.append(epu.cumulative_to_edge)
    _margin_cum_out.append(epu.cumulative_from_edge)
    if cfg.cell_division_enabled:
        _margin_cum_divide.append(epu.cumulative_edge_divisions)
    
    # plot just the total (and with no legend)
    margin_count_data: PlotData = {"data": _margin_count, "fmt": ".b"}
    maximum: int = max(_margin_count)
    limits: tuple[float, float] = _expand_limits_if_needed(limits=(-2, maximum + 2), data=_margin_count)
    _plot_datasets_v_time([margin_count_data],
                          filename="Margin cell rearrangement",
                          limits=limits,
                          ylabel="Number of margin cells")

    # Plot all four (now with legend, and no ylabel), and save under a different name
    margin_count_data["label"] = "Total margin cell count"
    margin_cum_in_data: PlotData = {"data": _margin_cum_in, "fmt": "--b", "label": "Cumulative in"}
    margin_cum_out_data: PlotData = {"data": _margin_cum_out, "fmt": ":b", "label": "Cumulative out"}
    margin_cum_divide_data: PlotData = {"data": _margin_cum_divide, "fmt": "-b", "label": "Cumulative divisions"}
    datasets: list[PlotData] = [margin_count_data, margin_cum_in_data, margin_cum_out_data]
    maximum = max(maximum, max(_margin_cum_in), max(_margin_cum_out))
    if cfg.cell_division_enabled:
        datasets.append(margin_cum_divide_data)
        maximum = max(maximum, max(_margin_cum_divide))
    limits = _expand_limits_if_needed(limits=(-2, maximum + 2), data=_margin_count)
    _plot_datasets_v_time(datasets,
                          filename="Margin cell rearrangement, plus cumulative",
                          limits=limits,
                          legend_loc="center right")

def _show_forces() -> None:
    """Plots the TOTAL force on the leading edge globally
    
    Force per unit edge is constant, so this just reflects the changing circumference over time.
    """
    _forces.append(mt.current_total_force())
    forces_data: PlotData = {"data": _forces}
    _plot_datasets_v_time([forces_data],
                          filename="Forces on leading edge",
                          ylabel="Forces",
                          plot_formats="b.")

def _show_progress_graph() -> None:
    progress_data: PlotData = {"data": _leading_edge_phi}
    
    yticks = {"major_range": [np.pi / 2, np.pi * 3/4, np.pi],
              "minor_range": [np.pi * 5/8, np.pi * 7/8],
              "labels": [r"$\pi$/2", r"3$\pi$/4", r"$\pi$"]}

    _plot_datasets_v_time([progress_data],
                          filename="Leading edge phi",
                          limits=(np.pi * 7 / 16, np.pi),
                          ylabel=r"Leading edge  $\bar{\phi}$  (radians)",
                          plot_formats="b.",
                          yticks=yticks,
                          plot_v_time=True)

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
        
    if cfg.cell_division_enabled and epu.cell_division_cessation_timestep == 0:
        # We haven't yet detected threshold crossing, so check it.
        if epu.leading_edge_mean_phi() > epu.cell_division_cessation_phi:
            # We've detected that phi has crossed the threshold and cell division has ceased,
            # so record when that happened.
            epu.cell_division_cessation_timestep = _timestep

    # Don't need to add to the graphs every timestep.
    simtime_interval: float = 4
    timestep_interval: int = round(simtime_interval / cfg.dt)
    if _timestep % timestep_interval == 0 or end:
        # alternative measures of time, available to all plots:
        _timesteps.append(_timestep)
        _leading_edge_phi.append(round(epu.leading_edge_mean_phi(), 4))
        
        _show_progress_graph()
        _show_bond_counts()
        _show_forces()
        _show_cylindrical_straightness()
        _show_margin_population()
        _show_avg_tensions_v_microtime()

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
            "straightness_cyl": _straightness_cyl,
            "margin_lopsidedness": _margin_lopsidedness,
            "margin_count": _margin_count,
            "margin_cum_in": _margin_cum_in,
            "margin_cum_out": _margin_cum_out,
            "margin_cum_divide": _margin_cum_divide,
            "median_tension_leading_edge": _median_tension_leading_edge,
            "median_tension_all": _median_tension_all,
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
    global _timestep, _bonds_per_particle, _leading_edge_phi, _forces
    global _straightness_cyl
    global _margin_lopsidedness, _timesteps
    global _margin_count, _margin_cum_in, _margin_cum_out, _margin_cum_divide
    global _median_tension_leading_edge, _median_tension_all
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
    _straightness_cyl = d["straightness_cyl"]
    _margin_lopsidedness = d["margin_lopsidedness"]
    _margin_count = d["margin_count"]
    _margin_cum_in = d["margin_cum_in"]
    _margin_cum_out = d["margin_cum_out"]
    _margin_cum_divide = d["margin_cum_divide"]
    _median_tension_leading_edge = d["median_tension_leading_edge"]
    _median_tension_all = d["median_tension_all"]
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
    
def post_process_graphs(simulation_data: list[dict]) -> None:
    def normalize(datadicts: list[PlotData]) -> None:
        datadict: PlotData
        for datadict in datadicts:
            if "timesteps" in datadict:
                timesteps: list[int] = datadict["timesteps"]
                datadict["norm_times"] = list(np.array(timesteps) / timesteps[-1])
    
    def color_code_and_clean_up_labels(datadicts: list[PlotData], legend_format: str = "{}") -> None:
        """Color code plot lines according to treatment (parameter value); and only label one plot per treatment
        
        On entry, each PlotData["label"] is a numerical value. Sort the list according to that value (so that
        the labels will be in the right order in the legend); then remove all labels except one per treatment
        (so that each label only appears once in the legend); and then wrap that numerical
        value in a string that explains what it is. Then set distinct plot colors for each treatment,
        but the SAME plot color for the multiple plot lines of the SAME treatment.
        
        :param datadicts: one PlotData for each line that is to be plotted. "label" field should
        be numerical, representing the treatment.
        :param legend_format: a string containing a replacement field, into which the treatment value for
        each legend will be inserted
        """
        datadicts.sort(key=lambda plot_data: plot_data["label"])
        datadict: PlotData
        previous_label: int | float = -1
        cycler_index: int = -1
        
        # For each distinct treatment, provide a different color; and a str label only for one plot of each treatment
        for datadict in datadicts:
            # datadict["label"] is known to be numerical
            current_label: int | float = datadict["label"]  # type: ignore
            if current_label > previous_label:
                datadict["label"] = legend_format.format(current_label)
                previous_label = current_label
                cycler_index += 1
            else:
                del datadict["label"]
            datadict["fmt"] = f"-C{cycler_index}"
    
    def show_multi_progress_by_constraint_k() -> None:
        """Overlay multiple progress plots on one Axes, color-coded by treatment (edge bond-angle constraint lambda)"""
        datadicts: list[PlotData] = [{"data": simulation["plot"]["leading_edge_phi"],
                                      "timesteps": simulation["plot"]["timesteps"],
                                      "label": simulation["config"]["config_values"]["model"]["k_edge_bond_angle"]
                                      } for simulation in simulation_data]
        normalize(datadicts)
        
        color_code_and_clean_up_labels(datadicts, legend_format=r"$\lambda$ = {}")

        yticks = {"major_range": [np.pi / 2, np.pi * 3 / 4, np.pi],
                  "minor_range": [np.pi * 5 / 8, np.pi * 7 / 8],
                  "labels": [r"$\pi$/2", r"3$\pi$/4", r"$\pi$"]}

        _plot_datasets_v_time(datadicts,
                              filename="Leading edge phi v. time",
                              limits=(np.pi * 7 / 16, np.pi),
                              ylabel=r"Leading edge  $\bar{\phi}$  (radians)",
                              yticks=yticks,
                              plot_v_time=True,
                              post_process=True)

        _plot_datasets_v_time(datadicts,
                              filename="Leading edge phi v. normalized time",
                              limits=(np.pi * 7 / 16, np.pi),
                              ylabel=r"Leading edge  $\bar{\phi}$  (radians)",
                              yticks=yticks,
                              plot_v_time=True,
                              normalize_time=True,
                              post_process=True)

    def show_multi_margin_pop_by_constraint_k() -> None:
        """Overlay multiple margin pop plots on one Axes, color-coded by edge bond-angle constraint lambda"""
        margin_count_dicts: list[PlotData] = []
        margin_cum_dicts: list[PlotData] = []
        simulation: dict
        for simulation in simulation_data:
            leading_edge_phi: list[float] = simulation["plot"]["leading_edge_phi"]
            label: int = simulation["config"]["config_values"]["model"]["k_edge_bond_angle"]
            
            margin_count: PlotData = {"data": simulation["plot"]["margin_count"],
                                      "phi": leading_edge_phi,
                                      "label": label}
            margin_count_dicts.append(margin_count)
            
            margin_cum_in: list[int] = simulation["plot"]["margin_cum_in"]
            margin_cum_out: list[int] = simulation["plot"]["margin_cum_out"]
            margin_cum_total: list[int] = list(np.add(margin_cum_in, margin_cum_out))
            margin_cum: PlotData = {"data": margin_cum_total,
                                    "phi": leading_edge_phi,
                                    "label": label}
            margin_cum_dicts.append(margin_cum)
            
        color_code_and_clean_up_labels(margin_count_dicts, legend_format=r"$\lambda$ = {}")
        color_code_and_clean_up_labels(margin_cum_dicts, legend_format=r"$\lambda$ = {}")
        
        all_count_data: list[list[int]] = [datadict["data"] for datadict in margin_count_dicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=(-2, 10), data=all_count_data)
        _plot_datasets_v_time(margin_count_dicts,
                              filename="Margin cell count",
                              limits=limits,
                              post_process=True)
        
        all_cum_data: list[list[int]] = [datadict["data"] for datadict in margin_cum_dicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=(-2, 10), data=all_cum_data)
        _plot_datasets_v_time(margin_cum_dicts,
                              filename="Margin cell rearrangement, cumulative",
                              limits=limits,
                              post_process=True)

    def show_composite_medians(rawdicts: list[PlotData],
                               dataname: str,
                               default_limits: tuple[float, float],
                               legend_format: str) -> None:
        """Combine multiple datasets into composite metrics, one per 'treatment'.
        
        'Treatment' refers to the different values of a single variable that we are contrasting.
        
        The handling of the labels and legends assumes there is more than one treatment being
        compared in the plot, but if we ever need to do this for just a single treatment, that can
        be tweaked as necessary.
        
        :param rawdicts: one PlotData for each simulation that is to be plotted. It should have already
        been normalized (normalized time data calculated for each simulation). "label" field should
        be numerical, representing the treatment.
        :param dataname: will be used both as the title of the y-axis, and as part of the filename for
        the saved plot, so should be suitable for both.
        :param default_limits: y-axis limits for whatever data was passed. These will be expanded if
        the range of the actual data exceeds the default_limits.
        :param legend_format: a string containing a replacement field, into which the treatment value for
        each legend will be inserted
        """
        composite_dicts: dict[str: PlotData] = {}
        composite_key: str
        # Using str keys for the different treatments, since I'm unsure how safe it is to use floats as keys
        # composite_dicts will contain one PlotData for each treatment, keyed by the treatment value

        for rawdict in rawdicts:
            composite_key = str(rawdict["label"])
            
            # Create the appropriate key-value pair in which to store the content of the current rawdict,
            # if it doesn't yet exist. (I.e., the first time each treatment is encountered.)
            if composite_key not in composite_dicts:
                composite_dicts[composite_key] = {"data": [],
                                                  "phi": [],
                                                  "timesteps": [],  # Not sure needed; process with the others for now
                                                  "norm_times": [],
                                                  "label": rawdict["label"]}
                
            # Add each list from the current rawdict into the corresponding list in the composite dict
            composite_dict: PlotData = composite_dicts[composite_key]
            plotdata_key: str
            for plotdata_key in ["data", "phi", "norm_times"]:
                # (Type checker doesn't like variables as keys; it's fine.)
                composite_dict[plotdata_key].extend(rawdict[plotdata_key])  # type: ignore
            
        # After exiting the above outer loop (for rawdict in rawdicts), we now have a dict (composite_dicts)
        # containing exactly one sub-dict for each treatment. Their lists are no longer in chronological
        # order, since we simply concatenated all component lists from the rawdicts. But this won't matter
        # since we'll now bin them. (In two different ways, once for plotting v. phi, and once for plotting v.
        # normalized time.)
        
        # Binned dicts will be structured just like composite_dicts, i.e., each will contain one sub-dict for
        # each treatment. But now the lists inside each dict will only contain num_bins elements,
        # representing the averaged time and data values from multiple original rawdicts (multiple
        # original simulation runs). Each binned dict will have a "data" member, but they won't contain the
        # same values, because the binning will shake out differently, depending on whether it's for plotting
        # v. phi, or v. normalized time.
        num_bins = 20
        binned_v_phi_dicts: dict[str: PlotData] = {}
        binned_v_time_dicts: dict[str: PlotData] = {}
        
        # bin the time values over this range:
        min_time: float = 0.0
        max_time: float = 1.0
        
        # bin the phi values over this range:
        all_phi: list[list[float]] = [composite_dict["phi"] for composite_dict in composite_dicts.values()]
        flat_phi_iterator = chain.from_iterable(all_phi)
        min_phi: float = min(flat_phi_iterator)
        max_phi: float = np.pi
        
        for composite_key, composite_dict in composite_dicts.items():
            np_data = np.array(composite_dict["data"])
            np_phi = np.array(composite_dict["phi"])
            np_times = np.array(composite_dict["norm_times"])
            
            time_edges: np.ndarray = np.linspace(min_time, max_time, num_bins + 1)
            phi_edges: np.ndarray = np.linspace(min_phi, max_phi, num_bins + 1)
            time_bin_size: float = time_edges[1] - time_edges[0]
            phi_bin_size: float = phi_edges[1] - phi_edges[0]
            time_indices: np.ndarray = np.digitize(np_times, time_edges)
            phi_indices: np.ndarray = np.digitize(np_phi, phi_edges)
            
            # Note: numpy ufunc equality and masking!
            # https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
            time_bins: list[np.ndarray] = [np_data[time_indices == i] for i in range(1, time_edges.size)]
            phi_bins: list[np.ndarray] = [np_data[phi_indices == i] for i in range(1, phi_edges.size)]

            # Finally, construct the x and y datasets we'll actually plot
            time_bin: np.ndarray
            phi_bin: np.ndarray
            time_median_data: list[float] = []
            phi_median_data: list[float] = []
            time_axis: list[float] = []
            phi_axis: list[float] = []
            for i, time_bin in enumerate(time_bins):
                if time_bin.size > 0:
                    # should be always.
                    # np.median() returns ndarray but is really float because time_bin is 1d
                    time_median_data.append(np.median(time_bin).item())
                    # position each point horizontally halfway between the bin edges
                    time_axis.append(time_edges[i] + time_bin_size / 2)
            for i, phi_bin in enumerate(phi_bins):
                if phi_bin.size > 0:
                    phi_median_data.append(np.median(phi_bin).item())
                    phi_axis.append(phi_edges[i] + phi_bin_size / 2)
                    
            binned_v_time_dicts[composite_key] = {"data": time_median_data,
                                                  "norm_times": time_axis,
                                                  "label": composite_dict["label"]}
            binned_v_phi_dicts[composite_key] = {"data": phi_median_data,
                                                 "phi": phi_axis,
                                                 "label": composite_dict["label"]}
            
        # Finally, after exiting the above loop, we have two of dict[str: PlotData], one binned by
        # normalized times, and one by phi. Now we can plot them.
        time_dicts_list: list[PlotData] = list(binned_v_time_dicts.values())
        phi_dicts_list: list[PlotData] = list(binned_v_phi_dicts.values())
        color_code_and_clean_up_labels(time_dicts_list, legend_format)
        color_code_and_clean_up_labels(phi_dicts_list, legend_format)
        
        # Since the data was binned differently in the two dicts, they might have slightly
        # different ranges. So to make the y-axis scales identical on the two plots we'll
        # generate, combine ALL the data from both to determine the y limits:
        all_data: list[list[float]] = [plot_data["data"] for plot_data in time_dicts_list]
        all_data.extend([plot_data["data"] for plot_data in phi_dicts_list])
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_data)
        
        _plot_datasets_v_time(time_dicts_list,
                              filename=f"Median {dataname} v. normalized time",
                              limits=limits,
                              ylabel=f"Median {dataname}",
                              plot_v_time=True,
                              normalize_time=True,
                              post_process=True)
        
        _plot_datasets_v_time(phi_dicts_list,
                              filename=f"Median {dataname} v. phi",
                              limits=limits,
                              ylabel=f"Median {dataname}",
                              post_process=True)
        
    def show_multi_straightness_by_constraint_k() -> None:
        """Overlay multiple Straightness Index plots on one Axes, color-coded by edge bond-angle constraint lambda"""
        datadicts: list[PlotData] = [{"data": simulation["plot"]["straightness_cyl"],
                                      "phi": simulation["plot"]["leading_edge_phi"],
                                      "timesteps": simulation["plot"]["timesteps"],
                                      "label": simulation["config"]["config_values"]["model"]["k_edge_bond_angle"]
                                      } for simulation in simulation_data]
        normalize(datadicts)
        
        dataname: str = "Straightness Index (SI)"
        default_limits: tuple[float, float] = (0.9, 1.001)
        legend_format: str = r"$\lambda$ = {}"
        show_composite_medians(datadicts, dataname, default_limits, legend_format)

        color_code_and_clean_up_labels(datadicts, legend_format)
        
        all_data: list[list[float]] = [data["data"] for data in datadicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_data)

        _plot_datasets_v_time(datadicts,
                              filename=f"{dataname} v. phi",
                              limits=limits,
                              ylabel=dataname,
                              post_process=True)

        _plot_datasets_v_time(datadicts,
                              filename=f"{dataname} v. time",
                              limits=limits,
                              ylabel=dataname,
                              plot_v_time=True,
                              post_process=True)

        _plot_datasets_v_time(datadicts,
                              filename=f"{dataname} v. normalized time",
                              limits=limits,
                              ylabel=dataname,
                              plot_v_time=True,
                              normalize_time=True,
                              post_process=True)

    def show_multi_tension() -> None:
        """Overlay multiple tension plots on one Axes: all cells vs. leading edge cells, in different colors"""
        # Get axvline position, which should be the same in all the sims, as long as they all started
        # at the same epiboly_initial_percentage and had the same cell_division_cessation_percentage.
        # So just grab it from the first one:
        axvline = simulation_data[0]["epiboly"]["cell_division_cessation_phi"]
        
        # From each simulation, get two datasets: tension for leading edge cells, and for all cells
        datadicts: list[PlotData] = []
        simulation: dict
        for index, simulation in enumerate(simulation_data):
            leading_edge_data: PlotData = {"data": simulation["plot"]["median_tension_leading_edge"],
                                           "phi": simulation["plot"]["leading_edge_phi"],
                                           "fmt": "-m"}
            all_cells_data: PlotData = {"data": simulation["plot"]["median_tension_all"],
                                        "phi": simulation["plot"]["leading_edge_phi"],
                                        "fmt": "-b"}
            if index == 0:
                # Only need to add legend labels once
                leading_edge_data["label"] = "Leading edge cells"
                all_cells_data["label"] = "All cells"
            datadicts.extend([leading_edge_data, all_cells_data])
            
        _plot_datasets_v_time(datadicts,
                              filename=f"Median tension v. leading edge progress (phi)",
                              limits=(-0.01, 0.2),
                              ylabel="Median particle tension",
                              axvline=axvline,
                              post_process=True)

    _init_graphs()
    # print(simulation_data)
    show_multi_tension()
    show_multi_straightness_by_constraint_k()
    show_multi_margin_pop_by_constraint_k()
    show_multi_progress_by_constraint_k()
