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
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import matplotlib as mpl

import tissue_forge as tf
import epiboly_globals as g

import biology.bond_maintenance as bonds
import biology.forces as fo
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
_median_tension_circumferential: list[float] = []
_median_speed_leading_edge: list[float] = []
_speed_by_phi_diffs: list[float] = []

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

# For speed calculations. Initialize in first call to show_graphs():
_previous_leading_edge_phi: float = 0.0
_previous_universe_time: float = 0.0

class PlotData(TypedDict, total=False):
    data: list[float] | list[int]   # required
    range_low: list[float]          # for post-process use only; ignored in real-time plotting
    range_high: list[float]
    phi: list[float]                # x axis (any of these 3) required in post-process; ignored in real-time plotting
    timesteps: list[int]
    norm_times: list[float]
    model_id: int
    fmt: str                        # not required
    color: str                      # not required, and if present and not None, overrides fmt color
    label: object                   # not required; can be str or anything that can be turned into str (float, int...)
    second_label: bool | None       # not required; must be bool, because two keys/labels only allowed when both bool

def _init_graphs() -> None:
    """Initialize matplotlib and also a subdirectory in which to put the saved plots
    
    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _plot_path
    
    # Set font globally (Helvetica preference, with fallbacks)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    
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
    
    axes.set_xlabel("Leading edge position")
    axes.set_xlim(np.pi * 7 / 16, np.pi)
    axes.set_xticks([np.pi / 2, np.pi * 5/8, np.pi * 3/4, np.pi * 7/8, np.pi],
                    labels=[r"$\pi$/2", "", r"3$\pi$/4", "", r"$\pi$"])

def _balanced_force_ever() -> bool:
    """True if the leading edge was not moving, due to balanced forces, at least PART of the time"""
    return cfg.run_balanced_force_control or cfg.balanced_force_equilibration_kludge

def _plot_datasets_v_time(datadicts: list[PlotData],
                          filename: str,
                          limits: tuple[float, float] = None,
                          ylabel: str = None,
                          plot_formats: str = None,
                          axvline: float = None,
                          legend_loc: str = None,
                          yticks: dict = None,
                          title: str = None,
                          plot_v_time: bool = None,
                          normalize_time: bool = False,
                          suppress_timestep_zero: bool = False,
                          post_process: bool = False,
                          plot_ranges: bool = False,
                          desired_height_inches: float = None) -> None:
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
        phi just reflects cessation config); but it won't work on a plot v. timesteps, because the position
        will vary among the datasets. If necessary to display that, it would have to be by a different means, like
        a special marker on each line in the plot.
    :param legend_loc: optional, specify where legend will go if there is one. In post-process, interpret None to
        mean draw the legend outside the plot axes, as opposed to "best".
    :param yticks: Currently using this in only 2 spots, ad hoc, not anticipating more. But if I start using this
        more generally, then... ToDo: define a proper typed dict and do parameter validation
    :param title: A title for the plot.
    :param plot_v_time: whether to plot vs. time instead of vs. phi. In real-time plotting only, a value of None
        means to decide based on cfg.run_balanced_force_control. In post-process plotting, None is treated as False.
    :param normalize_time: In real-time plotting, ignored. In post-process plotting vs. phi, ignored. In post-process
        plotting vs. time, whether to normalize (time axis goes from 0 to 1).
    :param suppress_timestep_zero: don't plot the first point in the dataset (regardless of whether we are plotting
        vs. timesteps or something else). Useful for e.g. leading edge speed, since it starts as 0 but this is
        an artifact; we want to start at the very high value that comes at the next timestep measured.
    :param post_process: plotting post-process (presumably with multiple datasets per plot) as opposed to during
        the real-time simulation (which may have one or more datasets per plot).
    :param plot_ranges: in consensus plots, add a shaded area for the data range. Those values should be provided in
        the "range_low" and "range_high" element of each Plotdata. Those will only be present post-process (so only
        when post_process is True), and only for consensus plots. When plot_ranges is False, ignore them; when
        plot_ranges is True, check for their presence and plot them. Note that Axes.fill_between() is fine with only
        a single set of boundary data passed (the other defaulting to float 0), and uses 0 for the missing boundary,
        so I might as well support that even though I don't currently need it. So at least one of the two must
        be present.
    :param desired_height_inches: when plotting final version for publication, adjust everything for the desired
        size at publication. Proportions will be maintained, so width is not explicitly specified. This should
        result in appropriate font and axis tick sizes and so on.
    """
    def adjust_figsize(move_legend_outside: bool) -> None:
        """Resize the Figure to the desired height. And then move the legend outside.

        (With help/advice from ChatGPT on how to get the scaling effect I want in matplotlib.)
        """
        # Draw figure in its expected export form so we can measure
        fig.tight_layout()
        fig.canvas.draw()
        
        # Scale
        current_width: float
        current_height: float
        current_width, current_height = fig.get_size_inches()
        scale: float = desired_height_inches / current_height
        desired_width_inches: float = current_width * scale
        fig.set_size_inches(desired_width_inches, desired_height_inches)
        
        # I don't quite remember why I included the following "if True", since I'm finally committing this code,
        # several months after I wrote it. I think I wanted the option to retain the previous logic, of only moving
        # the legend outside when no explicit legend location was specified, so I built that logic in here,
        # but then overrode it with "if True" so that for now, as long as a size adjustment is being
        # applied, I *always* move the legend outside. Now that I've generated all the figures this way,
        # it seems to work pretty well.
        if True or move_legend_outside:
            legend: Legend = ax.get_legend()
            
            # Cannot assume the figure has a legend
            if legend:
                legend.set_bbox_to_anchor((1.02, 0.5))  # type: ignore  # where to pin the legend
                legend.set_loc("center left")  # what part of the legend to pin
    
    if plot_v_time is None:
        plot_v_time = False if post_process else _balanced_force_ever()
        
    fig: Figure
    ax: Axes
    
    fig, ax = plt.subplots()
    if ylabel:
        ax.set_ylabel(ylabel)
    _add_time_axis(ax, plot_v_time, post_process and normalize_time)
    if axvline is not None:
        ax.axvline(x=axvline, linestyle="--", color="0.25", linewidth=1.0)
    
    if limits:
        ax.set_ylim(limits)
    if yticks:
        # For now, assuming yticks dict has all the correct content and format, since I'm only passing this ad hoc.
        ax.set_yticks(yticks["major_range"], labels=yticks["labels"])
        ax.set_yticks(yticks["minor_range"], minor=True)
        
    if title:
        ax.set_title(title)
        
    data: list[float]
    legend_needed: bool = False

    x: list[float] | list[int] = []
    if not post_process:
        x = _timesteps if plot_v_time else _leading_edge_phi
        if suppress_timestep_zero:
            x = x[1:]
    
    for datadict in datadicts:
        if post_process:
            x = (datadict["phi"] if not plot_v_time else
                 datadict["norm_times"] if normalize_time else
                 datadict["timesteps"])
            if suppress_timestep_zero:
                x = x[1:]
                
        range_y1: list[float] = []
        range_y2: list[float] = []
        if plot_ranges:
            # Determine whether either or both fill boundaries are present, and if so, use them.
            if "range_low" in datadict:
                range_y1 = datadict["range_low"]
                if "range_high" in datadict:
                    range_y2 = datadict["range_high"]
            elif "range_high" in datadict:
                range_y1 = datadict["range_high"]
                
        data: list[float] = datadict["data"]
        if suppress_timestep_zero:
            data = data[1:]
            range_y1 = range_y1[1:]
            range_y2 = range_y2[1:]
        plot_format: str = datadict["fmt"] if "fmt" in datadict else plot_formats if plot_formats else "-"
        color: str = datadict["color"] if "color" in datadict else None
        label: object = None if "label" not in datadict else datadict["label"]
        if label is not None:
            legend_needed = True
        if color:
            # use the specified colors, overriding the ones in plot_format
            ax.plot(x, data, "-", color=color, label=label)
            if range_y1:
                # This case should never occur since I think color is only set for individual trajectory plots,
                # while range_y1 will only ever be non-empty for consensus plots, but include just in case:
                ax.fill_between(x, range_y1, range_y2 or 0, color=color, alpha=0.1)
        else:
            # use the colors specified in plot_format. Since it was passed in as a format string, which fill_between()
            # doesn't understand, use this clever trick to robustly extract the color from the plotted line, no
            # matter how the string was specified (thanks, ChatGPT):
            [line] = ax.plot(x, data, plot_format, label=label)
            if range_y1:
                ax.fill_between(x, range_y1, range_y2 or 0, color=line.get_color(), alpha=0.1)
    if legend_needed:
        ax.legend(loc=legend_loc)
    
    if post_process and desired_height_inches is not None:
        adjust_figsize(move_legend_outside=legend_loc is None)
    
    # save
    savepath: str = os.path.join(_plot_path, filename + (".svg" if post_process else ".png"))
    fig.savefig(savepath, transparent=False, bbox_inches="tight")   # "transparent" needed for png; ignored for .svg
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
        binned_values_ax.axvline(x=axvline, linestyle="--", color="0.25", linewidth=1.0)
    binned_values_ax.set_xlim(0, np.pi)
    binned_values_ax.set_xticks([0, np.pi / 2, np.pi], labels=["0", r"$\pi$/2", r"$\pi$"])
    if ylabel is not None:
        binned_values_ax.set_ylabel(ylabel)
    if axhline is not None:
        binned_values_ax.axhline(y=axhline, linestyle="--", color="0.25", linewidth=1.0)
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
    tensions_ax.axhline(y=0, linestyle="--", color="0.25", linewidth=1.0)  # tension/compression boundary
    tensions_ax.text(0.02, 0.97, f"T={_timestep}", transform=tensions_ax.transAxes,
                     verticalalignment="top", horizontalalignment="left",
                     fontsize=28, fontweight="bold")
    
    phandle: tf.ParticleHandle
    tensions: list[float] = []
    particle_phi: list[float] = []
    for phandle in g.Evl.items():
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
    for phandle in g.Evl.items():
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

def _get_cell_division_cessation_threshold() -> float | None:
    """Return x-coord at which to draw cell-division cessation. In terms of phi if possible, else timestep
    
    If running balanced force control, then plotting v. timestep, so return cell_division_cessation_timestep.
    Else plotting v. phi, so return cell_division_cessation_phi.
    Return None if cell division disabled, or if plotting v. time and the cessation timestep hasn't been
    determined yet. In other words, if None is returned, don't draw the line.
    """
    axvline: float | None = None
    if _balanced_force_ever():
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
            
    return axvline

def _show_avg_tensions_v_microtime() -> None:
    """Show tension on a much finer timescale; median tension of all cells, and of leading edge cells"""
    axvline: float | None = _get_cell_division_cessation_threshold()

    # calculate tensions
    leading_edge_cells: list = []
    leading_edge_cells.extend(g.LeadingEdge.items())
    all_cells: list = []
    all_cells.extend(g.LeadingEdge.items())
    all_cells.extend(g.Evl.items())

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
    time_axis: str = "time" if _balanced_force_ever() else "leading edge progress (phi)"
    _plot_datasets_v_time([leading_edge_data, all_cells_data],
                          filename=f"Median tension v. {time_axis}",
                          limits=(-0.01, 0.2),
                          ylabel="Median particle tension",
                          axvline=axvline)

def _show_circumferential_edge_tension_v_microtime() -> None:
    """Like _show_avg_tensions_v_microtime(), but get tension only along edge bonds, not whole cell tension"""
    axvline: float | None = _get_cell_division_cessation_threshold()

    # calculate circumferential edge tensions - i.e., only the tension of edge bonds
    leading_edge_bonds: list[tf.BondHandle] = [bhandle for bhandle in tf.BondHandle.items()
                                               if bonds.is_edge_bond(bhandle)]

    leading_edge_tensions: list[float] = [epu.bond_tension(bhandle) for bhandle in leading_edge_bonds]
    leading_edge_tension: float = np.median(leading_edge_tensions).item()
    _median_tension_circumferential.append(leading_edge_tension)

    # Plot
    limits: tuple[float, float] = _expand_limits_if_needed(limits=(-0.01, 0.2), data=_median_tension_circumferential)
    leading_edge_data: PlotData = {"data": _median_tension_circumferential,
                                   "fmt": "-b"}
    time_axis: str = "time" if _balanced_force_ever() else "leading edge progress (phi)"
    _plot_datasets_v_time([leading_edge_data],
                          filename=f"Median circumferential edge tension v. {time_axis}",
                          limits=limits,
                          ylabel="Median circumferential tension",
                          axvline=axvline)

def _show_speed_by_phi_diffs() -> None:
    """Plots the speed of leading edge movement based on tracking leading edge position, rather than particle velocities

    Speed based on particle velocities was ridiculously noisy; hence all the time averaging with the original
    approach, and then the crazy noisy plots with the approach in _show_leading_edge_speed_v_microtime().
    Instead, let's just track the position of the leading edge, which is a mean over all the edge particles and
    isn't noisy at all. Then calculate the speed from that. Still a bit noisy, late in epiboly! But definitely an
    improvement over the particle-velocity approach.
    """
    global _previous_leading_edge_phi, _previous_universe_time
    
    axvline: float | None = _get_cell_division_cessation_threshold()

    # Calculate speed
    # Scale is arbitrary but do it in terms of Universe.time so the values will be robust to changes
    # either in cfg.dt (timestep interval) or in the frequency of plot generation.
    new_leading_edge_phi: float = epu.leading_edge_mean_phi()
    new_universe_time: float = tf.Universe.time
    radians_traveled: float = new_leading_edge_phi - _previous_leading_edge_phi
    time_elapsed: float = new_universe_time - _previous_universe_time
    _previous_leading_edge_phi = new_leading_edge_phi
    _previous_universe_time = new_universe_time
    
    # First time through, time_elapsed will be 0, and speed will be zero (which we won't be plotting anyway)
    current_speed: float = 0 if time_elapsed == 0 else radians_traveled / time_elapsed
    _speed_by_phi_diffs.append(current_speed)

    # Plot
    limits: tuple[float, float] = _expand_limits_if_needed(limits=(-0.0005, 0.03), data=_speed_by_phi_diffs)
    speed_data: PlotData = {"data": _speed_by_phi_diffs,
                            "fmt": "-b"}
    time_axis: str = "time" if _balanced_force_ever() else "leading edge progress (phi)"
    _plot_datasets_v_time([speed_data],
                          filename=f"Speed by phi diffs v. {time_axis}",
                          limits=limits,
                          ylabel="Epiboly speed",
                          axvline=axvline,
                          suppress_timestep_zero=True)

def _phi_and_vegetal_speed(phandle: tf.ParticleHandle) -> tuple[float, float]:
    theta, particle_position_phi = epu.embryo_coords(phandle)
    tangent_phi: float = particle_position_phi + np.pi / 2
    tangent_vec: tf.fVector3 = tfu.cartesian_from_spherical([1, theta, tangent_phi])
    velocity: tf.fVector3 = phandle.velocity
    return particle_position_phi, tfu.signed_scalar_from_vector_projection(velocity, tangent_vec)

def _show_leading_edge_speed_v_microtime() -> None:
    """Show speed on a much finer timescale; median veg-directed velocity component of leading edge cells
    
    Don't plot timestep 0 (the first datapoint) because it has a meaningless value near 0. Start with the
    first measured value after the simulation starts, after the speed jumps up to its true starting value.
    """
    axvline: float | None = _get_cell_division_cessation_threshold()

    # calculate leading edge speeds - i.e., only the speed of leading edge particles
    phi_and_speed_tuple_generator = (_phi_and_vegetal_speed(phandle) for phandle in g.LeadingEdge.items())
    leading_edge_speeds: list[float] = [speed for _, speed in phi_and_speed_tuple_generator]
    # leading_edge_speed: float = np.median(leading_edge_speeds).item()
    # ToDo: If I use this version with fmean() instead, fix all the var names and strings!
    leading_edge_speed: float = fmean(leading_edge_speeds)
    _median_speed_leading_edge.append(leading_edge_speed)

    # Plot
    limits: tuple[float, float] = _expand_limits_if_needed(limits=(-0.005, 0.08), data=_median_speed_leading_edge)
    speed_data: PlotData = {"data": _median_speed_leading_edge,
                            "fmt": "-b"}
    time_axis: str = "time" if _balanced_force_ever() else "leading edge progress (phi)"
    _plot_datasets_v_time([speed_data],
                          filename=f"Median leading edge speed v. {time_axis}",
                          limits=limits,
                          ylabel=r"Median leading edge speed, $\Vert\mathbf{v_{veg}}\Vert$",
                          axvline=axvline,
                          suppress_timestep_zero=True)

def _show_piv_speed_v_phi(finished_accumulating: bool, end: bool) -> None:
    """Particle Image Velocimetry - or the one aspect of it that's relevant in this context
    
    Embryo is cylindrically symmetrical. We just want to know the magnitude of the vegetally-pointing
    component of the velocity, as a function of phi and time.
    
    Further, generate strain rates from these values, so plot those here as well.
    """
    if end:
        # Normally we've been accumulating into these lists over multiple timesteps, so we just continue to add to them.
        # But if end, we'll keep things simple by dumping earlier data (if any) and gathering just the current
        # data and plotting it (so, not time averaged as usual).
        _speeds.clear()
        _speeds_particle_phi.clear()

    phandle: tf.ParticleHandle
    for phandle in g.Evl.items():
        particle_position_phi, speed = _phi_and_vegetal_speed(phandle)
        _speeds.append(speed)
        _speeds_particle_phi.append(particle_position_phi)
    for phandle in g.LeadingEdge.items():
        particle_position_phi, speed = _phi_and_vegetal_speed(phandle)
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

    yticks = {"major_range": np.arange(0, 0.102 * np.pi, 0.05 * np.pi),
              "minor_range": np.arange(0, 0.102 * np.pi, 0.01 * np.pi),
              "labels": ["0", r"0.05$\pi$", r"0.10$\pi$"]}
    
    lopsidedness_data: PlotData = {"data": _margin_lopsidedness}
    _plot_datasets_v_time([lopsidedness_data],
                          filename="Margin lopsidedness",
                          limits=(-0.002 * np.pi, 0.102 * np.pi),
                          ylabel="Margin lopsidedness",
                          plot_formats=".-b",
                          yticks=yticks)

def _show_bond_counts() -> None:
    phandle: tf.ParticleHandle
    
    # logical: it's the mean of how many neighbors each particle has:
    mean_bonds_per_particle: float = fmean([len(phandle.bonded_neighbors) for phandle in g.Evl.items()])
    
    # better & faster: it's twice the ratio of bonds to particles. (Have to include leading edge if doing it this way.)
    # On the other hand, I have not tested to be sure BondHandle.items() isn't affected by the phantom-bond bug,
    # something I probably need ToDo.
    # So save this and maybe use it later:
    # mean_bonds_per_particle: float = (2 * len(tf.BondHandle.items()) /
    #                                   (len(g.Evl.items()) + len(g.LeadingEdge.items())))
    
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
    _forces.append(fo.current_total_force())
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
                          ylabel="Leading edge position",
                          plot_formats="b.",
                          yticks=yticks,
                          plot_v_time=True)

def show_graphs(end: bool = False) -> None:
    global _timestep, _previous_leading_edge_phi, _previous_universe_time
    
    if not g.LeadingEdge.items() or g.LeadingEdge.items()[0].frozen_z:
        # Prior to instantiating any LeadingEdge particles, or during the z-frozen phase of equilibration,
        # don't even increment _timestep, so that once we start graphing, it will start at time 0, representing the
        # moment when the leading edge is both present, and free to move. (Frozen/unfrozen state, and pre-/post-
        # instantiation, refer to the initialization methods by config and by graph boundary, respectively.)
        return

    if not _plot_path:
        # if init hasn't been run yet, run it
        _init_graphs()
        _previous_leading_edge_phi = epu.leading_edge_mean_phi()
        _previous_universe_time = tf.Universe.time
        
    if cfg.cell_division_enabled and epu.cell_division_cessation_timestep == 0:
        # We haven't yet detected threshold crossing, so check it.
        if epu.leading_edge_mean_phi() > epu.cell_division_cessation_phi:
            # We've detected that phi has crossed the threshold and cell division has ceased,
            # so record when that happened.
            epu.cell_division_cessation_timestep = _timestep

    # Don't need to add to the graphs every timestep.
    if _timestep % cfg.simple_plot_interval_timesteps == 0 or end:
        # alternative measures of time, available to all plots:
        _timesteps.append(_timestep)
        _leading_edge_phi.append(round(epu.leading_edge_mean_phi(), 4))
        
        _show_progress_graph()
        _show_bond_counts()
        _show_forces()
        _show_cylindrical_straightness()
        _show_margin_population()
        _show_avg_tensions_v_microtime()
        _show_circumferential_edge_tension_v_microtime()
        _show_leading_edge_speed_v_microtime()
        _show_speed_by_phi_diffs()

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
    
    # kludge/hack alert: this is just to quickly test something; not the right way to do this!
    # After a delay, turn off the equilibration flag and set force back to what is configured
    if epu.balanced_force_equilibration_kludge and _timestep > 600:
        if not cfg.run_balanced_force_control:
            fo._force_per_unit_length = cfg.yolk_cortical_tension + cfg.external_force
        epu.balanced_force_equilibration_kludge = False
    
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
            "median_tension_circumferential": _median_tension_circumferential,
            "median_speed_leading_edge": _median_speed_leading_edge,
            "speed_by_phi_diffs": _speed_by_phi_diffs,
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
    global _median_tension_circumferential, _median_speed_leading_edge, _speed_by_phi_diffs
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
    _median_tension_circumferential = d["median_tension_circumferential"]
    _median_speed_leading_edge = d["median_speed_leading_edge"]
    _speed_by_phi_diffs = d["speed_by_phi_diffs"]
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
    
def post_process_graphs(simulation_data: list[dict],
                        range_low: float,
                        range_high: float,
                        include_legends: bool = True,
                        config_section_key: str = "model",
                        first_config_var_key: str = "",
                        second_config_var_key: str = "",
                        num_legend_format: str = "{}",
                        true_legend_format: str = "True",
                        false_legend_format: str = "False",
                        second_true_legend_format: str = "True",
                        second_false_legend_format: str = "False",
                        x_axis_types: list[str] = None,
                        x_axis_types_share_y_limits: bool = False,
                        flip_bool_color: bool = False,
                        desired_height_inches: float = None) -> None:
    """Print various plots with data from multiple simulation runs
    
    All the nested functions access these parameters:
    
    :param simulation_data: contains the entire plot history and config of each simulation run,
        from which we will pull the data needed to draw the composite plots.
    :param range_low: low percentile (should be < 50) to plot with medians to show the range of the data
    :param range_high: high percentile (should be > 50) to plot with medians to show the range of the data
    :param include_legends: if False, ignore the remaining parameters, and plot all the simulations
        ungrouped, all in the same color, and with no legends.
    :param config_section_key: the key for the config section in which to find the variable that we are varying.
        See config.py, get_state().
    :param first_config_var_key: the key for the particular config variable that we are varying.
        See config.py, get_state().
    :param second_config_var_key: If a second treatment is used, assume both treatments are boolean.
    :param num_legend_format: for numerical variables, a string containing a replacement field,
        into which the treatment value for each legend will be inserted to create each legend.
    :param true_legend_format: for boolean variables, a string to use as the legend when the variable is True.
    :param false_legend_format: for boolean variables, a string to use as the legend when the variable is False.
    :param second_true_legend_format: for boolean variables, a string to use in the legend when the variable is True.
    :param second_false_legend_format: for boolean variables, a string to use in the legend when the variable is False.
    :param x_axis_types: list of x axis types to plot, including any or all of: ["phi", "timesteps", "normalized time"]
    :param x_axis_types_share_y_limits: whether to force a consensus set of y-axis limits on all x-axis types.
    :param flip_bool_color: if True, then the config var should be of type bool, and we wish the True value
        (instead of the False value, which is the default) to appear first in the legend and to be plotted
        using cycler color C0.
    :param desired_height_inches: when plotting final version for publication, adjust everything for the desired
        size at publication. Proportions will be maintained, so width is not explicitly specified. This should
        result in appropriate font and axis tick sizes and so on.
    """
    def normalize(datadicts: list[PlotData], suppress_scaling: bool = False) -> None:
        """Calculate the normalized-time axis based on the timestep axis
        
        Normally, the last timestep of each sim maps to 1.0. But, if we are comparing Model 1 to Model 2,
        then that doesn't really make sense. We still want all the Model 1 sims to align, but we don't want
        to stretch them out the same as Model 2, because time 1.0 is meant to represent "completion", i.e.,
        the mean polar angle of the leading edge reaching cfg.stopping_condition_phi, which usually does not
        take place with Model 1. So, find the median stopping time of the Model 1 sims, find its ratio to the
        median stopping time of the Model 2 sims, and normalize the Model 1 sims such that their final timestep
        maps to that fraction.
        """
        datadict: PlotData
        assert all(["timesteps" in datadict for datadict in datadicts]), "Can't normalize; timesteps not present!"
        
        def sim_is_model_1(sim: PlotData) -> bool:
            return sim["model_id"] == 1

        comparing_models_1_and_2: bool = False
        if first_config_var_key == "force_is_weighted_by_distance_from_pole":
            # Ensure there are some of each
            are_model_1: list[bool] = [sim_is_model_1(datadict) for datadict in datadicts]
            comparing_models_1_and_2 = any(are_model_1) and not all(are_model_1)

        def should_scale(sim: PlotData) -> bool:
            return comparing_models_1_and_2 and sim_is_model_1(sim)
        
        model_1_fraction: float = 1.0
        if comparing_models_1_and_2:
            # Since both are present, neither of the following will be empty
            model_1_finals: list[int] = [datadict["timesteps"][-1] for datadict in datadicts
                                         if sim_is_model_1(datadict)]
            model_2_finals: list[int] = [datadict["timesteps"][-1] for datadict in datadicts
                                         if not sim_is_model_1(datadict)]
            model_1_fraction = np.median(model_1_finals).item() / np.median(model_2_finals).item()
            
        for datadict in datadicts:
            timesteps: list[int] = datadict["timesteps"]
            result: np.ndarray = np.array(timesteps) / timesteps[-1]
            if should_scale(datadict):
                result = result * model_1_fraction
            datadict["norm_times"] = list(result)
    
    def color_code_and_clean_up_labels(datadicts: list[PlotData],
                                       use_alpha: bool = False,
                                       extradata: PlotData = None,
                                       extradata_colorindex: int = None) -> None:
        """Color code plot lines according to treatment (parameter value); and only label one plot per treatment
        
        On entry, each PlotData["label"] is a numerical or bool value. Sort the list according to that value (so that
        the labels will be in the right order in the legend); then remove all labels except one per treatment
        (so that each label only appears once in the legend); and then wrap that numerical
        value in a string that explains what it is. Then set distinct plot colors for each treatment,
        but the SAME plot color for the multiple plot lines of the SAME treatment.
        
        When specified for certain bool values, reverse the ordering of the colors & legend.
        I.e., select whether the True or the False will come first in the legend and use cycler color C0.
        This is so that these plots seem more consistent with the other types of plots.
        
        :param datadicts: one PlotData for each line that is to be plotted. "label" field should
            be numerical or bool, representing the treatment.
        :param use_alpha: if True, use this color hack. alpha = 0.5, but brighter colors than the ones in
            the default color cycler, to prevent them from getting all washed out by the white background.
            I'm probably not doing this right.
        :param extradata: An external dataset, not coming from the simulations themselves, that we wish
            to superimpose over the rest of the data. "label" will be used verbatim, so the PlotData should
            specify it as desired. It will be plotted as dots. Since the external data won't have anything
            to do with our sim timesteps, it's intended to be used with normalized time.
        :param extradata_colorindex: which item in the color cycle to use, if you want the dots to match the
            color of one of the treatments in the sim data. If None, it will be plotted in its own color
        """
        if not include_legends:
            return
        
        datadict: PlotData
        true_format: str = true_legend_format
        false_format: str = false_legend_format
        if flip_bool_color:
            # Swap the legend labels
            # This will be only for the first treatment. If we are plotting against two of them,
            # the second one gets whatever it gets.
            true_format = false_legend_format
            false_format = true_legend_format
            
            # To reverse the colors, invert the bools before sorting and assigning the strings
            for datadict in datadicts:
                datadict["label"] = not datadict["label"]
        
        # Create the possible combination labels for two boolean treatments:
        double_bool_label_dict = {0: false_format + " " + second_false_legend_format,
                                  1: false_format + " " + second_true_legend_format,
                                  10: true_format + " " + second_false_legend_format,
                                  11: true_format + " " + second_true_legend_format}
        if second_config_var_key:
            # We are combining two treatments.
            # We can assume both labels are boolean, and we want to sort into four categories.
            # We can treat the True/False values as 0 and 1, and combine them to produce 00, 01, 10, 11;
            # then store that in label, and henceforth ignore second_label (and we can then sort on those):
            for datadict in datadicts:
                datadict["label"] = 10 * bool(datadict["label"]) + datadict["second_label"]
                
        datadicts.sort(key=lambda plot_data: plot_data["label"])
        previous_label: int | float | bool = -1
        cycler_index: int = -1
        
        # For each distinct treatment, provide a different color; and a str label only for one plot of each treatment
        for datadict in datadicts:
            # datadict["label"] is known to be int, float, or bool
            current_label: int | float | bool = datadict["label"]  # type: ignore
            if current_label > previous_label:
                if second_config_var_key:
                    # Convert the 0|1|10|11 into the right combination of strings:
                    datadict["label"] = double_bool_label_dict[current_label]
                elif not isinstance(current_label, bool):
                    datadict["label"] = num_legend_format.format(current_label)
                else:
                    datadict["label"] = true_format if current_label else false_format
                previous_label = current_label
                cycler_index += 1
            else:
                del datadict["label"]
            if use_alpha and cycler_index < 3:
                # Color hack. I don't have time to really master color control in matplotlib. But the
                # default colors look awful when I just decrease alpha, because they get all washed out.
                # This is my attempt to substitute different colors that are brighter, just using my rgb
                # intuition. Can't use the format string for those, so have to use the color parameter instead.
                # Can add more colors to this hack if I want to. Full default color cycle is here:
                # https://matplotlib.org/stable/users/explain/colors/colors.html and the hex definitions of
                # those colors are here: https://gist.github.com/leblancfg/b145a966108be05b4a387789c4f9f474
                # Following are my experiments with substituting better colors (alpha = 0.5/'80' in all):
                # Some bright, pure colors; blue, red, green (not terrible; alpha 0.25 also not terrible):
                # datadict["color"] = f"{['#0000ff80', '#ff000080', '#00ff0080'][cycler_index]}"
                # The first three default color cycler colors from the link above, but with alpha:
                # datadict["color"] = f"{['#5778a480', '#e4944480', '#6a9f5880'][cycler_index]}"
                # Those default colors, brightened up (brightest channel to ff):
                # datadict["color"] = f"{['#5778ff80', '#ff944480', '#6aff5880'][cycler_index]}"
                # Those default colors, with brighter brights and darker darks (not bad):
                datadict["color"] = f"{['#0021ff80', '#ff500080', '#12ff0080'][cycler_index]}"
            else:
                datadict["fmt"] = f"-C{cycler_index}"
        
        if extradata:
            # Now add in the extradata, plot with just small dots on a dotted line, and its own color
            if extradata_colorindex is None:
                extradata_colorindex = cycler_index + 1
            extradata["fmt"] = f".:C{extradata_colorindex}"
            datadicts.append(extradata)
    
    def get_cell_division_cessation_phi(force: bool = False) -> float:
        """Return the phi value where cell division stopped (to be able to mark it on the plot)
        
        Normally (when we are not color coding by cell division enabled vs. disabled), we will assume that
        either all of the sims had cell division enabled, or none of them did. Either way, they will all
        have the same value for cell_division_cessation_phi (0.0 if cell division was disabled) – as long as,
        if enabled, they all had the same cell_division_cessation_percentage, which we also assume to be true.
        We can just grab the value from the first sim in the list. "force" is ignored.
        
        When we ARE color coding by cell division enabled/disabled, we assume we have a mix of both, and we
        don't know which are which. We let the caller decide how to treat it. If force == True, search through
        the list and find one that has phi > 0 (cell division was enabled), and return that. If force == False,
        then just return 0 (meaning the line won't be plotted).
        
        (Note that for multi-plotting, this will only work on plots v. phi. In plots v. time, each simulation
        will have crossed the threshold at a slightly different time, so would make a mess if displayed.)
        """
        if first_config_var_key == "cell_division_enabled" or second_config_var_key == "cell_division_enabled":
            sim: dict
            if force:
                for sim in simulation_data:
                    phi: float = sim["epiboly"]["cell_division_cessation_phi"]
                    if phi > 0.0:
                        return phi
            return 0.0
        return simulation_data[0]["epiboly"]["cell_division_cessation_phi"]

    def show_multi_progress() -> None:
        """Overlay multiple progress plots on one Axes, color-coded by treatment"""
        # extract_simulation_data() puts "leadging_edge_phi" data in the "phi" field of each PlotData; by
        # also passing that as our data_key, we put those values into each PlotData twice. It's the data we
        # want to plot (hence will go in ths "data" field), not the x-axis (so we won't use the "phi" field
        # in this case), but interpolate_and_show_medians() needs it to be there.
        datadicts: list[PlotData] = extract_simulation_data("leading_edge_phi")
        normalize(datadicts)
        
        def kimmel_percent_epiboly_as_phi() -> PlotData:
            """Convert Kimmel et al. 1995 Fig. 12 to plot polar angle instead of percent epiboly"""
            # The following stages (as percent epiboly) and times (as hours post fertilization) come
            # from Kimmel et al. 1995, Table 2 and the text descriptions under their "Stages During the
            # Blastula Period" and "Stages During the Gastrula Period"; and/or from their Fig. 11 legend.
            # Note that we substitute 43% epiboly for what they describe as 30% epiboly.
            kimmel_pct_epiboly: list[int] = [43, 50, 70, 75, 80, 90, 100]
            # Convert:
            kimmel_polar_angle: list[float] = [epu.phi_for_epiboly(pct) for pct in kimmel_pct_epiboly]
            # hours of germ-ring and shield-stage pause; in text they say "about 1 hr";
            # I use 1.1 as estimated from their figure 12:
            pause: float = 1.1
            # These are their reported development times for each of those stages, adjusted for the pause
            # at 50% epiboly so that we can compare apples to apples:
            kimmel_hours: list[float] = [4.67, 5.25, 7.7 - pause, 8 - pause, 8.4 - pause, 9 - pause, 10 - pause]
            # Adjust to measure from 30% epiboly instead of from fertilization, just like our sims:
            offset: float = kimmel_hours[0]
            kimmel_relative_hours: list[float] = [hpf - offset for hpf in kimmel_hours]
            # Just multiply by 100 to get integers without losing precision. We'll use them as the "timesteps"
            # scale. They are not comparable to the timesteps in our simulations, but we don't care, because
            # we are interested in the plot vs. normalized time; and these will normalize just fine;
            kimmel_time_ints: list[int] = [int(round(100 * kimmel_hour)) for kimmel_hour in kimmel_relative_hours]
            
            # Construct a PlotData for the Kimmel data:
            kimmeldata: PlotData = {"data": kimmel_polar_angle,
                                    "phi": kimmel_polar_angle,
                                    "timesteps": kimmel_time_ints,
                                    "model_id": 0,
                                    "label": "Kimmel et al. 1995"}
            # Create the normalized time axis
            normalize([kimmeldata])
            return kimmeldata

        filename = "Leading edge phi"
        ylabel: str = "Leading edge position"
        limits: tuple[float, float] = (np.pi * 7 / 16, np.pi + 0.05)
        yticks = {"major_range": [np.pi / 2, np.pi * 3 / 4, np.pi],
                  "minor_range": [np.pi * 5 / 8, np.pi * 7 / 8],
                  "labels": [r"$\pi$/2", r"3$\pi$/4", r"$\pi$"]}
        
        # interpolate_and_show_medians() uses the nonlocal variable x_axis_types (parameter passed to the
        # enclosing function post_process_graphs()), but in this edge case we want interpolate_and_show_medians()
        # to ignore that and always plot vs. timesteps and normalized time, so (kludge...)
        # temporarily swap it out, then restore it so any other plotting function will still work
        nonlocal x_axis_types
        original_x_axis_types: list[str] = x_axis_types.copy()
        x_axis_types = ["timesteps", "normalized time"]
        interpolate_and_show_medians(datadicts, filename, ylabel, limits, yticks=yticks)
        
        # Plot again with the Kimmel data added in, for purposes of publication. Originally, used color index 3
        # because this was intended for the particular case where the figure was grouped by Model 1/Model 2, and
        # with/without cell division (so 4 separate curves), and we wanted it to match the color of Model 2 with
        # cell division (model flag and cell division flag both == True, so "11", i.e., 3). Later, switched to
        # showing Model 1 and 2 in SEPARATE plots (2 curves each), so Model 2 with cell division would now be
        # color index 1. However, with only 2 curves in the plot, it doesn't seem so important to match the
        # colors that way, and I decided I actually like it better in the nice bold red. So leave it as is.
        # And this time, we only want it for the one x-axis type.
        x_axis_types = ["normalized time"]
        extradata: PlotData = kimmel_percent_epiboly_as_phi()
        interpolate_and_show_medians(datadicts, filename + " plus Kimmel data", ylabel, limits, yticks=yticks,
                                     extradata=extradata, extradata_colorindex=3)
        x_axis_types = original_x_axis_types

        color_code_and_clean_up_labels(datadicts)

        _plot_datasets_v_time(datadicts,
                              filename=f"{filename} v. timesteps",
                              limits=limits,
                              ylabel=ylabel,
                              yticks=yticks,
                              plot_v_time=True,
                              post_process=True,
                              desired_height_inches=desired_height_inches)

        _plot_datasets_v_time(datadicts,
                              filename=f"{filename} v. normalized time",
                              limits=limits,
                              ylabel=ylabel,
                              yticks=yticks,
                              plot_v_time=True,
                              normalize_time=True,
                              post_process=True,
                              desired_height_inches=desired_height_inches)

    def replot_individual_margin_pop(simulation: dict,
                                     limits: tuple[float, float],
                                     plot_num: int) -> None:
        """Plot from a single sim, four different margin population metrics together on a single Axes
        
        Basically reconstruct what _show_margin_population() does, but as an after-the fact-reconstruction.
        Show four metrics together (cumulative migration into, and out of, the margin; total number of margin cells;
        and, only for sims with cell division, show cumulative divisions). But with the following differences:
        - Assume our consensus plots contain two lines each, one with cell division and one without; so, use the
            same colors as those, meaning C0 or C1, depending on whether this sim has cell division.
        - But line style should be like the original, since all 4 lines will be the same color; distinguish them
            by their dots/dashes.
        - Don't add legends based on treatment, since this is just a single sim; instead, the legends will be
            added manually for each metric, as in _show_margin_population().
        - limits must be passed in, not calculated here, because they'll be the same for all sims, based on the
            maximum required for any of the datasets, so that the separate sims, each on their own set of Axes,
            will be comparable.
        - Do both plot v. phi and plot v. timesteps.
        """
        margin_count: list[int] = simulation["plot"]["margin_count"]
        margin_in: list[int] = simulation["plot"]["margin_cum_in"]
        margin_out: list[int] = simulation["plot"]["margin_cum_out"]
        margin_divide: list[int] = simulation["plot"]["margin_cum_divide"]
        leading_edge_phi: list[float] = simulation["plot"]["leading_edge_phi"]
        timesteps: list[int] = simulation["plot"]["timesteps"]
        cell_division_enabled: bool = True if margin_divide else False
        
        has_div: str = "WITH cell division" if cell_division_enabled else "NO cell division"
        color: str = "C0" if flip_bool_color == cell_division_enabled else "C1"
        title: str = "Cell Division Enabled" if cell_division_enabled else "Cell Division Disabled"
        
        margin_count_data: PlotData = {"data": margin_count, "fmt": f".{color}", "label": "Total margin cell count"}
        margin_cum_in_data: PlotData = {"data": margin_in, "fmt": f"--{color}", "label": "Cumulative in"}
        margin_cum_out_data: PlotData = {"data": margin_out, "fmt": f":{color}", "label": "Cumulative out"}
        margin_cum_divide_data: PlotData = {"data": margin_divide, "fmt": f"-{color}", "label": "Cumulative divisions"}
        datasets: list[PlotData] = [margin_count_data, margin_cum_in_data, margin_cum_out_data]
        if cell_division_enabled:
            datasets.append(margin_cum_divide_data)
        plotdata: PlotData
        for plotdata in datasets:
            plotdata["phi"] = leading_edge_phi
            plotdata["timesteps"] = timesteps
            
        _plot_datasets_v_time(datasets,
                              filename=f"Margin cell rearrangement, plus cumulative v. phi, {has_div} {plot_num}",
                              limits=limits,
                              legend_loc="upper left",
                              title=title,
                              post_process=True,
                              desired_height_inches=desired_height_inches)
        _plot_datasets_v_time(datasets,
                              filename=f"Margin cell rearrangement, plus cumulative v. timesteps, {has_div} {plot_num}",
                              limits=limits,
                              legend_loc="upper left",
                              title=title,
                              plot_v_time=True,
                              post_process=True,
                              desired_height_inches=desired_height_inches)

    def replot_individual_margin_pops() -> None:
        """Plot from two individual sims, four different margin population metrics together, but just one sim per Axes

        This one a bit different from the others here. Basically reconstruct what _show_margin_population() does,
        but after-the fact, matching the style of the other post-process plots. To avoid premature
        generalization, we will assume that we are comparing sims that have cell division, with those that do not.
        So to avoid a huge explosion of output files, we plot only if that's the case. Provide a disambiguating
        identifier so the output files will have unique names. And furthermore, plot a maximum of 10 sims.
        
        For each sim, reconstruct that original plot from the data, which shows four metrics together (cumulative
        migration into, and out of, the margin; total number of margin cells; and for the one with cell division,
        cumulative divisions). But unlike the original real-time plots, both should use the same y-limits, based
        on the maximum range required for any of the metrics, so that the two Axes will be comparable. (Once I've
        selected which two to use in my figure, run again with just those two, and the limits will be right.)
        """
        if not include_legends or first_config_var_key != "cell_division_enabled":
            return
        
        all_metrics_data: list[list[int]] = []
        simulation: dict
        for simulation in simulation_data[:10]:
            all_metrics_data.append(simulation["plot"]["margin_count"])
            all_metrics_data.append(simulation["plot"]["margin_cum_in"])
            all_metrics_data.append(simulation["plot"]["margin_cum_out"])
            all_metrics_data.append(simulation["plot"]["margin_cum_divide"])
        limits: tuple[float, float] = _expand_limits_if_needed(limits=(-2, 2), data=all_metrics_data)
        
        plot_num: int = 0
        for simulation in simulation_data[:10]:
            plot_num += 1
            replot_individual_margin_pop(simulation, limits, plot_num)

    def show_multi_margin_pop() -> None:
        """Overlay multiple margin pop plots on one Axes, grouped and color-coded by the provided config_var"""
        axvline: float = get_cell_division_cessation_phi(force=True)

        margin_count_dicts: list[PlotData] = []
        margin_cum_dicts: list[PlotData] = []
        simulation: dict
        for simulation in simulation_data:
            leading_edge_phi: list[float] = simulation["plot"]["leading_edge_phi"]
            timesteps: list[int] = simulation["plot"]["timesteps"]
            model_id: int = 2 if (
                    simulation["config"]["config_values"]["model"]["force_is_weighted_by_distance_from_pole"]
            ) else 1
            label = (None if not include_legends else
                     simulation["config"]["config_values"][config_section_key][first_config_var_key])
            second_label = (None if not second_config_var_key else
                            simulation["config"]["config_values"][config_section_key][second_config_var_key])
            
            margin_count: PlotData = {"data": simulation["plot"]["margin_count"],
                                      "phi": leading_edge_phi,
                                      "timesteps": timesteps,
                                      "model_id": model_id,
                                      "label": label,
                                      "second_label": second_label}
            margin_count_dicts.append(margin_count)
            
            margin_cum_in: list[int] = simulation["plot"]["margin_cum_in"]
            margin_cum_out: list[int] = simulation["plot"]["margin_cum_out"]
            margin_cum_total: list[int] = list(np.add(margin_cum_in, margin_cum_out))
            margin_cum: PlotData = {"data": margin_cum_total,
                                    "phi": leading_edge_phi,
                                    "timesteps": timesteps,
                                    "model_id": model_id,
                                    "label": label,
                                    "second_label": second_label}
            margin_cum_dicts.append(margin_cum)
            
        normalize(margin_count_dicts)
        normalize(margin_cum_dicts)
        count_filename: str = "Margin cell count"
        cum_filename: str = "Margin cell rearrangement, cumulative"
        count_ylabel: str = "Margin cell count"
        cum_ylabel: str = "Cumulative edge\nrearrangement events"
        default_limits: tuple[float, float] = (-2, 10)
        interpolate_and_show_medians(margin_count_dicts, count_filename, count_ylabel, default_limits, axvline)
        interpolate_and_show_medians(margin_cum_dicts, cum_filename, cum_ylabel, default_limits, axvline)

        color_code_and_clean_up_labels(margin_count_dicts)
        color_code_and_clean_up_labels(margin_cum_dicts)
        
        all_count_data: list[list[int]] = [datadict["data"] for datadict in margin_count_dicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_count_data)
        plot_datasets_v_selected_time_proxies(margin_count_dicts,
                                              filename=count_filename,
                                              limits=limits,
                                              axvline=axvline)
        
        all_cum_data: list[list[int]] = [datadict["data"] for datadict in margin_cum_dicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_cum_data)
        plot_datasets_v_selected_time_proxies(margin_cum_dicts,
                                              filename=cum_filename,
                                              limits=limits,
                                              axvline=axvline)

    def plot_datasets_v_selected_time_proxies(datadicts: list[PlotData],
                                              filename: str,
                                              ylabel: str = None,
                                              limits: tuple[float, float] = None,
                                              axvline: float = None,
                                              yticks: dict = None,
                                              suppress_timestep_zero: bool = False) -> None:
        """Send data to _plot_datasets_v_time() for each selected time axis proxy.
        
        :param datadicts: one PlotData for each simulation to be plotted
        :param filename: will be used as part of the filename for the saved plots.
        :param ylabel: title of the y-axis.
        :param limits: y-axis limits
        :param yticks: special tick marks for the y-axis.
        :param axvline: assumed to be identical for all simulations v. phi (otherwise you wouldn't be able to
            plot it), so calculated once by caller and passed in. Only to be used for plots v. phi, not time.
        :param suppress_timestep_zero: don't plot the first point in each dataset (regardless of x_axis_type).
        """
        x_axis_type: str
        filename_suffix: str = f", grouped by {first_config_var_key}" if include_legends else ""
        filename_suffix += f" and {second_config_var_key}" if second_config_var_key else ""
        for x_axis_type in x_axis_types:
            _plot_datasets_v_time(datadicts,
                                  filename=f"{filename} v. {x_axis_type}{filename_suffix}",
                                  limits=limits,
                                  ylabel=ylabel,
                                  axvline=axvline if x_axis_type == "phi" else None,
                                  yticks=yticks,
                                  plot_v_time=(x_axis_type != "phi"),
                                  normalize_time=(x_axis_type == "normalized time"),
                                  suppress_timestep_zero=suppress_timestep_zero,
                                  post_process=True,
                                  desired_height_inches=desired_height_inches)

    def compute_filtered_medians(x: list[float],
                                 all_data: list[list[float]],
                                 exclusion_flag: float,
                                 remove_bias: bool = False
                                 ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Compute the median (and low/high percentile) of each column in all_data after filtering out invalid values.
        We assume the excluded values represent sims that ended early, so if, for high values of x
        in a given row, the data are equal to exclusion_flag, that means that sim was finished by that point.
        The flag should not be included in the data.
        
        This function was helpfully written by ChatGPT, according to my specifications. ChatGPT also
        helped me design the algorithm by helping me to understand alternative statistical approaches
        for dealing with biased data. The full conversation can be found here:
        https://chatgpt.com/share/67a065a6-6c58-8013-b0d8-0ff1dddcc45a

        Each column represents a dataset for a specific x-coordinate. Values equal to `exclusion_flag` are
        removed. Optionally (if remove_bias is True), the same number of lowest remaining values are also
        removed before computing the median. Columns where all values are excluded are omitted.

        :param x: List of x-coordinates corresponding to columns in `all_data`.
        :param all_data: 2D list where each row is a dataset, and each column corresponds to an x-coordinate.
        :param exclusion_flag: Value indicating an invalid data point, which should be excluded.
        :param remove_bias: If True, assume the invalid data points represent a high data value, so that the
            remaining values would be biased toward lower value. In each column, remove an equivalent number
            of low values to the number of excluded invalid values, to try to account for this bias.
            ToDo: this works great for lopsidedness! If I ever want to use this approach for Straightness
             Index, I think it would be the opposite: I'd want to compensate by removing high values rather
             than low. For now, not needed.
            
        :return: A tuple containing:
            - A filtered list of x-coordinates.
            - The computed median values for the remaining data in each column.
            - The computed low percentile for the same data.
            - The computed high percentile for the same data.
        """
        # Convert input data to a NumPy array for efficient filtering
        data_array: np.ndarray = np.array(all_data, dtype=float)
    
        # Lists to store the filtered x-values and their corresponding median/low/high data values
        filtered_x: list[float] = []
        filtered_medians: list[float] = []
        filtered_range_lows: list[float] = []
        filtered_range_highs: list[float] = []
    
        # Iterate over each column in all_data
        for col_idx in range(data_array.shape[1]):
            column_values: np.ndarray = data_array[:, col_idx]
        
            # Exclude values equal to exclusion_flag
            valid_values: np.ndarray = column_values[column_values != exclusion_flag]
        
            # Determine how many values were excluded
            num_excluded: int = len(column_values) - len(valid_values)
        
            # Skip this column if all values would be removed
            num_to_remove: int = num_excluded if remove_bias else 0
            if len(valid_values) > num_to_remove:
                if remove_bias:
                    # Sort and remove the lowest `num_to_remove` values
                    valid_values.sort()
                    valid_values = valid_values[num_to_remove:]
            
                # Compute the median and percentile range, and store results
                nd_low, nd_median, nd_high = np.percentile(valid_values, [range_low, 50, range_high])
                filtered_x.append(x[col_idx])
                filtered_medians.append(float(nd_median))
                filtered_range_lows.append(float(nd_low))
                filtered_range_highs.append(float(nd_high))
    
        return filtered_x, filtered_medians, filtered_range_lows, filtered_range_highs

    def interpolate_and_show_medians(rawdicts: list[PlotData],
                                     filename: str,
                                     ylabel: str,
                                     default_limits: tuple[float, float],
                                     axvline: float = None,
                                     yticks: dict = None,
                                     suppress_timestep_zero: bool = False,
                                     extradata: PlotData = None,
                                     extradata_colorindex: int = None,
                                     remove_bias: bool = False) -> None:
        """Combine multiple datasets into composite metrics, one per 'treatment'.

        'Treatment' refers to the different values of a single variable that we are contrasting.

        The handling of the labels and legends assumes there is more than one treatment being
        compared in the plot, but if we ever need to do this for just a single treatment, that can
        be tweaked as necessary.

        Note that when balanced-force control data is included, the plots v. phi will be garbage because
        there is no phi progression; and the plots v. normalized time should work but "1.0" no longer means
        epiboly completion, but just reaching a predetermined timestep. Those control datasets should look
        the same on the timestep and normalized time plots, just with the x-axes labeled accordingly.

        :param rawdicts: one PlotData for each simulation that is to be plotted. It should have already
            been normalized (normalized time data calculated for each simulation). "label" field should
            be numerical, representing the treatment.
        :param filename: will be used as part of the filename for the saved plots.
        :param ylabel: title of the y-axis.
        :param default_limits: y-axis limits for whatever data was passed. These will be expanded if
            the range of the actual data exceeds the default_limits.
        :param yticks: special tick marks for the y-axis.
        :param axvline: assumed to be identical for all simulations v. phi (otherwise you wouldn't be able to
            plot it), so calculated once by caller and passed in. Only to be used for plots v. phi, not plots v. time.
        :param suppress_timestep_zero: don't plot the first point in each dataset (regardless of x_axis_type).
        :param extradata: Pass-through to color_code_and_clean_up_labels(); see definition there.
        :param extradata_colorindex: Pass-through to color_code_and_clean_up_labels(); see definition there.
        :param remove_bias: If True, assume the invalid data points represent a high data value, so that the
            remaining values would be biased toward lower value. In each column, remove an equivalent number
            of low values to the number of excluded invalid values, to try to account for this bias.
            (Passthrough to compute_filtered_medians().)
        """
        def is_model_1(sim_list: list[PlotData]) -> bool:
            # We assume every sim within a treatment is the same model, so we can check it by looking at sim_list[0].
            if not sim_list:
                # None or empty list
                return False
            return sim_list[0]["model_id"] == 1
        
        def calculate_percentiles(sim_list: list[PlotData]) -> PlotData:
            """The simpler procedure: Just take the median and percentile range of each column"""
            # Each PlotData in the list contains the same relevant x-axis and other fields, we want
            # our result to have all that, and just combine the "data" fields.
            result: PlotData = sim_list[0].copy()
            all_data: list[list[float]] = [sim["data"] for sim in sim_list]
            
            nd_lows, nd_medians, nd_highs = np.percentile(all_data, [range_low, 50, range_high], axis=0)
            result["data"] = nd_medians.tolist()
            result["range_low"] = nd_lows.tolist()
            result["range_high"] = nd_highs.tolist()
            return result
        
        def calculate_percentiles_allow_truncated_domains(x_axis_type: str, sim_list: list[PlotData]) -> PlotData:
            """Allow for truncated domains; same idea, procedure just a bit more involved"""
            # Each PlotData in the list contains the same relevant x-axis and other fields, we want
            # our result to have all that, and just combine the "data" fields.
            result: PlotData = sim_list[0].copy()
            all_data: list[list[float]] = [sim["data"] for sim in sim_list]
            
            filtered_x: list[float]
            medians: list[float]
            lows: list[float]
            highs: list[float]
            x_axis: list[float] = result["phi"] if x_axis_type == "phi" else result["norm_times"]
            filtered_x, medians, lows, highs = compute_filtered_medians(x_axis,
                                                                        all_data,
                                                                        exclusion_flag=beyond_domain,
                                                                        remove_bias=remove_bias)
            if x_axis_type == "phi":
                result["phi"] = filtered_x
            else:
                result["norm_times"] = filtered_x
            result["data"] = medians
            result["range_low"] = lows
            result["range_high"] = highs
            return result
        
        # At least for now, don't try this with timesteps as the x-axis. I don't think it can work,
        # because the duration of a sim can vary so much; and I don't think I need it.
        # This means we're plotting against phi, or normalized time, or both.
        x_axis_type: str
        x_axes: list[str] = x_axis_types.copy()
        if "timesteps" in x_axes:
            x_axes.remove("timesteps")
        
        # Sort the list[Plotdata] into treatments, and for each treatment, make a duplicate sub-list
        # of the data, once for each x_axis_type
        sim_list: list[PlotData]                                # all the same treatment and x-axis type
        x_axis_lists: dict[str, list[PlotData]]                 # map x-axis type to its data (all the same treatment)
        treatments: dict[str, dict[str, list[PlotData]]] = {}   # map treatment to a dict of x_axis_lists
        treatment_medians: dict[str, dict[str, PlotData]] = {}  # map treatment to a dict of consensus (median) PlotData
        treatment_key: str

        for rawdict in rawdicts:
            treatment_key = str(rawdict["label"]) + str(rawdict["second_label"])
        
            # Create an appropriate key-value pair in which to create an entry for each treatment
            # if it doesn't yet exist. (I.e., the first time each treatment is encountered.)
            if treatment_key not in treatments:
                treatments[treatment_key] = {}
                treatment_medians[treatment_key] = {}
                # And in each treatment dictionary, add lists for each x_axis_type,
                # in which to store the current rawdict
                for x_axis_type in x_axes:
                    treatments[treatment_key][x_axis_type] = []

            # Add each rawdict to the appropriate lists. But a copy, possibly modified:
            newdict: PlotData = rawdict.copy()
            plotdata_key: str
            # Make it a deep copy:
            for plotdata_key in ["data", "phi", "timesteps", "norm_times"]:
                # (Type checker doesn't like variables as keys; it's fine.)
                newdata: list = newdict[plotdata_key].copy()  # type: ignore
                if suppress_timestep_zero:
                    newdata = newdata[1:]
                newdict[plotdata_key] = newdata  # type: ignore
            for x_axis_type in x_axes:
                treatments[treatment_key][x_axis_type].append(newdict)
            
        # Now for each and every PlotData, interpolate. All PlotData within a given x_axis_type and treatment,
        # will be interpolated the same way (i.e., with the same x-axis points and spacing, though some may
        # get more points than others, depending on the domain of the data). In general the domains within
        # x_axis_type:treatment should be the same, but there will be cases when some extend further to the
        # right than others.
        beyond_domain: float = -10  # a value we don't expect in the data, to flag x is beyond end-of-sim
        for treatment_key, x_axis_lists in treatments.items():
            for x_axis_type, sim_list in x_axis_lists.items():
                sim: PlotData
                
                # First, figure out the min and max x that we want in our interpolated plots.
                # They should be within the common range of all plots, so that we can always interpolate
                # (no need to extrapolate).
                # For plotting v. phi, each sim (that reached completion) should have a point >= stopping condition.
                max_x: float = 1.0 if x_axis_type == "normalized time" else cfg.stopping_condition_phi
                
                # The lowest x, when plotting v. phi, is stochastic, so may differ among replicates.
                # And for both kinds of plots, we may have removed the first data point (if suppress_timestep_zero).
                # So find the lowest x of each replicate (the left edge of each plot), and for our min x
                # select the RIGHT-most of those, i.e. the furthest right of all the left edges.
                # That way, each sim will have a point <= the left edge of the interpolated axis.
                all_left_edges: list[float] = [sim["phi"][0] if x_axis_type == "phi" else sim["norm_times"][0]
                                               for sim in sim_list]
                min_x: float = max(all_left_edges)
                
                # Prepare to handle cases where the individual sims don't have similar domains (x-axis range)
                # This is an issue mainly with Model 1 (unregulated force), where each sim gets terminated when
                # a projection gets too near the vegetal pole, leading to different termination points for
                # each. However, it's also occasionally needed for Model 2, if we're including some treatments
                # that never complete epiboly on their own and therefore are terminated manually.
                # This approach should always work, so just do it always. However, I'm retaining (commented out)
                # the simpler procedure that I formerly used with Model 2: both because I haven't yet exhaustively
                # tested the blanket approach with ALL possible use cases, and because it presents a simpler version
                # of the algorithm for the benefit of any future reader trying to understand what's being done here.
                # "left" and "right" tell the interp() function how to behave when attempting to interpolate
                # outside the domain. When used, they need to match our data type (float). Use them to flag
                # right-truncated data:
                left = None
                right = beyond_domain
                
                # # (Previously, when allowing for truncated domain plotting only for Model 1,
                # # but keeping it simple for Model 2: )
                # left = None
                # right = None
                # if is_model_1(sim_list):
                #     right = beyond_domain
                    
                # Get the new x axis
                x_interpolated: np.ndarray = np.linspace(min_x, max_x, num=21)
                for sim in sim_list:
                    # Interpolate, and replace the original data with the new data
                    x_axis: list[float] = sim["phi"] if x_axis_type == "phi" else sim["norm_times"]
                    sim["data"] = np.interp(x_interpolated, x_axis, sim["data"], left=left, right=right).tolist()
                    # (x_interpolated contains floats, and I'm assigning it to a list[float], but for some reason
                    # type checker thinks tolist() is returning 'object')
                    x_axis = x_interpolated.tolist()  # type: ignore
                    if x_axis_type == "phi":
                        sim["phi"] = x_axis
                    else:
                        sim["norm_times"] = x_axis
                        
        # Now we have for each x_axis type in each treatment, a list of PlotData with interpolations.
        # From each of those lists, create a single PlotData with the median of all the values. These
        # are what we want to plot.
        for treatment_key, x_axis_lists in treatments.items():
            for x_axis_type, sim_list in x_axis_lists.items():
                result: PlotData = calculate_percentiles_allow_truncated_domains(x_axis_type, sim_list)
                
                # # (Previously, when allowing for truncated domain plotting only for Model 1,
                # # but keeping it simple for Model 2)
                # result: PlotData
                # if is_model_1(sim_list):
                #     result = calculate_percentiles_allow_truncated_domains(x_axis_type, sim_list)
                # else:
                #     result = calculate_percentiles(sim_list)
                    
                treatment_medians[treatment_key][x_axis_type] = result
                
        # Now gather the sets of interpolations we want to plot together. One interpolated over the phi axis,
        # and one interpolated over the normalized time axis:
        phi_dicts: list[PlotData] = []
        normtime_dicts: list[PlotData] = []
        for treatment_key, interpolated_data in treatment_medians.items():
            if "phi" in x_axes:
                phi_dicts.append(interpolated_data["phi"])
            if "normalized time" in x_axes:
                normtime_dicts.append(interpolated_data["normalized time"])
            
        color_code_and_clean_up_labels(normtime_dicts, extradata=extradata, extradata_colorindex=extradata_colorindex)
        color_code_and_clean_up_labels(phi_dicts)
    
        # Calculate two sets of y-limits, one for plotting without ranges, and one for plotting with...
        #
        # For using consensus y-limits:
        # Since the data was interpolated differently in the time dict vs the phi dict, they might have slightly
        # different ranges. So to make the y-axis scales identical on the two plots we'll generate, combine
        # ALL the data from both to determine the y limits.
        all_data: list[list[float]] = []
        all_range_data: list[list[float]] = []
        if "normalized time" in x_axes:
            all_data.extend([plot_data["data"] for plot_data in normtime_dicts])
            all_range_data.extend([plot_data["range_low"] for plot_data in normtime_dicts
                                   # if extradata was present, that one won't have a range, so filter it out
                                   if "range_low" in plot_data])
            all_range_data.extend([plot_data["range_high"] for plot_data in normtime_dicts
                                   if "range_high" in plot_data])
        if "phi" in x_axes:
            all_data.extend([plot_data["data"] for plot_data in phi_dicts])
            all_range_data.extend([plot_data["range_low"] for plot_data in phi_dicts])
            all_range_data.extend([plot_data["range_high"] for plot_data in phi_dicts])
    
        filename_suffix: str = f", grouped by {first_config_var_key}" if include_legends else ""
        filename_suffix += f" and {second_config_var_key}" if second_config_var_key else ""
        for x_axis_type in x_axes:
            dicts_list: list[PlotData] = (phi_dicts if x_axis_type == "phi" else normtime_dicts)
        
            # if not using consensus y-limits:
            # in this case, each of the plots gets its own tailored y-limits.
            # They might each be very different:
            current_data: list[list[float]] = [plot_data["data"] for plot_data in dicts_list]
            current_range_data: list[list[float]] = []
            current_range_data.extend([plot_data["range_low"] for plot_data in dicts_list
                                       # if extradata was present, that one won't have a range, so filter it out
                                       if "range_low" in plot_data])
            current_range_data.extend([plot_data["range_high"] for plot_data in dicts_list
                                       if "range_high" in plot_data])
        
            the_data:       list[list[float]] = all_data if x_axis_types_share_y_limits else current_data
            the_range_data: list[list[float]] = all_range_data if x_axis_types_share_y_limits else current_range_data
            limits:       tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=the_data)
            range_limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=the_range_data)
        
            # Ranges are present, but first plot without them
            _plot_datasets_v_time(datadicts=dicts_list,
                                  filename=f"{filename} v. {x_axis_type}, Median{filename_suffix}",
                                  limits=limits,
                                  ylabel=ylabel,
                                  axvline=axvline if x_axis_type == "phi" else None,
                                  yticks=yticks,
                                  plot_v_time=(x_axis_type != "phi"),
                                  normalize_time=(x_axis_type == "normalized time"),
                                  post_process=True,
                                  desired_height_inches=desired_height_inches)
            
            # Plot again with ranges: same params except range_limits instead of limits, and with an extended filename
            _plot_datasets_v_time(datadicts=dicts_list,
                                  filename=f"{filename} v. {x_axis_type}, Median{filename_suffix} (with ranges)",
                                  limits=range_limits,
                                  ylabel=ylabel,
                                  axvline=axvline if x_axis_type == "phi" else None,
                                  yticks=yticks,
                                  plot_v_time=(x_axis_type != "phi"),
                                  normalize_time=(x_axis_type == "normalized time"),
                                  post_process=True,
                                  plot_ranges=True,
                                  desired_height_inches=desired_height_inches)

    def show_multi_straightness() -> None:
        """Overlay multiple Straightness Index plots on one Axes, grouped and color-coded by the provided config_var"""
        axvline: float = get_cell_division_cessation_phi()
        datadicts: list[PlotData] = extract_simulation_data("straightness_cyl")
        normalize(datadicts)
        
        filename: str = "Straightness Index"
        ylabel: str = "Straightness Index (SI)"
        default_limits: tuple[float, float] = (0.9, 1.001)
        # Use axvline here, because I want it where I use this fig.:
        interpolate_and_show_medians(datadicts, filename, ylabel, default_limits, axvline)

        color_code_and_clean_up_labels(datadicts, use_alpha=True)
        
        all_data: list[list[float]] = [data["data"] for data in datadicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_data)

        # but don't use axvline here, because I don't want it where I use this fig.:
        plot_datasets_v_selected_time_proxies(datadicts, filename, ylabel, limits)

    def show_multi_lopsidedness() -> None:
        """Overlay multiple Lopsidedness plots on one Axes, grouped and color-coded by the provided config_var"""
        datadicts: list[PlotData] = extract_simulation_data("margin_lopsidedness")
        normalize(datadicts)
    
        filename: str = "Margin lopsidedness"
        ylabel: str = "Margin lopsidedness"
        default_limits: tuple[float, float] = (-0.002 * np.pi, 0.102 * np.pi)
        yticks = {"major_range": np.arange(0, 0.102 * np.pi, 0.05 * np.pi),
                  "minor_range": np.arange(0, 0.102 * np.pi, 0.01 * np.pi),
                  "labels": ["0", r"0.05$\pi$", r"0.10$\pi$"]}
        interpolate_and_show_medians(datadicts, filename, ylabel, default_limits, yticks=yticks, remove_bias=True)
    
        color_code_and_clean_up_labels(datadicts, use_alpha=True)
    
        all_data: list[list[float]] = [data["data"] for data in datadicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_data)
    
        plot_datasets_v_selected_time_proxies(datadicts, filename, ylabel, limits, yticks=yticks)

    def show_multi_tension() -> None:
        """Overlay multiple leading edge tension plots, grouped and color-coded by the provided config_var
        
        Deprecated in favor of show_multi_circumferential_tension()
        """
        axvline: float = get_cell_division_cessation_phi()
        datadicts: list[PlotData] = extract_simulation_data("median_tension_leading_edge")
        normalize(datadicts)
        
        filename: str = "Leading edge tension"
        ylabel: str = "Average tension at leading edge"
        default_limits: tuple[float, float] = (-0.01, 0.2)
        interpolate_and_show_medians(datadicts, filename, ylabel, default_limits, axvline)
        
        color_code_and_clean_up_labels(datadicts)
        
        all_data: list[list[float]] = [data["data"] for data in datadicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_data)
        
        plot_datasets_v_selected_time_proxies(datadicts, filename, ylabel, limits, axvline)

    def show_multi_circumferential_tension() -> None:
        """Overlay multiple **circumferential** tension plots, grouped and color-coded by the provided config_var"""
        axvline: float = get_cell_division_cessation_phi(force=True)
        datadicts: list[PlotData] = extract_simulation_data("median_tension_circumferential")
        normalize(datadicts)
    
        filename: str = "Circumferential tension"
        ylabel: str = "Circumferential tension"
        default_limits: tuple[float, float] = (-0.01, 0.2)
        interpolate_and_show_medians(datadicts, filename, ylabel, default_limits, axvline)
    
        color_code_and_clean_up_labels(datadicts)
    
        all_data: list[list[float]] = [data["data"] for data in datadicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_data)
    
        plot_datasets_v_selected_time_proxies(datadicts, filename, ylabel, limits, axvline)

    def show_multi_speed_leading_edge_velocity_based() -> None:
        """Overlay multiple leading-edge speed plots, grouped and color-coded by the provided config_var
        
        Deprecated in favor of show_multi_speed_leading_edge_position_based()
        """
        axvline: float = get_cell_division_cessation_phi()
        datadicts: list[PlotData] = extract_simulation_data("median_speed_leading_edge")
        normalize(datadicts)

        filename: str = "Leading edge speed (velocity based)"
        ylabel: str = "Average vegetalward speed"
        default_limits: tuple[float, float] = (-0.005, 0.08)
        interpolate_and_show_medians(datadicts, filename, ylabel, default_limits, axvline, suppress_timestep_zero=True)

        color_code_and_clean_up_labels(datadicts)

        all_data: list[list[float]] = [data["data"] for data in datadicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_data)

        plot_datasets_v_selected_time_proxies(datadicts, filename, ylabel, limits, axvline, suppress_timestep_zero=True)

    def show_multi_speed_leading_edge_position_based() -> None:
        """Overlay multiple leading-edge speed plots, grouped and color-coded by the provided config_var"""
        axvline: float = get_cell_division_cessation_phi()
        datadicts: list[PlotData] = extract_simulation_data("speed_by_phi_diffs")
        normalize(datadicts)
    
        filename: str = "Leading edge speed (position based)"
        ylabel: str = "Epiboly speed"
        default_limits: tuple[float, float] = (-0.0005, 0.03)
        interpolate_and_show_medians(datadicts, filename, ylabel, default_limits, axvline, suppress_timestep_zero=True)
    
        color_code_and_clean_up_labels(datadicts)
    
        all_data: list[list[float]] = [data["data"] for data in datadicts]
        limits: tuple[float, float] = _expand_limits_if_needed(limits=default_limits, data=all_data)
    
        plot_datasets_v_selected_time_proxies(datadicts, filename, ylabel, limits, axvline, suppress_timestep_zero=True)
        
    def extract_simulation_data(data_key: str) -> list[PlotData]:
        return [{
                "data": simulation["plot"][data_key],
                "phi": simulation["plot"]["leading_edge_phi"],
                "timesteps": simulation["plot"]["timesteps"],
                "model_id": 2 if (
                        simulation["config"]["config_values"]["model"]["force_is_weighted_by_distance_from_pole"]
                        ) else 1,
                "label": (None if not include_legends else
                          simulation["config"]["config_values"][config_section_key][first_config_var_key]),
                "second_label": (None if not second_config_var_key else
                                 simulation["config"]["config_values"][config_section_key][second_config_var_key])
                } for simulation in simulation_data]

    def show_multi_tension_hello_world() -> None:
        """Overlay multiple tension plots on one Axes: all cells vs. leading edge cells, in different colors
        
        This is the first multi-plot I tried. I'm doing things a bit differently now so no longer using this,
        but keep it around for now, in case I want something like it back again.
        """
        axvline: float = get_cell_division_cessation_phi()
        
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

    def handle_balanced_force_equilibration_kludge() -> None:
        nonlocal x_axis_types
        model_control_values: dict = simulation_data[0]["config"]["config_values"]["model control"]
        if "balanced_force_equilibration_kludge" in model_control_values:
            # I want to still be able to read in data from before I created that, so only proceed if it's there.
            # If it's not there (older sims), then we can treat as if the value is False (no need to do anything).
            # In case of a mixed set (old and new), then we can also assume it's False in any simulation that has it,
            # because you'd never mix simulations with the value set to true, with ones that don't have it.
            if model_control_values["balanced_force_equilibration_kludge"]:
                # If it's there and True, safe to assume it's True for all simulations. Let's override the config
                # and see all x_axis_types. Note that when plotting v. phi, the equilibration phase is all compressed
                # because phi isn't changing, but it's still helpful to see the plot.
                # This is just for testing the idea of a balanced force equilibration. If I ever decide to add this
                # to the model everywhere, better to just exclude that equilibration phase from plotting.
                x_axis_types = ["phi", "timesteps", "normalized time"]

    if x_axis_types is None:
        x_axis_types = ["phi", "timesteps", "normalized time"]
    handle_balanced_force_equilibration_kludge()
    
    _init_graphs()
    # print(simulation_data)
    
    # show_multi_tension()
    show_multi_circumferential_tension()
    # show_multi_speed_leading_edge_velocity_based()
    show_multi_speed_leading_edge_position_based()
    show_multi_straightness()
    show_multi_lopsidedness()
    show_multi_margin_pop()
    show_multi_progress()
    replot_individual_margin_pops()

if __name__ == "__main__":
    # test my syntax, and understanding of numpy's axis parameter.
    sim_list: list[PlotData] = [{"data": [1, 2, 3]},
                                {"data": [4, 5, 6]}]
    print(f"sim_list: {sim_list}")
    data_list = [sim["data"] for sim in sim_list]
    print(f"type(data_list) (should be list[list[float]]): {type(data_list)}")
    print(f"data_list: {data_list}")
    np_medians = np.median(data_list, axis=0)
    print(f"type(np_medians) (should be np.ndarray): {type(np_medians)}")
    print(f"np_medians: {np_medians}")
    median_data = np_medians.tolist()  # type: ignore
    print(f"type(median_data) (should be list[float]): {type(median_data)}")
    print(f"median_data: {median_data}")
    
    # test what happens with numpy.interp() if values along new axis are beyond the range of the old axis.
    # (I.e., if testing a new x value beyond the original domain, where extrapolation would be needed.)
    xp = fp = [1, 2, 3, 4]
    x = np.linspace(0, 5, num=51)
    print(f"\nxp = fp = {xp}")
    print(f"new x = {x}")
    y = np.interp(x, xp, fp, left=-1, right=-1)
    print(f"y = {y}")
    # Result: if left=right=None, you just get horizontal lines outside the range.
    # if left=right=complex, it's only allowed if fp is complex. (I tried it.)
    # So, to specify a value that is used as a flag for being outside the range, you have to pick something
    # that you know is outside the possible range of y.
