"""Config

Flags and magic numbers, especially ones used in more than one place.
"""
from enum import Enum
import math

# Temporary (and not bothering to export):
# ToDo: Finally works, but changes coming in next release. Once that's stabilized, make this the new normal.
use_alt_cell_splitting_method: bool = True

# Batch execute: if True, redirect stdout & stderr to a file. (Probably want to override export root, too.)
batch_execute: bool = False

# Export path: if override False, export to user's home directory; else to the provided path. Mainly
# intended for batch processing, but can use independently as well.
override_export_root_directory: bool = False
export_root_directory_name: str = "/N/scratch/sminsuk"

# For normal initialization, leave blank. To start from a previously exported state, provide the
# directory name of the export. This is the parent directory with the datetime in the name. Script
# will search in the "Sim_state" subdirectory to find the most recent export, and start with that.
initialization_directory_name: str = ""

# Just like it says on the tin. Since config is saved as part of state export as metadata for this run, this
# provides for a free-form comment at the top of that file, for a permanent commentary on the generated output.
# That line at top of the export file can then be edited after the run is over to add comment on results.
comment: str = ""

# -------------------- The model --------------------

# Tissue Forge time increment. Tissue Forge default value is 0.01. If using a different value, consider
# adjusting time_avg_accumulation_steps to compensate, which is defined in terms of timesteps (it counts
# dt intervals). (Other similar values – see plotting_interval_timesteps, sim_state_timesteps_per_export,
# screenshots_timesteps_per_export – are config'ed as units of Universe.time and then automatically converted
# to timesteps, so that you can adjust dt without having to make compensating adjustments in those. But I
# wasn't sure what exactly is the best behavior for time_avg_accumulation_steps, so I'm leaving it alone
# for now. May need to manually adjust as appopriate.)
dt: float = 0.1

# Use newer init method (discover leading edge of particles using graph theory definition of "boundary"; as opposed
# to the older method of arbitrarily deciding how many edge particles to have, and creating a ring of them.
# If true, then config variable num_leading_edge_points is ignored.
initialization_algo_graph_based: bool = True

# Cell division: whether or not, and how, and how much:
cell_division_enabled: bool = True
# Cell division rate parameters:
# Approximate number of divisions to take place (the actual value will be stochastically determined):
total_epiboly_divisions: int = 412  # (Campinho et al., 2013; a 62% increase)
# The percentage of epiboly by which total_epiboly_divisions will be reached, and after which cell division ceases:
cell_division_cessation_percentage: int = 55
# Spatial distribution of cell division events: i.e. how to select particles to divide. Uniform vs. by tension
# vs. largest first, are mutually exclusive; bias by tension squared is an option under bias by tension
cell_division_largest_first: bool = True
cell_division_biased_by_tension: bool = False
tension_squared: bool = False  # (ignored unless cell_division_biased_by_tension is True)

# Useful to turn this off while tuning setup and equilibration. When external force is artificially low,
# and if Angle bonds too high, they cause instability and big waviness in the leading edge. (Seems fixed now.)
# (Deprecated: turns out these not actually needed, and Mullins lab (experimentalists) agree the model looks
# much more like real zebrafish embryos without this. Leave False from now on.)
angle_bonds_enabled: bool = False

# Prevent holes (especially when cell division is disabled) by giving particles a nudge away from the
# particles they are bound to, and hence toward open space. (Deprecated! Appears that the way I switched up
# the topological constraints - neighbor count and bond angle - surprisingly solved the problem better than
# this algorithm did, making this an unnecessary complication. Leave False from now on!)
space_filling_enabled: bool = False
k_particle_diffusion: float = 0.9

# Starting point of the simulation. Note that 43% is the true value for the misnomer "30% epiboly")
epiboly_initial_percentage: int = 43
epiboly_initial_num_evl_cells: int = 660  # (Campinho et al., 2013)

# This is the size of the particle, not of the cell. The size of the cell will be derived from
# epiboly_initial_percentage and epiboly_initial_num_evl_cells.
evl_particle_radius: float = 0.045

# Search for neighbors within this distance (multiple of cell radius) to set up initial bond network.
min_neighbor_initial_distance_factor: float = 1.5

# Some items for Potential- and Bond-making:
harmonic_repulsion_spring_constant: float = 0.5
harmonic_spring_constant: float = 0.5
harmonic_edge_spring_constant: float = 0.5
harmonic_yolk_evl_spring_constant: float = 4

# Bonds on smaller cells need smaller spring constant. For each bond, divide spring constant
# by this amount once for each smaller (divided) cell involved with the bond. When it was 1 (i.e. not yet
# implemented; all bonds had same spring constant), small cells were compressed, and large cells were overstretched,
# so their actual sizes did not match their assigned "target" size. This was readily visible in the plot of
# tension on the large vs. small cells, where larger cells experienced higher tension than smaller ones.
# Relaxing spring constant on the smaller cells counteracts this. This is important for tissue integrity.
#
# Note: initially tried the value 2 ** (1/4), about 1.1892, based on the conjecture that strength
# of cell adhesion should scale with surface-area-to-volume ratio. The relevant surface area being the
# lateral surface of the squamous cell (excluding apical and basal surfaces); for smaller cells, this means
# the relevant SA-to-V ratio is sqrt(2) times that of the larger cells. So divide by that when both cells
# are small; and by the square root of that (because it's the geometric mean of 1 and that value) when only
# one of the cells is small. That was the theory, so started with that. Worked great at first, but then it
# turned out that the effect is highly sensitive to other parameters, so, as we adjust other forces on the
# cells, have to adjust this as well. Aim for tension equalized on both sets of cells. If larger cells experience
# LESS tension than smaller ones (smaller cells are relatively stretched and larger ones compressed; in extreme
# cases eliminating the size difference entirely), then the value is too high. If larger cells experience MORE
# tension than smaller ones (larger cells relatively stretched and smaller ones compressed; actual size
# difference between cells is larger than their target sizes / assigned cell radii would indicate), then
# the value is too low.
#
# And note: after changing constraints (eliminate neighbor-count, enhance bond-angle to compensate), this seems
# to be no long necessary. Setting value to 1 means make no adjustment, i.e., 1 = disabled. ToDo: Revisit! (was 1.1)
spring_constant_cell_size_factor: float = 1

# With k=0.067: dt = 0.1 and 0.05 blew up, and dt = 0.02 was fine.
# Using k=0.033 allowed me to reduce time granularity a bit more to dt = 0.025, but it did cause
# some buckling of the leading edge at the very end of certain experiments.
# So, probably best to stick with k=0.067 and dt = 0.02.
# Note, the angle-bonds feature was deprecated - see angle_bonds_enabled - long before I implemented
# variable cell sizes, and spring_constant_cell_size_factor; I have no idea how that would apply to angle bonds,
# and I have not tried it. So, angle-bonds feature might be broken now. Probably should remove from code.
harmonic_angle_spring_constant: float = 0.067  # (for the Angles)
harmonic_angle_tolerance: float = 0.008 * math.pi

# Vegetalward forces applied to LeadingEdge. Initial values from manual tuning.
# These are highly dependent on the particle radius, spring constants, etc.,
# so have to be tuned accordingly if those change.
# yolk_cortical_tension: force generated within the yolk, balancing out the EVL internal tension (at T=0)
#   so that the leading edge is stable (not advancing) until some extrinsic additional force is applied.
# external_force: additional force applied to drive epiboly. Will be overridden and set to 0 if
#   run_balanced_force_control == True (see below, under "Controlling the model")
# Both of these values are in units of force-per-unit-length of EVL edge
yolk_cortical_tension: float = 0.25    # just balances interior bonds at initialization
external_force: float = 0.45           # +additional to produce full epiboly

# Note, for backward compatibility, the dict key for this in the metadata file has to still
# use the old "force_is_weighted_by..." name instead of this new "...enabled" name!
force_regulation_enabled: bool = True

# Potential.max any greater than this, numerical problems ensue
max_potential_cutoff: float = 6

# ToDo: Get rid of this cruft:
# Bond-making. Method for when cell division disabled. (When cell division enabled, just gets nearest non-bonded
# neighbor, which is equivalent to BOUNDED with min = max = 1, or UNIFORM with min = 1, max = 2.)
class BondableNeighborDiscovery(Enum):
    NEAREST = 1     # request the nearest non-bonded neighbor, and just bond to that one (the "classic" algorithm)
    OPEN_ENDED = 2  # request a minimum of 1 bondable neighbor; select at random from however many come back.
    BOUNDED = 3     # request between min and max bondable neighbors; select at random from however many come back.
    UNIFORM = 4     # request exactly randrange(min, max) bondable neighbors; select at random from those.
    # (These each behave a bit differently because of how the underlying neighbor search algorithm works. These
    # are in order of increasingly more predictable/understandable. The first two suffer from dependence on the
    # implementation details of that underlying search, resulting in strange distributions of the size of the
    # search result set. UNIFORM should decouple the result from the underlying search; each search decides in
    # advance (using randrange) how many neighbor particles to select from; the result set will be exactly that size.)
bondable_neighbor_discovery: BondableNeighborDiscovery = BondableNeighborDiscovery.NEAREST
bondable_neighbors_min_candidates: int = 1  # (_min_ and _max_ ignored for NEAREST and OPEN_ENDED)
bondable_neighbors_max_candidates: int = 7

bond_remodeling_enabled: bool = True

# For other than the special edge-transformations, what fraction of the bond-making and -breaking should
# be coupled (break an existing bond and make a new one to a different particle at the same time)? The
# remaining fraction will be uncoupled. Value should be between 0 and 1 inclusive. Note, 1 = all remodeling
# is coupled, would mean the total number of bonds in the system can never change (except along the leading edge),
# and this is really bad. 0 = all remodeling is uncoupled, would mean just like before, no way to apply
# bond-angle constraint to a pair of bonds and decide whether to switch them.
# (Conclusion: best result was with 0.3, but this is still not good enough. Need to use space-filling algorithm
# instead. "Instead" because with space-filling, coupled-bond remodeling does no harm but is also not needed,
# so just abandon it and eliminate unneeded complexity in the model. So settling on 0.)
coupled_bond_remodeling_freq: float = 0

# For neighbor count criterion. Pre-energy-calculation limits.
# (If max exceeded, don't bother calculating energy, just reject the change.)
# (min, also for initialization: ensure this constraint from beginning)
min_neighbor_count: int = 3
max_edge_neighbor_count: int = 3

# Strength of each term in energy calculations on topological constraints (neighbor count, bond angle).
# Separate values for the edge transformations, so they can be tuned separately if desired.
# Note, for backward compatibility, the dict keys for these in the metadata file have to still
# use the old "k_" names instead of the new "lambda_" names!
lambda_neighbor_count: float = 0
lambda_edge_neighbor_count: float = 0
lambda_bond_angle: float = 3.75
# Separately specified value for edge angles originally defaulted to the same optimum of 3.75 used for internal
# angles, but let's default to 4 instead because it's more convenient for the plots in our figures.
lambda_edge_bond_angle: float = 4
special_constraint_all_edge_bonds: bool = False

# For neighbor angle energy calculations
target_neighbor_angle: float = math.pi / 3
target_edge_angle: float = math.pi      # Currently not used; experimenting with dynamic

# For controlling which internal cells can become edge (must be within this distance)
leading_edge_recruitment_limit: float = 1.5     # in number of cell radii (not particle radii)

# -------------------- Controlling the model --------------------

# If true, use 0 external force. (For a turnkey entry point, other things will
#   be changed along with it, like how simulation end is decided, and the interval for plotting.)
run_balanced_force_control: bool = False
balanced_force_equilibration_kludge: bool = False

# Unlike the flags below this, this one prevents migration into the margin in the simulation ALWAYS,
# not just during the recoil experiment.
never_allow_enter_margin: bool = False

test_recoil_without_bond_remodeling: bool = False
test_recoil_with_bond_remodeling: bool = False
# While recoil experiment with bond remodeling is in progress, modify these flags to change the remodeling rules:
allow_exit_margin: bool = True
allow_enter_margin: bool = True
always_accept_enter_margin: bool = False
allow_internal_remodeling: bool = True
# How long for each test (in units of Universe.time):
recoil_duration_without_remodeling: float = 75
recoil_duration_with_remodeling: float = 75

stopping_condition_phi: float = math.pi * 0.95              # MEAN margin particle phi is compared to this
unbalanced_stopping_condition_phi: float = math.pi * 0.98   # MAX margin particle phi is compared to this

# -------------------- Tissue Forge --------------------

# Whether to use TF windowless mode, in which the simulation is driven only by
# tf.step() and never tf.show(), and no graphics are displayed.
# But name this flag as a positive rather than a negative, to avoid confusing double negatives ("not windowless").
windowed_mode: bool = False

# -------------------- Visualization of the model --------------------

# Whether to show the equilibration steps.
# In windowless mode, whether to include them in any exported screenshots;
# in windowed mode, whether to show them in the simulator window (and any exported screenshots), or hide in tf.step();
# useful to set True during development so I can see what I'm doing (or for demos); otherwise leave as False.
show_equilibration: bool = False

class PaintPattern(Enum):
    CELL_TYPE = 1           # LeadingEdge vs. Evl internal
    ORIGINAL_TIER = 2       # By position at initialization, ignoring type entirely
    VERTICAL_STRIPE = 3     # By position at initialization, ignoring type entirely (just proof of concept)
    PATCH = 4               # By position at initialization, ignoring type entirely
    SPECIES = 5             # Read a concentration from the particle, and use that concentration to determine the color.

paint_pattern: PaintPattern = PaintPattern.CELL_TYPE

# Which tier to paint, in the ORIGINAL_TIER PaintPattern.
paint_tier: int = 0

# Patch position (distance from leading edge) and size in degrees of arc, in the PATCH PaintPattern.
# Patch will be placed on front side of the embryo.
patch_margin_gap: float = 0
patch_width: float = 30
patch_height: float = 10

# Whether to color daughter cells differently from parent cells, in the CELL_TYPE PaintPattern
color_code_daughter_cells: bool = False

# Number of timesteps between screenshots. Set to 0 to disable screenshot export.
# If enabled, interval value can be adjusted dynamically at run time using the setter in module video_export.
# (Set the value in time units; calculated value in timesteps will be used during execution.)
screenshots_simtime_per_export: float = 0.5
screenshots_timesteps_per_export: int = (0 if screenshots_simtime_per_export == 0 else
                                         max(1, round(screenshots_simtime_per_export / dt)))

# If False, exported screenshots are deleted after being compiled into a movie at the end of the run.
# Setting True was helpful for browsing still images for publication.
# Warning: Thousands of images, gigabytes, are produced over the course of a typical run. Set True at your own risk.
retain_screenshots_after_movie: bool = False

# Interval between time points in the simple plots (the ones that are just a single value over time).
# (Set the value in time units; calculated value in timesteps will be used during execution.)
simple_plot_interval_simtime: float = 4.0
simple_plot_interval_timesteps: int = round(simple_plot_interval_simtime / dt)

# Interval between time points in the aggregate graphs. Depending on the experiment, a different value may work better.
# (Set the value in time units; calculated value in timesteps will be used during execution.)
plotting_interval_simtime: float = 50
plotting_interval_timesteps: int = int(round(plotting_interval_simtime / dt, -2))
# Should certain metrics be plotted as time-averages, instead of as single timesteps?
plot_time_averages: bool = True
# How many timesteps? (Calculated limit will keep the value sane.)
config_time_avg_accumulation_steps: int = 200
time_avg_accumulation_steps: int = min(plotting_interval_timesteps - 1, config_time_avg_accumulation_steps)
# And if so, should that also be applied to the simulation start, or just plot T0 as a single timestep?
plot_t0_as_single_timestep: bool = True  # (ignored unless plot_time_averages is True)

# -------------------- Data export --------------------

sim_state_export_enabled: bool = True

# Sim state exports are quite large. If Using for post-processing of entire simulation, set to True
# to retain all exports. If using only to be able to recover from premature exit, set to False to retain
# only the most recent export and delete the rest. Export timing will use timesteps or minutes, respectively.
sim_state_export_keep: bool = False

# Number of timesteps/minutes between sim state exports.
# (If using timesteps, set the value in simulation time units (simtime_per_export); calculated
# timesteps (timesteps_per_export) will be used during execution.)
sim_state_simtime_per_export: float = 10
sim_state_timesteps_per_export: int = round(sim_state_simtime_per_export / dt)
sim_state_minutes_per_export: int = 10

def get_state() -> dict:
    """generate state to be saved to disk
    
    In this case, it's for an atypical purpose. Normally we export state that needs to survive
    through termination and restart from import. Or, state that's needed for post-processing.
    These are all constants so don't need that. Rather, this is to capture the config state
    in a file stored with all the other simulation output, as a record of what the config WAS
    when the simulation was run. Because it has become way too complex to record simply by
    adding notes to the path name!
    """
    return {"config_values": {
                "comment": comment,
                "environment": {
                        "batch_execute": batch_execute,
                        "override_export_root_directory": override_export_root_directory,
                        "export_root_directory_name": export_root_directory_name,
                        },
                "model": {
                        "dt": dt,
                        "initialization_algo_graph_based": initialization_algo_graph_based,
                        "cell_division_enabled": cell_division_enabled,
                        "total_epiboly_divisions": total_epiboly_divisions,
                        "cell_division_cessation_percentage": cell_division_cessation_percentage,
                        "cell_division_largest_first": cell_division_largest_first,
                        "cell_division_biased_by_tension": cell_division_biased_by_tension,
                        "tension_squared": tension_squared,
                        "angle_bonds_enabled": angle_bonds_enabled,
                        "space_filling_enabled": space_filling_enabled,
                        "k_particle_diffusion": k_particle_diffusion,
                        "epiboly_initial_percentage": epiboly_initial_percentage,
                        "epiboly_initial_num_evl_cells": epiboly_initial_num_evl_cells,
                        "evl_particle_radius": evl_particle_radius,
                        "min_neighbor_initial_distance_factor": min_neighbor_initial_distance_factor,
                        "harmonic_repulsion_spring_constant": harmonic_repulsion_spring_constant,
                        "harmonic_spring_constant": harmonic_spring_constant,
                        "harmonic_edge_spring_constant": harmonic_edge_spring_constant,
                        "harmonic_yolk_evl_spring_constant": harmonic_yolk_evl_spring_constant,
                        "spring_constant_cell_size_factor": spring_constant_cell_size_factor,
                        "harmonic_angle_spring_constant": harmonic_angle_spring_constant,
                        "harmonic_angle_tolerance": harmonic_angle_tolerance,
                        "yolk_cortical_tension": yolk_cortical_tension,
                        "external_force": external_force,
                        "force_is_weighted_by_distance_from_pole": force_regulation_enabled,
                        "max_potential_cutoff": max_potential_cutoff,
                        "bondable_neighbor_discovery": bondable_neighbor_discovery.name,
                        "bondable_neighbors_min_candidates": bondable_neighbors_min_candidates,
                        "bondable_neighbors_max_candidates": bondable_neighbors_max_candidates,
                        "bond_remodeling_enabled": bond_remodeling_enabled,
                        "coupled_bond_remodeling_freq": coupled_bond_remodeling_freq,
                        "min_neighbor_count": min_neighbor_count,
                        "max_edge_neighbor_count": max_edge_neighbor_count,
                        "k_neighbor_count": lambda_neighbor_count,
                        "k_edge_neighbor_count": lambda_edge_neighbor_count,
                        "k_bond_angle": lambda_bond_angle,
                        "k_edge_bond_angle": lambda_edge_bond_angle,
                        "special_constraint_all_edge_bonds": special_constraint_all_edge_bonds,
                        "target_neighbor_angle": target_neighbor_angle,
                        "target_edge_angle": target_edge_angle,
                        "leading_edge_recruitment_limit": leading_edge_recruitment_limit,
                        },
                "model control": {
                        "run_balanced_force_control": run_balanced_force_control,
                        "balanced_force_equilibration_kludge": balanced_force_equilibration_kludge,
                        "never_allow_enter_margin": never_allow_enter_margin,
                        "test_recoil_without_bond_remodeling": test_recoil_without_bond_remodeling,
                        "test_recoil_with_bond_remodeling": test_recoil_with_bond_remodeling,
                        "allow_exit_margin": allow_exit_margin,
                        "allow_enter_margin": allow_enter_margin,
                        "always_accept_enter_margin": always_accept_enter_margin,
                        "allow_internal_remodeling": allow_internal_remodeling,
                        "recoil_duration_without_remodeling": recoil_duration_without_remodeling,
                        "recoil_duration_with_remodeling": recoil_duration_with_remodeling,
                        "stopping_condition_phi": stopping_condition_phi,
                        "unbalanced_stopping_condition_phi": unbalanced_stopping_condition_phi,
                        },
                "visualization": {
                        "show_equilibration": show_equilibration,
                        "paint_pattern": paint_pattern.name,
                        "paint_tier": paint_tier,
                        "patch_margin_gap": patch_margin_gap,
                        "patch_width": patch_width,
                        "patch_height": patch_height,
                        "color_code_daughter_cells": color_code_daughter_cells,
                        "screenshots_simtime_per_export": screenshots_simtime_per_export,
                        "retain_screenshots_after_movie": retain_screenshots_after_movie,
                        "simple_plot_interval_simtime": simple_plot_interval_simtime,
                        "plotting_interval_simtime": plotting_interval_simtime,
                        "plot_time_averages": plot_time_averages,
                        "config_time_avg_accumulation_steps": config_time_avg_accumulation_steps,
                        "plot_t0_as_single_timestep": plot_t0_as_single_timestep,
                        },
                "data export": {
                        "sim_state_simtime_per_export": sim_state_simtime_per_export,
                        "sim_state_minutes_per_export": sim_state_minutes_per_export,
                        "sim_state_export_keep": sim_state_export_keep,
                        },
                },
            "derived_values": {
                "sim_state_timesteps_per_export": sim_state_timesteps_per_export,
                "screenshots_timesteps_per_export": screenshots_timesteps_per_export,
                "simple_plot_interval_timesteps": simple_plot_interval_timesteps,
                "plotting_interval_timesteps": plotting_interval_timesteps,
                "time_avg_accumulation_steps": time_avg_accumulation_steps,
                }
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved.
    
    Since these are all constants, they normally don't need to be reconstituted.
    But this is helpful when I modify config values while simulations are running.
    Then terminate/restart/import would override pending changes to this file, and instead restore
    the values that were present when the simulation was initially launched, preventing errors.
    
    Though, note to self: if I actually add or remove config variables, that still requires more care.
    """
    global comment
    
    # environment
    global batch_execute, override_export_root_directory, export_root_directory_name
    
    # model
    global dt, initialization_algo_graph_based, cell_division_enabled, total_epiboly_divisions
    global cell_division_cessation_percentage, cell_division_largest_first
    global cell_division_biased_by_tension, tension_squared
    global angle_bonds_enabled, space_filling_enabled, k_particle_diffusion
    global epiboly_initial_percentage, epiboly_initial_num_evl_cells, evl_particle_radius
    global min_neighbor_initial_distance_factor
    global harmonic_repulsion_spring_constant, harmonic_spring_constant, harmonic_edge_spring_constant
    global spring_constant_cell_size_factor
    global harmonic_yolk_evl_spring_constant, harmonic_angle_spring_constant, harmonic_angle_tolerance
    global yolk_cortical_tension, external_force, force_regulation_enabled, max_potential_cutoff
    global bondable_neighbor_discovery, bondable_neighbors_min_candidates, bondable_neighbors_max_candidates
    global bond_remodeling_enabled
    global coupled_bond_remodeling_freq, min_neighbor_count, max_edge_neighbor_count
    global lambda_neighbor_count, lambda_edge_neighbor_count
    global lambda_bond_angle, lambda_edge_bond_angle, special_constraint_all_edge_bonds
    global target_neighbor_angle, target_edge_angle, leading_edge_recruitment_limit
    
    # model control
    global run_balanced_force_control, balanced_force_equilibration_kludge
    global never_allow_enter_margin
    global test_recoil_without_bond_remodeling, test_recoil_with_bond_remodeling
    global allow_exit_margin, allow_enter_margin, always_accept_enter_margin, allow_internal_remodeling
    global recoil_duration_without_remodeling, recoil_duration_with_remodeling
    global stopping_condition_phi, unbalanced_stopping_condition_phi
    
    # visualization
    global show_equilibration, paint_pattern, paint_tier
    global patch_margin_gap, patch_width, patch_height, color_code_daughter_cells
    global screenshots_simtime_per_export, retain_screenshots_after_movie
    global simple_plot_interval_simtime, plotting_interval_simtime
    global plot_time_averages, config_time_avg_accumulation_steps, plot_t0_as_single_timestep
    
    # data export
    global sim_state_simtime_per_export, sim_state_minutes_per_export, sim_state_export_keep
    
    # derived values
    global sim_state_timesteps_per_export, screenshots_timesteps_per_export
    global simple_plot_interval_timesteps, plotting_interval_timesteps, time_avg_accumulation_steps
    
    comment = d["config_values"]["comment"]
    
    environment: dict = d["config_values"]["environment"]
    batch_execute = environment["batch_execute"]
    override_export_root_directory = environment["override_export_root_directory"]
    export_root_directory_name = environment["export_root_directory_name"]
    
    model: dict = d["config_values"]["model"]
    dt = model["dt"]
    initialization_algo_graph_based = model["initialization_algo_graph_based"]
    cell_division_enabled = model["cell_division_enabled"]
    total_epiboly_divisions = model["total_epiboly_divisions"]
    cell_division_cessation_percentage = model["cell_division_cessation_percentage"]
    cell_division_largest_first = model["cell_division_largest_first"]
    cell_division_biased_by_tension = model["cell_division_biased_by_tension"]
    tension_squared = model["tension_squared"]
    angle_bonds_enabled = model["angle_bonds_enabled"]
    space_filling_enabled = model["space_filling_enabled"]
    k_particle_diffusion = model["k_particle_diffusion"]
    epiboly_initial_percentage = model["epiboly_initial_percentage"]
    epiboly_initial_num_evl_cells = model["epiboly_initial_num_evl_cells"]
    evl_particle_radius = model["evl_particle_radius"]
    min_neighbor_initial_distance_factor = model["min_neighbor_initial_distance_factor"]
    harmonic_repulsion_spring_constant = model["harmonic_repulsion_spring_constant"]
    harmonic_spring_constant = model["harmonic_spring_constant"]
    harmonic_edge_spring_constant = model["harmonic_edge_spring_constant"]
    harmonic_yolk_evl_spring_constant = model["harmonic_yolk_evl_spring_constant"]
    spring_constant_cell_size_factor = model["spring_constant_cell_size_factor"]
    harmonic_angle_spring_constant = model["harmonic_angle_spring_constant"]
    harmonic_angle_tolerance = model["harmonic_angle_tolerance"]
    yolk_cortical_tension = model["yolk_cortical_tension"]
    external_force = model["external_force"]
    force_regulation_enabled = model["force_is_weighted_by_distance_from_pole"]
    max_potential_cutoff = model["max_potential_cutoff"]
    bondable_neighbor_discovery = BondableNeighborDiscovery[model["bondable_neighbor_discovery"]]
    bondable_neighbors_min_candidates = model["bondable_neighbors_min_candidates"]
    bondable_neighbors_max_candidates = model["bondable_neighbors_max_candidates"]
    bond_remodeling_enabled = model["bond_remodeling_enabled"]
    coupled_bond_remodeling_freq = model["coupled_bond_remodeling_freq"]
    min_neighbor_count = model["min_neighbor_count"]
    max_edge_neighbor_count = model["max_edge_neighbor_count"]
    lambda_neighbor_count = model["k_neighbor_count"]
    lambda_edge_neighbor_count = model["k_edge_neighbor_count"]
    lambda_bond_angle = model["k_bond_angle"]
    lambda_edge_bond_angle = model["k_edge_bond_angle"]
    special_constraint_all_edge_bonds = model["special_constraint_all_edge_bonds"]
    target_neighbor_angle = model["target_neighbor_angle"]
    target_edge_angle = model["target_edge_angle"]
    leading_edge_recruitment_limit = model["leading_edge_recruitment_limit"]
    
    control: dict = d["config_values"]["model control"]
    run_balanced_force_control = control["run_balanced_force_control"]
    balanced_force_equilibration_kludge = control["balanced_force_equilibration_kludge"]
    never_allow_enter_margin = control["never_allow_enter_margin"]
    test_recoil_without_bond_remodeling = control["test_recoil_without_bond_remodeling"]
    test_recoil_with_bond_remodeling = control["test_recoil_with_bond_remodeling"]
    allow_exit_margin = control["allow_exit_margin"]
    allow_enter_margin = control["allow_enter_margin"]
    always_accept_enter_margin = control["always_accept_enter_margin"]
    allow_internal_remodeling = control["allow_internal_remodeling"]
    recoil_duration_without_remodeling = control["recoil_duration_without_remodeling"]
    recoil_duration_with_remodeling = control["recoil_duration_with_remodeling"]
    stopping_condition_phi = control["stopping_condition_phi"]
    unbalanced_stopping_condition_phi = control["unbalanced_stopping_condition_phi"]

    visualization: dict = d["config_values"]["visualization"]
    show_equilibration = visualization["show_equilibration"]
    paint_pattern = PaintPattern[visualization["paint_pattern"]]
    paint_tier = visualization["paint_tier"]
    patch_margin_gap = visualization["patch_margin_gap"]
    patch_width = visualization["patch_width"]
    patch_height = visualization["patch_height"]
    color_code_daughter_cells = visualization["color_code_daughter_cells"]
    screenshots_simtime_per_export = visualization["screenshots_simtime_per_export"]
    retain_screenshots_after_movie = visualization["retain_screenshots_after_movie"]
    simple_plot_interval_simtime = visualization["simple_plot_interval_simtime"]
    plotting_interval_simtime = visualization["plotting_interval_simtime"]
    plot_time_averages = visualization["plot_time_averages"]
    config_time_avg_accumulation_steps = visualization["config_time_avg_accumulation_steps"]
    plot_t0_as_single_timestep = visualization["plot_t0_as_single_timestep"]

    data_export: dict = d["config_values"]["data export"]
    sim_state_simtime_per_export = data_export["sim_state_simtime_per_export"]
    sim_state_minutes_per_export = data_export["sim_state_minutes_per_export"]
    sim_state_export_keep = data_export["sim_state_export_keep"]

    derived_values: dict = d["derived_values"]
    sim_state_timesteps_per_export = derived_values["sim_state_timesteps_per_export"]
    screenshots_timesteps_per_export = derived_values["screenshots_timesteps_per_export"]
    simple_plot_interval_timesteps = derived_values["simple_plot_interval_timesteps"]
    plotting_interval_timesteps = derived_values["plotting_interval_timesteps"]
    time_avg_accumulation_steps = derived_values["time_avg_accumulation_steps"]
