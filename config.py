"""Config

Flags and magic numbers, especially ones used in more than one place.
"""
from enum import Enum
import math

# Temporary (and not bothering to export):
# ToDo: Finally works, but changes coming in next release. Once that's stabilized, make this the new normal.
use_alt_cell_splitting_method: bool = True

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
dt: float = 0.16

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
# particles they are bound to, and hence toward open space.
space_filling_enabled: bool = True
k_particle_diffusion: float = 1.7

# Starting point of the simulation. Note that 43% is the true value for the misnomer "30% epiboly")
epiboly_initial_percentage: int = 43
epiboly_initial_num_evl_cells: int = 660  # (Campinho et al., 2013)

# This is the size of the particle, not of the cell. The size of the cell will be derived from
# epiboly_initial_percentage and epiboly_initial_num_evl_cells.
evl_particle_radius: float = 0.045

# Search for neighbors within this distance (multiple of cell radius) to set up initial bond network.
min_neighbor_initial_distance_factor: float = 1.5

# Some items for Potential- and Bond-making:
harmonic_repulsion_spring_constant: float = 0.3
harmonic_spring_constant: float = 0.4
harmonic_edge_spring_constant: float = 0.4
harmonic_yolk_evl_spring_constant: float = 2.7
# With k=0.067: dt = 0.1 and 0.05 blew up, and dt = 0.02 was fine.
# Using k=0.033 allowed me to reduce time granularity a bit more to dt = 0.025, but it did cause
# some buckling of the leading edge at the very end of certain experiments.
# So, probably best to stick with k=0.067 and dt = 0.02.
harmonic_angle_spring_constant: float = 0.067  # (for the Angles)
harmonic_angle_tolerance: float = 0.008 * math.pi

# ToDo: Maybe get rid of this entirely later? Or may try other algorithms?
class ForceAlgorithm(Enum):
    CONSTANT = 1          # Total force is constant, stays at its initial value
    LINEAR = 2            # Total force vs circumference is linear; defined by force_target_fraction

# Vegetalward forces applied to LeadingEdge. Initial values from manual tuning.
# These are highly dependent on the particle radius, spring constants, etc.,
# so have to be tuned accordingly if those change.
# yolk_cortical_tension: force generated within the yolk, balancing out the EVL internal tension (at T=0)
#   so that the leading edge is stable (not advancing) until some extrinsic additional force is applied.
# external_force: additional force applied to drive epiboly.
# force_algorithm: Defines relationship between total force and circumference
# force_target_fraction: For LINEAR, fraction of initial force to approach as circumf approaches 0
# run_balanced_force_control: if true, use 0 external force. (For a turnkey entry point, other things will
#   be changed along with it, like how simulation end is decided, and the interval for plotting.)
yolk_cortical_tension: float = 8  # just balances interior bonds at initialization
external_force: float = 7  # +additional to produce full epiboly

force_algorithm: ForceAlgorithm = ForceAlgorithm.LINEAR
force_target_fraction: float = 0

# Potential.max any greater than this, numerical problems ensue
max_potential_cutoff: float = 6

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

# For neighbor angle energy calculations
target_neighbor_angle: float = math.pi / 3
target_edge_angle: float = math.pi

# For controlling which internal cells can become edge (must be within this distance)
leading_edge_recruitment_limit: float = 2.0     # in number of cell radii (not particle radii)

# -------------------- Controlling the model --------------------

run_balanced_force_control: bool = False

test_recoil_without_bond_remodeling: bool = False
test_recoil_with_bond_remodeling: bool = False  # Ignored when cell division enabled, since not meaningful in that case
recoil_duration_without_remodeling: float = 75
recoil_duration_with_remodeling: float = 75

stopping_condition_phi: float = math.pi * 0.95

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

# Number of timesteps between screenshots. Set to 0 to disable screenshot export.
# If enabled, interval value can be adjusted dynamically at run time using the setter in module video_export.
# (Set the value in time units; calculated value in timesteps will be used during execution.)
screenshots_simtime_per_export: float = 1.6
screenshots_timesteps_per_export: int = (0 if screenshots_simtime_per_export == 0 else
                                         max(1, round(screenshots_simtime_per_export / dt)))

# Interval between time points in the aggregate graphs. Depending on the experiment, a different value may work better.
# (Set the value in time units; calculated value in timesteps will be used during execution.)
plotting_interval_simtime: float = 160
plotting_interval_timesteps: int = round(plotting_interval_simtime / dt)
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
    when the simulation was run. Because it's starting to be too complex to record simply by
    adding notes to the path name!
    """
    return {"config_values": {
                "comment": comment,
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
                        "harmonic_angle_spring_constant": harmonic_angle_spring_constant,
                        "harmonic_angle_tolerance": harmonic_angle_tolerance,
                        "yolk_cortical_tension": yolk_cortical_tension,
                        "external_force": external_force,
                        "force_algorithm": force_algorithm.name,
                        "force_target_fraction": force_target_fraction,
                        "max_potential_cutoff": max_potential_cutoff,
                        "bondable_neighbor_discovery": bondable_neighbor_discovery.name,
                        "bondable_neighbors_min_candidates": bondable_neighbors_min_candidates,
                        "bondable_neighbors_max_candidates": bondable_neighbors_max_candidates,
                        "coupled_bond_remodeling_freq": coupled_bond_remodeling_freq,
                        "min_neighbor_count": min_neighbor_count,
                        "max_edge_neighbor_count": max_edge_neighbor_count,
                        "target_neighbor_angle": target_neighbor_angle,
                        "target_edge_angle": target_edge_angle,
                        "leading_edge_recruitment_limit": leading_edge_recruitment_limit,
                        },
                "model control": {
                        "run_balanced_force_control": run_balanced_force_control,
                        "test_recoil_without_bond_remodeling": test_recoil_without_bond_remodeling,
                        "test_recoil_with_bond_remodeling": test_recoil_with_bond_remodeling,
                        "recoil_duration_without_remodeling": recoil_duration_without_remodeling,
                        "recoil_duration_with_remodeling": recoil_duration_with_remodeling,
                        "stopping_condition_phi": stopping_condition_phi,
                        },
                "visualization": {
                        "show_equilibration": show_equilibration,
                        "screenshots_simtime_per_export": screenshots_simtime_per_export,
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
    
    # model
    global dt, initialization_algo_graph_based, cell_division_enabled, total_epiboly_divisions
    global cell_division_cessation_percentage, cell_division_largest_first
    global cell_division_biased_by_tension, tension_squared
    global angle_bonds_enabled, space_filling_enabled, k_particle_diffusion
    global epiboly_initial_percentage, epiboly_initial_num_evl_cells, evl_particle_radius
    global min_neighbor_initial_distance_factor
    global harmonic_repulsion_spring_constant, harmonic_spring_constant, harmonic_edge_spring_constant
    global harmonic_yolk_evl_spring_constant, harmonic_angle_spring_constant, harmonic_angle_tolerance
    global yolk_cortical_tension, external_force, force_algorithm, force_target_fraction, max_potential_cutoff
    global bondable_neighbor_discovery, bondable_neighbors_min_candidates, bondable_neighbors_max_candidates
    global coupled_bond_remodeling_freq, min_neighbor_count, max_edge_neighbor_count
    global target_neighbor_angle, target_edge_angle, leading_edge_recruitment_limit
    
    # model control
    global run_balanced_force_control, test_recoil_without_bond_remodeling, test_recoil_with_bond_remodeling
    global recoil_duration_without_remodeling, recoil_duration_with_remodeling, stopping_condition_phi
    
    # visualization
    global show_equilibration, screenshots_simtime_per_export, plotting_interval_simtime
    global plot_time_averages, config_time_avg_accumulation_steps, plot_t0_as_single_timestep
    
    # data export
    global sim_state_simtime_per_export, sim_state_minutes_per_export, sim_state_export_keep
    
    # derived values
    global sim_state_timesteps_per_export, screenshots_timesteps_per_export
    global plotting_interval_timesteps, time_avg_accumulation_steps
    
    comment = d["config_values"]["comment"]
    
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
    harmonic_angle_spring_constant = model["harmonic_angle_spring_constant"]
    harmonic_angle_tolerance = model["harmonic_angle_tolerance"]
    yolk_cortical_tension = model["yolk_cortical_tension"]
    external_force = model["external_force"]
    force_algorithm = ForceAlgorithm[model["force_algorithm"]]
    force_target_fraction = model["force_target_fraction"]
    max_potential_cutoff = model["max_potential_cutoff"]
    bondable_neighbor_discovery = BondableNeighborDiscovery[model["bondable_neighbor_discovery"]]
    bondable_neighbors_min_candidates = model["bondable_neighbors_min_candidates"]
    bondable_neighbors_max_candidates = model["bondable_neighbors_max_candidates"]
    coupled_bond_remodeling_freq = model["coupled_bond_remodeling_freq"]
    min_neighbor_count = model["min_neighbor_count"]
    max_edge_neighbor_count = model["max_edge_neighbor_count"]
    target_neighbor_angle = model["target_neighbor_angle"]
    target_edge_angle = model["target_edge_angle"]
    leading_edge_recruitment_limit = model["leading_edge_recruitment_limit"]
    
    control: dict = d["config_values"]["model control"]
    run_balanced_force_control = control["run_balanced_force_control"]
    test_recoil_without_bond_remodeling = control["test_recoil_without_bond_remodeling"]
    test_recoil_with_bond_remodeling = control["test_recoil_with_bond_remodeling"]
    recoil_duration_without_remodeling = control["recoil_duration_without_remodeling"]
    recoil_duration_with_remodeling = control["recoil_duration_with_remodeling"]
    stopping_condition_phi = control["stopping_condition_phi"]

    visualization: dict = d["config_values"]["visualization"]
    show_equilibration = visualization["show_equilibration"]
    screenshots_simtime_per_export = visualization["screenshots_simtime_per_export"]
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
    plotting_interval_timesteps = derived_values["plotting_interval_timesteps"]
    time_avg_accumulation_steps = derived_values["time_avg_accumulation_steps"]
