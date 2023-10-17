"""Config

Flags and magic numbers, especially ones used in more than one place.
"""
from enum import Enum
import math

# For normal initialization, leave blank. To start from a previously exported state, provide the
# directory name of the export. This is the parent directory with the datetime in the name. Script
# will search in the "Sim_state" subdirectory to find the most recent export, and start with that.
initialization_directory_name: str = ""

# Just like it says on the tin. Since config is saved as part of state export as metadata for this run, this
# provides for a free-form comment at the top of that file, for a permanent commentary on the generated output.
comment: str = ""

# Whether to use TF windowless mode, in which the simulation is driven only by
# tf.step() and never tf.show(), and no graphics are displayed.
# But name this flag as a positive rather than a negative, to avoid confusing double negatives ("not windowless").
windowed_mode: bool = False

# Tissue Forge time increment. Tissue Forge default value is 0.01. If using a different value, consider
# adjusting these other values to compensate, which are defined in terms of timesteps (they count dt intervals):
# plot_interval, sim_state_export_timestep_interval, screenshot_export_interval, time_avg_accumulation_steps
dt: float = 0.02

# Whether to show the equilibration steps.
# In windowless mode, whether to include them in any exported screenshots;
# in windowed mode, whether to show them in the simulator window (and any exported screenshots), or hide in tf.step();
# useful to set True during development so I can see what I'm doing; otherwise leave as False.
show_equilibration: bool = False

sim_state_export_enabled: bool = True

# Number of timesteps/minutes between sim state exports.
sim_state_export_timestep_interval: int = 500
sim_state_export_minutes_interval: int = 10

# Sim state exports are quite large. If Using for post-processing of entire simulation, set to True
# to retain all exports. If using only to be able to recover from premature exit, set to False to retain
# only the most recent export and delete the rest. Export timing will use timesteps or minutes, respectively.
sim_state_export_keep: bool = False

# Number of timesteps between screenshots. Set to 0 to disable screenshot export.
# If enabled, interval value can be adjusted dynamically at run time using the setter in module video_export.
screenshot_export_interval: int = 5

# Cell division: whether or not, and how, and how much:
cell_division_enabled: bool = True
# Cell division rate parameters. See justification in docstring of cell_division.cell_division().
# Calibrate either to timesteps, or to EVL area increase:
calibrate_division_rate_to_timesteps: bool = False
total_epiboly_divisions: int = 7500  # currently throttled to 3064; TBD: how much to actually use
# Spatial distribution of cell division events:
cell_division_biased_by_tension: bool = False
tension_squared: bool = False  # (ignored unless cell_division_biased_by_tension is True)

# Interval between time points in the aggregate graphs. Depending on the experiment, a different value may work better.
plot_interval: int = 5000
# Should certain metrics be plotted as time-averages, instead of as single timesteps?
plot_time_averages: bool = True
# How many timesteps?
time_avg_accumulation_steps: int = 200
# And if so, should that also be applied to the simulation start, or just plot T0 as a single timestep?
plot_t0_as_single_timestep: bool = True  # (ignored unless plot_time_averages is True)

# Useful to turn this off while tuning setup and equilibration. When external force is artificially low,
# and if Angle bonds too high, they cause instability and big waviness in the leading edge. (Seems fixed now.)
angle_bonds_enabled: bool = True

# Yet to be determined, whether to use this space-filling algorithm. It needs to be either abandoned or improved.
# Currently alternating sims between having it on and off while I work on other things, and observing its behavior.
space_filling_enabled: bool = False

# real value for the misnomer "30% epiboly")
epiboly_initial_percentage: int = 43

# How many leading edge and interior cells to make (for entire sphere, prior to filtering out the ones below the edge)
num_leading_edge_points: int = 110
num_spherical_positions: int = 5000

# Search for neighbors within this distance (multiple of particle radius) to set up initial bond network.
min_neighbor_initial_distance_factor: float = 1.5

# Some items for Potential- and Bond-making:
harmonic_repulsion_spring_constant: float = 5.0
harmonic_spring_constant: float = 12.0
harmonic_edge_spring_constant: float = 12.0  # (for the Bonds)
# With k=1.0: dt = 0.1 and 0.05 blew up, and dt = 0.02 was fine.
# Using k=0.5 allowed me to reduce time granularity a bit more to dt = 0.025, but it did cause
# some buckling of the leading edge at the very end of certain experiments.
# So, probably best to stick with k=1.0 and dt = 0.02.
harmonic_angle_spring_constant: float = 1.0  # (for the Angles)
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
yolk_cortical_tension: int = 120    # just balances interior bonds at initialization
external_force: int = 100   # +additional to produce full epiboly (original "baseline" was 255)
force_algorithm: ForceAlgorithm = ForceAlgorithm.LINEAR
force_target_fraction: float = 0
run_balanced_force_control: bool = False

# Potential.max any greater than this, numerical problems ensue
max_potential_cutoff: float = 6

stopping_condition_phi: float = math.pi * 0.95

# For neighbor count criterion. Pre-energy-calculation limits.
# (If exceeded, don't bother calculating energy, just reject the change.)
# (min, also for initialization: ensure this constraint from beginning)
min_neighbor_count: int = 3
max_edge_neighbor_count: int = 3

# For neighbor angle energy calculations. Not only to avoid magic numbers,
# but also because these would otherwise be calculated millions of times,
# which is wasteful, even for an inexpensive operation.
target_neighbor_angle: float = math.pi / 3
target_edge_angle: float = math.pi
leading_edge_recruitment_limit: float = 2.0     # in number of radii

# For the same reason, just some common numbers useful in a variety of contexts
two_pi: float = math.pi * 2
pi_over_2: float = math.pi / 2

def get_state() -> dict:
    """generate state to be saved to disk
    
    In this case, it's for an atypical purpose. Normally we export state that needs to survive
    through termination and restart from import. Or, state that's needed for post-processing.
    These are all constants so don't need that. Rather, this is to capture the config state
    in a file stored with all the other simulation output, as a record of what the config WAS
    when the simulation was run. Because it's starting to be too complex to record simply by
    adding notes to the path name!
    """
    return {"comment": comment,
            "dt": dt,
            "show_equilibration": show_equilibration,
            "sim_state_export_timestep_interval": sim_state_export_timestep_interval,
            "sim_state_export_minutes_interval": sim_state_export_minutes_interval,
            "sim_state_export_keep": sim_state_export_keep,
            "screenshot_export_interval": screenshot_export_interval,
            "cell_division_enabled": cell_division_enabled,
            "calibrate_division_rate_to_timesteps": calibrate_division_rate_to_timesteps,
            "total_epiboly_divisions": total_epiboly_divisions,
            "cell_division_biased_by_tension": cell_division_biased_by_tension,
            "tension_squared": tension_squared,
            "plot_interval": plot_interval,
            "plot_time_averages": plot_time_averages,
            "time_avg_accumulation_steps": time_avg_accumulation_steps,
            "plot_t0_as_single_timestep": plot_t0_as_single_timestep,
            "angle_bonds_enabled": angle_bonds_enabled,
            "space_filling_enabled": space_filling_enabled,
            "epiboly_initial_percentage": epiboly_initial_percentage,
            "num_leading_edge_points": num_leading_edge_points,
            "num_spherical_positions": num_spherical_positions,
            "min_neighbor_initial_distance_factor": min_neighbor_initial_distance_factor,
            "harmonic_repulsion_spring_constant": harmonic_repulsion_spring_constant,
            "harmonic_spring_constant": harmonic_spring_constant,
            "harmonic_edge_spring_constant": harmonic_edge_spring_constant,
            "harmonic_angle_spring_constant": harmonic_angle_spring_constant,
            "harmonic_angle_tolerance": harmonic_angle_tolerance,
            "yolk_cortical_tension": yolk_cortical_tension,
            "external_force": external_force,
            "force_algorithm": force_algorithm.name,
            "force_target_fraction": force_target_fraction,
            "run_balanced_force_control": run_balanced_force_control,
            "max_potential_cutoff": max_potential_cutoff,
            "stopping_condition_phi": stopping_condition_phi,
            "min_neighbor_count": min_neighbor_count,
            "max_edge_neighbor_count": max_edge_neighbor_count,
            "target_neighbor_angle": target_neighbor_angle,
            "target_edge_angle": target_edge_angle,
            "leading_edge_recruitment_limit": leading_edge_recruitment_limit}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved.
    
    Since these are all constants, we don't actually need to reconstitute anything here.
    Including this function only for consistency.
    
    Though, note to self: If I want to protect from my own manual edits intended for
    future runs while a current run is executing, I could in fact import and reconstitute
    everything, so that changes to this file would be ignored on terminate/restart/import.
    """
    pass
