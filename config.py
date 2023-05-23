"""Config

Flags and magic numbers, especially ones used in more than one place.
"""
import math

# For normal initialization, leave blank. To start from a previously exported state, provide the
# directory name of the export. This is the parent directory with the datetime in the name. Script
# will search in the "Sim_state" subdirectory to find the most recent export, and start with that.
initialization_directory_name: str = ""

# Whether to use TF windowless mode, in which the simulation is driven only by
# tf.step() and never tf.show(), and no graphics are displayed.
# But name this flag as a positive rather than a negative, to avoid confusing double negatives ("not windowless").
windowed_mode: bool = False

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
screenshot_export_interval: int = 10

cell_division_enabled: bool = True
plot_time_averages: bool = True

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

# Cell division rate parameters. See justification in docstring of cell_division.cell_division().
total_epiboly_divisions: int = 7500
expected_timesteps: int  # For now, not a constant and not used; TBD

# Search for neighbors within this distance (multiple of particle radius) to set up initial bond network.
min_neighbor_initial_distance_factor: float = 1.5

# Some items for Potential- and Bond-making:
harmonic_repulsion_spring_constant: float = 5.0
harmonic_spring_constant: float = 12.0
harmonic_edge_spring_constant: float = 12.0  # (for the Bonds)
harmonic_angle_spring_constant: float = 1.0  # (for the Angles)
harmonic_angle_tolerance: float = 0.008 * math.pi

# Vegetalward forces applied to LeadingEdge. Probably will later want to change these dynamically,
# but these are good initial values. These are highly dependent on the particle radius, spring constants, etc.,
# so have to be tuned accordingly if those change.
# yolk_cortical_tension: force generated within the yolk, balancing out the EVL internal tension
#   so that the leading edge is stable (not advancing) until some extrinsic additional force is applied.
# external_force: additional force applied to drive epiboly.
yolk_cortical_tension: int = 120    # just balances interior bonds at initialization
external_force: int = 255   # +255 to produce full epiboly

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
