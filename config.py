"""Config

Flags and magic numbers, especially ones used in more than one place.
"""
import math

# Whether to use TF windowless mode, in which the simulation is driven only by
# tf.step() and never tf.show(), and no graphics are displayed.
# But name this flag as a positive rather than a negative, to avoid confusing double negatives ("not windowless").
windowed_mode: bool = False

# Whether to show the equilibration steps.
# In windowless mode, whether to include them in any exported screenshots;
# in windowed mode, whether to show them in the simulator window (and any exported screenshots), or hide in tf.step();
# useful in windowed mode to set True during development so I can see what I'm doing; otherwise leave as False.
show_equilibration: bool = False

# Number of timesteps between screenshots. User configurable here at load time. Can also be adjusted dynamically
# at run time (only to adjust the value, not to enable/disable), but callers should never set this directly;
# instead use the setter in module video_export, which owns it. To disable screenshot export, set to 0.
screenshot_export_interval: int = 10

# Useful to turn this off while tuning setup and equilibration. When external force is artificially low,
# and if Angle bonds too high, they cause instability and big waviness in the leading edge. (Seems fixed now.)
angle_bonds_enabled: bool = True

# real value for the misnomer "30% epiboly")
epiboly_initial_percentage: int = 43

# How many leading edge and interior cells to make
num_leading_edge_points: int = 110
num_spherical_positions: int = 5000

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
leading_edge_recruitment_min_angle: float = math.pi / 2.5   # empirically determined

# For the same reason, just some common numbers useful in a variety of contexts
two_pi: float = math.pi * 2
pi_over_2: float = math.pi / 2
