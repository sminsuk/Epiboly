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

# Just while determining empirically, the right number of interior particles to use.
# If there are too many, with frozen LeadingEdge, they'll pop past. If too few,
# they'll never fill the space no matter how long they equilibrate. Once the right
# number is determined, they should fill the space without popping through. Then
# Can comment this out and never use it again. (Or, perhaps will always use it during
# equilibration step, and release it afterward.)
# With this approach, instead of guessing how many particles to use and letting the
# leading edge get pushed downward by their expansion to an appropriate place, we'll
# set the location we *want* the leading edge, and get the right number of particles
# in there.
# LeadingEdge.frozen_z = True
epiboly_initial_percentage: int = 43     # real value for the misnomer "30% epiboly")

# How many leading edge and interior cells to make
num_leading_edge_points: int = 105
num_spherical_positions: int = 4950

# Some items for Potential- and Bond-making:
harmonic_repulsion_spring_constant: float = 5.0
harmonic_spring_constant: float = 12.0
harmonic_edge_spring_constant: float = 12.0  # (for the Bonds)
harmonic_angle_spring_constant: float = 3.0  # (for the Angles)
harmonic_angle_tolerance: float = 0.008 * math.pi
angle_bonds_enabled: bool = True

# Potential.max any greater than this, numerical problems ensue
max_potential_cutoff: float = 6

stopping_condition_phi: float = math.pi * 0.95

# For neighbor count criterion. Pre-energy-calculation limits.
# (If exceeded, don't bother calculating energy, just reject the change.)
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
