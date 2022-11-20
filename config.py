"""Config

Magic numbers, especially ones used in more than one place.
"""
import math

from epiboly_init import Little, LeadingEdge

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

# See comments in function initialize_interior() in main for explanation
num_leading_edge_points: int = 60
num_spherical_positions: int = 2000  # 2050
edge_margin_interior_points: float = 0.15

# Some items for bond-making:
# harmonic potential:
harmonic_spring_constant: float = 7.0

# Potential.max any greater than this, numerical problems ensue
max_potential_cutoff: float = 6

# Huge maximum for neighbor-finding distance_factor in bond-making algorithm, that should never be reached.
# Just insurance against a weird infinite loop.
# this value used as distance_factor will result in an absolute search distance = max_potential_cutoff
max_distance_factor: float = max_potential_cutoff / Little.radius

# For neighbor angle energy calculations. Not only to avoid magic numbers,
# but also because these would otherwise be calculated millions of times,
# which is wasteful, even for an inexpensive operation.
target_neighbor_angle: float = math.pi / 3
target_edge_angle: float = math.pi
leading_edge_recruitment_limit: float = 2 * LeadingEdge.radius
two_pi: float = math.pi * 2
