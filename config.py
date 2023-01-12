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
num_leading_edge_points: int = 75
num_spherical_positions: int = 2000  # 2050
edge_margin_interior_points: float = 0.15

# Some items for bond-making:
# harmonic potential:
harmonic_spring_constant: float = 7.0
harmonic_edge_spring_constant: float = 7.0  # (for the Bonds)
harmonic_angle_spring_constant: float = 5.0  # (for the Angles)
harmonic_angle_tolerance: float = 0.008 * math.pi
angle_bonds_enabled: bool = True

def harmonic_angle_equilibrium_value() -> float:
    """A function because it depends on the number of particles in the ring"""
    # Equilibrium angle might look like π from within the plane of the leading edge, but the actual angle is
    # different. And, it changes if the number of leading edge particles changes. Hopefully it won't need to be
    # dynamically updated to be that precise. If the number of particles changes, they'll "try" to reach a target angle
    # that is not quite right, but will be opposed by the same force acting on the neighbor particles, so hopefully
    # it all balances out. (For the same reason, π would probably also work, but this value is closer to the real one.)
    return math.pi - (two_pi / len(LeadingEdge.items()))

# Potential.max any greater than this, numerical problems ensue
max_potential_cutoff: float = 6

stopping_condition_phi: float = math.pi * 0.95

# Huge maximum for neighbor-finding distance_factor in bond-making algorithm, that should never be reached.
# Just insurance against a weird infinite loop.
# this value used as distance_factor will result in an absolute search distance = max_potential_cutoff
max_distance_factor: float = max_potential_cutoff / Little.radius

# For neighbor count criterion. Pre-energy-calculation limits.
# (If exceeded, don't bother calculating energy, just reject the change.)
min_neighbor_count: int = 3
max_edge_neighbor_count: int = 3

# Adhesion energies: named, and for convenience, also in a dict[type_id: dict[type_id: energy]]
energy_little_little: float = 5.0
energy_little_edge: float = 10.0
energy_edge_edge: float = 20.0
adhesion_energy = {Little.id: {Little.id: energy_little_little,
                               LeadingEdge.id: energy_little_edge},
                   LeadingEdge.id: {Little.id: energy_little_edge,
                                    LeadingEdge.id: energy_edge_edge}}

# For neighbor angle energy calculations. Not only to avoid magic numbers,
# but also because these would otherwise be calculated millions of times,
# which is wasteful, even for an inexpensive operation.
target_neighbor_angle: float = math.pi / 3
target_edge_angle: float = math.pi
leading_edge_recruitment_limit: float = 2 * LeadingEdge.radius
leading_edge_recruitment_min_angle: float = math.pi / 3.5   # empirically determined

# For the same reason, just some common numbers useful in a variety of contexts
two_pi: float = math.pi * 2
pi_over_2: float = math.pi / 2
