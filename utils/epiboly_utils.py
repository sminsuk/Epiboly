"""Epiboly utilities

These are utility functions specific to this simulation.
"""
import numpy as np
from statistics import fmean

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import neighbors as nbrs            # Caution, circular import; if causes a problem, move the needed function to tfu
import utils.global_catalogs as gc
import utils.tf_utils as tfu

# parking place for some values that other modules can write to, and read from:

# Keep track of cumulative movement of EVL cells into and out of the leading edge:
cumulative_to_edge: int = 0
cumulative_from_edge: int = 0
cumulative_edge_divisions: int = 0

# Are we in the main sim or in the recoil experiment?
recoil_experiment_in_progress: bool = False

# Timestep, and leading edge mean phi, when cell division terminated.
# Plot module needs to know them in order to mark them on plots.
# phi value is known in advance and set by cell division module when it is initialized.
# timestep value isn't known until phi threshold is crossed, and cell division module can't set it because
# it isn't tracking timesteps; so plot module does that (by testing epiboly progress against phi value).
cell_division_cessation_timestep: int = 0
cell_division_cessation_phi: float = 0.0

# kludge/hack alert: this is just to quickly test something; not the right way to do this!
# If True:
# For the first 600 timesteps, override external force and turn off cell division (i.e. temporary balanced
# force control), then after that delay, set this to False and adopt whatever the configured behavior is.
balanced_force_equilibration_kludge: bool = cfg.balanced_force_equilibration_kludge

# Cell radius is distinct from PARTICLE radius. It represents the extent of the cell, an average distance
# from the center of mass, to the edge of the cell. Particles only represent the point center of mass, and their
# radii are mainly for visualization (though the TF neighbor search also uses them, to define the search space).
# Cell radius is used to determine the equilibrium distances of Potentials between EVL cells, and therefore
# determines the effective radius: how close particles can get to one another. This allows us to decouple cell
# size from particle size. Particles are not intended to represent cells, just their centers.
# Note that this is not the actual extent of the current cell, but more of a "target" value. So when bonds get made,
# they can be under tension because the centers are actually further apart than the r0 we give to the potential.
# Note also that for yolk-to-evl potentials, we'll still use particle radius. That way, the EVL doesn't have to get
# thicker, just because the cells get larger in apical surface area. So the particles can still hug the yolk surface
# (it would look really weird if they didn't), and we achieve a "squamous cell" effect in TF even though TF only
# knows about spheres.
#
# This will be calculated from the desired number of cells during initialization of the simulation,
# and stored here for later use.
initial_cell_radius: float = 0

# Central place to define colors used in the simulation, by their purpose.
# Everywhere else, use these rather than the color names
evl_undivided_color: tf.fVector3 = tfu.cornflower_blue
evl_divided_color: tf.fVector3 = evl_undivided_color
evl_margin_undivided_color: tf.fVector3 = tfu.gold
evl_margin_divided_color: tf.fVector3 = evl_margin_undivided_color
if cfg.color_code_daughter_cells:
    evl_divided_color = tfu.lighter_blue
    evl_margin_divided_color = tfu.dk_yellow_brown
lineage_unlabeled_color: tf.fVector3 = tfu.gray
lineage_labeled_color: tf.fVector3 = tfu.red
yolk_unlabeled_color: tf.fVector3 = tfu.gray
lineage_tracing_bond_color: tf.fVector3 = tfu.gray

# Bond color actually set at time of bond creation, but at least for now is fixed at start of sim.
# So just check the config once at the start. Set only for lineage tracing, else allow to default:
lineage_tracing_patterns = [cfg.PaintPattern.ORIGINAL_TIER,
                            cfg.PaintPattern.VERTICAL_STRIPE,
                            cfg.PaintPattern.PATCH]
bond_color: None | tf.fVector3 = None
if cfg.paint_pattern in lineage_tracing_patterns:
    bond_color = lineage_tracing_bond_color

def is_undivided(p: tf.ParticleHandle) -> bool:
    """Determine whether particle is undivided, based on its CELL radius"""
    # Testing greater-than with a tolerance threshold, instead of just equality, because
    # initial_cell_radius is an arbitrary float, so I don't want to rely on equality comparison.
    return gc.get_cell_radius(p) > 0.9 * initial_cell_radius

def is_divided(p: tf.ParticleHandle) -> bool:
    """Convenience function"""
    return not is_undivided(p)

def bond_tension(bhandle: tf.BondHandle) -> float:
    """Return the tension on a bond
    
    This depends on the spring constant, which cannot be read out of the potential!
    """
    k: float = gc.get_spring_constant(bhandle)
    return k * (bhandle.length - bhandle.potential.r0)

def tension(p: tf.ParticleHandle) -> float:
    """Return the aggregate tension on a particle, which is the mean of the signed tension of all its bonds"""
    p_bonds: list[tf.BondHandle] = tfu.bonds(p)
    return 0 if not p_bonds else fmean([bond_tension(bhandle) for bhandle in p_bonds])

def update_color(p: tf.ParticleHandle) -> None:
    """Paint the particle the correct color for its ParticleType and cell division state
    
    Useful after .become() or after cell division
    Cell must be in the global dictionary, with its correct cell radius assigned, and must have a .style object
    """
    match cfg.paint_pattern:
        case cfg.PaintPattern.CELL_TYPE:
            if p.type() == g.LeadingEdge:
                p.style.color = (evl_margin_undivided_color if is_undivided(p)
                                 else evl_margin_divided_color)
            elif p.type() == g.Little:
                p.style.color = (evl_undivided_color if is_undivided(p)
                                 else evl_divided_color)
        case cfg.PaintPattern.ORIGINAL_TIER | cfg.PaintPattern.VERTICAL_STRIPE | cfg.PaintPattern.PATCH:
            # lineage tracing patterns. Depends on the lineage tracer having been set at initialization
            if p.type() == g.Big:
                p.style.color = yolk_unlabeled_color
            elif gc.get_lineage_tracer(p):
                p.style.color = lineage_labeled_color
            else:
                p.style.color = lineage_unlabeled_color

def update_all_particle_colors():
    """ (For anticipated future use with .SPECIES) """
    p: tf.ParticleHandle
    for p in g.Little.items():
        update_color(p)
    for p in g.LeadingEdge.items():
        update_color(p)
    update_color(g.Big.items()[0])

def reset_camera():
    """A good place to park the camera for epiboly
    
    Future note: I'd like to be able to enable lagging here, programmatically, but it's missing from the API.
    TJ will add it in a future release.
    """
    tf.system.camera_view_front()
    tf.system.camera_zoom_to(-12)

def embryo_radius() -> float:
    return g.Big.radius + g.Little.radius

def embryo_phi(p: tf.fVector3 | tf.ParticleHandle) -> float:
    """phi relative to the animal/vegetal axis
    
    Overload to get phi based on either a position or an existing particle.
    """
    theta, phi = embryo_coords(p)
    return phi

def embryo_theta(p: tf.fVector3 | tf.ParticleHandle) -> float:
    """theta relative to the animal/vegetal axis
    
    Overload to get theta based on either a position or an existing particle.
    """
    theta, phi = embryo_coords(p)
    return theta

def embryo_coords(p: tf.fVector3 | tf.ParticleHandle, rotation_matrix: np.ndarray = None) -> tuple[float, float]:
    """theta, phi relative to the animal/vegetal axis
    
    Overload to get theta, phi based on either a position or an existing particle.
    
    To get a particle's position relative to an axis other than animal-vegetal (other than the z-axis),
    pass a transformation matrix that rotates the desired axis to vertical.
    """
    if rotation_matrix is None:
        rotation_matrix = np.identity(3)
        
    position: tf.fVector3 = p if type(p) is tf.fVector3 else p.position
    yolk_particle: tf.ParticleHandle = g.Big.items()[0]
    
    # For the rotation to work correctly, need to subtract the yolk center manually before rotating;
    # then convert cartesian to spherical relative to the origin. (As opposed to just making use of the
    # "origin" parameter of the conversion function to convert relative to the yolk center, because
    # then the rotation would be around the wrong point.)
    position -= yolk_particle.position
    rotated_position: tf.fVector3 = tf.fVector3(np.dot(rotation_matrix, position))
    r, theta, phi = tf.metrics.cartesian_to_spherical(postion=rotated_position, origin=tf.fVector3(0, 0, 0))
    # (sic, argument name is misspelled in the API)
    
    return theta, phi

def embryo_cartesian_coords(p: tf.ParticleHandle) -> tf.fVector3:
    """x, y, z relative to the yolk center"""
    yolk_particle: tf.ParticleHandle = g.Big.items()[0]
    normal_vec: tf.fVector3 = p.position - yolk_particle.position
    return normal_vec

def random_tangent(p: tf.ParticleHandle) -> tf.fVector3:
    """Return a vector pointing randomly within the plane tangent to the yolk surface at particle p"""
    normal_vec: tf.fVector3 = embryo_cartesian_coords(p)
    return tfu.random_perpendicular(normal_vec)

def horizontal_tangent(p: tf.ParticleHandle) -> tf.fVector3:
    """Return a unit vector pointing horizontally within the plane tangent to the yolk surface at particle p"""
    theta, phi = embryo_coords(p)
    return tf.fVector3(tfu.cartesian_from_spherical([1, theta + np.pi / 2, np.pi / 2]))

def vegetalward(p: tf.ParticleHandle) -> tf.fVector3:
    """Return a unit vector pointing in the vegetalward direction from particle p"""
    theta, phi = embryo_coords(p)
    return tf.fVector3(tfu.cartesian_from_spherical([1, theta, phi + np.pi / 2]))
    
def leading_edge_max_phi() -> float:
    """phi of the most progressed leading edge particle (or 0 if there are none - i.e. before any are instantiated)"""
    return max([embryo_phi(particle) for particle in g.LeadingEdge.items()], default=0)

def leading_edge_mean_phi() -> float:
    """mean phi for all leading edge particles (or 0 if there are none - i.e. before any are instantiated)"""
    phi_values = [embryo_phi(particle) for particle in g.LeadingEdge.items()]
    return fmean(phi_values) if phi_values else 0

def leading_edge_mean_z() -> float:
    """mean z for all leading edge particles"""
    particle: tf.ParticleHandle
    z_values = [particle.position.z() for particle in g.LeadingEdge.items()]
    return fmean(z_values)
    
def leading_edge_min_mean_max_phi() -> tuple[float, float, float]:
    """minimum, mean, and max phi for all leading edge particles"""
    phi_values = [embryo_phi(particle) for particle in g.LeadingEdge.items()]
    return min(phi_values), fmean(phi_values), max(phi_values)

def internal_evl_max_phi() -> float:
    """phi of the most progressed Little (internal EVL) particle

    This is useful for plots that only consider the internal particles, like the binned tension plot
    """
    return max([embryo_phi(particle) for particle in g.Little.items()])

def radius_of_circle_at_relative_z(relative_z: float) -> float:
    """Radius of a circle on the surface of the embryo (latitude line) at a given z value relative to the equator

    :param relative_z: height above or below the equator (sign does not matter, because of symmetry of the result)
    :return: radius of the circle at that height
    """
    return np.sqrt(np.square(embryo_radius()) - np.square(relative_z))

def radius_of_circle_at_particle(p: tf.ParticleHandle) -> float:
    return np.sqrt(np.square(p.position.x()) + np.square(p.position.y()))

def circumference_of_circle_at_relative_z(relative_z: float) -> float:
    """Circumference of a circle on the surface of the embryo (latitude line) at a given z value relative to the equator
    
    :param relative_z: height above or below the equator (sign does not matter, because of symmetry of the result)
    :return: circumference of the circle at that height
    """
    return 2 * np.pi * radius_of_circle_at_relative_z(relative_z)

def circumference_of_circle_at_z(z: float) -> float:
    yolk: tf.ParticleHandle = g.Big.items()[0]
    relative_z: float = z - yolk.position.z()
    return circumference_of_circle_at_relative_z(relative_z)

def circumference_at_particle(p: tf.ParticleHandle) -> float:
    return circumference_of_circle_at_z(p.position.z())

def leading_edge_circumference() -> float:
    """Circumference of the idealized marginal ring (circumference of a circle at mean z of all the margin particles)"""
    return circumference_of_circle_at_z(leading_edge_mean_z())

def fraction_of_radius_above_equator(epiboly_percentage: float) -> float:
    """Vertical position on the embryo expressed as a fraction of the embryo radius
    
    :param epiboly_percentage: standard zebrafish staging metric; see definition in phi_for_epiboly()
    :return: distance above the equator as a fraction of embryo radius; ranges from -1 (vegetal pole)
        to +1 (animal pole)
    """
    percentage_above_equator: float = 50 - epiboly_percentage
    return percentage_above_equator / 50

def relative_z_from_epiboly_percentage(epiboly_percentage: float) -> float:
    """Return height above the equator for a given position on the embryo surface
    
    :param epiboly_percentage: standard zebrafish staging metric; see definition in phi_for_epiboly()
    :return: height above the equator (negative if position is below the equator)
    """
    return embryo_radius() * fraction_of_radius_above_equator(epiboly_percentage)

def get_leading_edge_ordered_particles() -> list[tf.ParticleHandle]:
    """For iterating over leading edge particles, get them in order of the bonded ring

    Rather than the previous shortcut of just sorting on theta and then iterating in that order, follow
    the bonds to find the exact sequence of particles and iterate over that instead. This improves accuracy
    by accounting for the rare situation when the edge gets kinked, doubling back on itself locally.
    """
    particles: tf.ParticleList = g.LeadingEdge.items()
    assert len(particles) > 0, "Can't order empty particle list!"
    current_particle: tf.ParticleHandle = particles[0]
    ordered_particles: list[tf.ParticleHandle] = [current_particle]
    neighbors: list[tf.ParticleHandle] = nbrs.bonded_neighbors_of_types(current_particle, [g.LeadingEdge])
    assert len(neighbors) == 2, "Something messed up in the connected edge ring"
    while not (neighbors[0] in ordered_particles and neighbors[1] in ordered_particles):
        next_particle: tf.ParticleHandle = neighbors[0]
        if next_particle in ordered_particles:
            next_particle = neighbors[1]
        ordered_particles.append(next_particle)
        current_particle = next_particle
        neighbors = nbrs.bonded_neighbors_of_types(current_particle, [g.LeadingEdge])
    return ordered_particles

def leading_edge_best_fit_plane() -> tuple[tf.fVector3, tf.fVector3]:
    """Return the best-fit plane to the leading-edge particle positions

    This function was written with advice from ChatGPT 3.5.

    :return: a tuple of two vectors. The first vector is the centroid of all the leading edge particle positions,
             which should lie on the best-fit plane. The second vector is a unit vector normal to the plane
             and points upward.
    """
    covariance_matrix = np.cov(g.LeadingEdge.items().positions, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # eigenvector corresponding to the smallest eigenvalue is the normal vector to the plane
    normal_vec: tf.fVector3 = tf.fVector3(eigenvectors[:, np.argmin(eigenvalues)])
    
    # One thing I'm not sure of is, since a normal vector to a plane can point in either of two directions,
    # will this one always point in one consistent direction, either in terms of the coordinate system (up/down),
    # or in terms of the center of the sphere (toward/away)? We want to always be pointing up, so to be on the
    # safe side, just test it and flip it if necessary:
    if normal_vec.z() < 0:
        normal_vec *= -1
    return g.LeadingEdge.items().centroid, normal_vec

def rotation_matrix(axis: tf.fVector3) -> np.ndarray:
    """Return a transformation matrix to rotate an axis to vertical"""
    r, theta, phi = tfu.spherical_from_cartesian(axis)
    angle: float = phi
    
    # Courtesy ChatGPT 3.5, after fact-checking:
    rotation_axis = np.cross(axis, [0, 0, 1])
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_matrix = np.array(
            [[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
              rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
              rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
             [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
              np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
              rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
             [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
              rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
              np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]]
            )
    return rotation_matrix

def phi_for_epiboly(epiboly_percentage: float) -> float:
    """Convert % epiboly into phi for spherical coordinates (in radians)

    epiboly_percentage: % of *vertical* distance from animal to vegetal pole (not % of arc).
    Note that % of vertical distance equals % of embryo surface area, because the two are directly proportional.
    From staging description at zfin.org:

    'The extent to which the blastoderm has spread over across the yolk cell provides an extremely useful staging
    index from this stage until epiboly ends. We define percent-epiboly to mean the fraction of the yolk cell that
    the blastoderm covers; percent-coverage would be a more precise term for what we mean to say, but
    percent-epiboly immediately focuses on the process and is in common usage. Hence, at 30%-epiboly the blastoderm
    margin is at 30% of the entire distance between the animal and vegetal poles, as one estimates along the
    animal-vegetal axis.'
    
    (Note, it's not really true! The NAME of the developmental stage "30% epiboly" is not quantitative. Placing the
    leading-edge cells at 30% epiboly by this definition, produces a configuration not resembling any published
    photo of a "30% epiboly" stage embryo, but instead has an abnormally high leading edge. By trial and error,
    I find that placing the leading-edge cells at 43% epiboly, comes closest to resembling those published photos.
    Thus, embryos customarily referred to as "30% epiboly" actually are at around 43% epiboly, using the
    quantitative definition. In fact, from the canonical figure 8F in Kimmel et al. 1995, also found here:
    https://zfin.org/zf_info/zfbook/stages/figs/fig8.html - it turns out you can measure the 43% directly!)
    """
    cosine_phi: float = fraction_of_radius_above_equator(epiboly_percentage)
    phi_rads: float = np.arccos(cosine_phi)
    rounded_phi: float = round(phi_rads, 4)
    pi_factor: float = round(phi_rads / np.pi, 2)
    degrees: float = round(np.rad2deg(phi_rads), 2)
    print(f"{epiboly_percentage}% epiboly = {rounded_phi} ({pi_factor} x pi) radians or {degrees} degrees")
    return phi_rads

def get_state() -> dict:
    """In composite runs, save state of sim"""
    return {"cumulative_from_edge": cumulative_from_edge,
            "cumulative_to_edge": cumulative_to_edge,
            "cumulative_edge_divisions": cumulative_edge_divisions,
            "recoil_experiment_in_progress": recoil_experiment_in_progress,
            "initial_cell_radius": initial_cell_radius,
            "cell_division_cessation_timestep": cell_division_cessation_timestep,
            "cell_division_cessation_phi": cell_division_cessation_phi,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global cumulative_from_edge, cumulative_to_edge, cumulative_edge_divisions
    global recoil_experiment_in_progress, initial_cell_radius
    global cell_division_cessation_timestep, cell_division_cessation_phi
    cumulative_from_edge = d["cumulative_from_edge"]
    cumulative_to_edge = d["cumulative_to_edge"]
    cumulative_edge_divisions = d["cumulative_edge_divisions"]
    recoil_experiment_in_progress = d["recoil_experiment_in_progress"]
    initial_cell_radius = d["initial_cell_radius"]
    cell_division_cessation_timestep = d["cell_division_cessation_timestep"]
    cell_division_cessation_phi = d["cell_division_cessation_phi"]

if __name__ == "__main__":
    """Print a correspondence between epiboly percentage, and phi"""
    phi_for_epiboly(cfg.epiboly_initial_percentage)
    print()
    for epiboly_percentage in range(30, 101):
        phi_for_epiboly(epiboly_percentage)
        