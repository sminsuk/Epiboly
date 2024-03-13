"""Epiboly utilities

These are utility functions specific to this simulation.
"""
import numpy as np
from statistics import fmean

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import utils.global_catalogs as gc
import utils.tf_utils as tfu

# parking place for some values that other modules can write to, and read from:

# Keep track of cumulative movement of EVL cells into and out of the leading edge:
cumulative_to_edge: int = 0
cumulative_from_edge: int = 0
cumulative_edge_divisions: int = 0

# Are we in the main sim or in the recoil experiment?
recoil_experiment_in_progress: bool = False

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
    if p.type() == g.LeadingEdge:
        p.style.color = (evl_margin_undivided_color if is_undivided(p)
                         else evl_margin_divided_color)
    elif p.type() == g.Little:
        p.style.color = (evl_undivided_color if is_undivided(p)
                         else evl_divided_color)

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

def embryo_coords(p: tf.fVector3 | tf.ParticleHandle) -> tuple[float, float]:
    """theta, phi relative to the animal/vegetal axis
    
    Overload to get theta, phi based on either a position or an existing particle.
    """
    position: tf.fVector3 = p if type(p) is tf.fVector3 else p.position
    yolk_particle: tf.ParticleHandle = g.Big.items()[0]
    r, theta, phi = tf.metrics.cartesian_to_spherical(postion=position, origin=yolk_particle.position)
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

def circumference_of_circle_at_relative_z(relative_z: float) -> float:
    """Circumference of a circle on the surface of the embryo (latitude line) at a given z value relative to the equator
    
    :param relative_z: height above or below the equator (sign does not matter, because of symmetry of the result)
    :return: circumference of the circle at that height
    """
    return 2 * np.pi * radius_of_circle_at_relative_z(relative_z)

def leading_edge_circumference() -> float:
    """Circumference of the idealized marginal ring (circumference of a circle at mean z of all the margin particles)"""
    yolk: tf.ParticleHandle = g.Big.items()[0]
    leading_edge_height: float = leading_edge_mean_z() - yolk.position.z()
    return circumference_of_circle_at_relative_z(leading_edge_height)

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

def phi_for_epiboly(epiboly_percentage: float):
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
    phi_rads = np.arccos(cosine_phi)
    print(f"{epiboly_percentage}% epiboly = {round(phi_rads, 4)} radians or {round(np.rad2deg(phi_rads), 2)} degrees")
    return phi_rads

def get_state() -> dict:
    """In composite runs, save state of sim"""
    return {"cumulative_from_edge": cumulative_from_edge,
            "cumulative_to_edge": cumulative_to_edge,
            "cumulative_edge_divisions": cumulative_edge_divisions,
            "recoil_experiment_in_progress": recoil_experiment_in_progress,
            "initial_cell_radius": initial_cell_radius,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global cumulative_from_edge, cumulative_to_edge, cumulative_edge_divisions
    global recoil_experiment_in_progress, initial_cell_radius
    cumulative_from_edge = d["cumulative_from_edge"]
    cumulative_to_edge = d["cumulative_to_edge"]
    cumulative_edge_divisions = d["cumulative_edge_divisions"]
    recoil_experiment_in_progress = d["recoil_experiment_in_progress"]
    initial_cell_radius = d["initial_cell_radius"]
