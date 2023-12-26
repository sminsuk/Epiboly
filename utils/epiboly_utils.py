"""Epiboly utilities

These are utility functions specific to this simulation.
"""
import numpy as np
from statistics import fmean

import tissue_forge as tf
import epiboly_globals as g
import utils.tf_utils as tfu

# parking place for some cumulative measures that other modules can write to, and read from:
# Keep track of movement of EVL cells into and out of the leading edge:
cumulative_to_edge: int = 0
cumulative_from_edge: int = 0

def reset_camera():
    """A good place to park the camera for epiboly
    
    Future note: I'd like to be able to enable lagging here, programmatically, but it's missing from the API.
    TJ will add it in a future release.
    """
    tf.system.camera_view_front()
    tf.system.camera_zoom_to(-12)

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

def embryo_cartesian_coords(p: tf.fVector3 | tf.ParticleHandle) -> tf.fVector3:
    """x, y, z relative to the yolk center
    
    Overload to get theta, phi based on either a position or an existing particle.
    """
    yolk_particle: tf.ParticleHandle = g.Big.items()[0]
    normal_vec: tf.fVector3 = p.position - yolk_particle.position
    return normal_vec

def random_tangent(p: tf.ParticleHandle) -> tf.fVector3:
    """Return a vector pointing randomly within the plane tangent to the yolk surface at particle p"""
    normal_vec: tf.fVector3 = embryo_cartesian_coords(p)
    return tfu.random_perpendicular(normal_vec)

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

def leading_edge_circumference() -> float:
    hypotenuse: float = g.Big.radius + g.Little.radius
    yolk: tf.ParticleHandle = g.Big.items()[0]
    leading_edge_height: float = leading_edge_mean_z() - yolk.position.z()
    leading_edge_radius: float = np.sqrt(np.square(hypotenuse) - np.square(leading_edge_height))
    return 2 * np.pi * leading_edge_radius

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
    radius_percentage: float = 2 * epiboly_percentage
    adjacent: float = 100 - radius_percentage
    cosine_phi: float = adjacent / 100
    phi_rads = np.arccos(cosine_phi)
    # print("intermediate results: radius_percentage, adjacent, cosine_phi, degrees =",
    #       radius_percentage, adjacent, cosine_phi, np.rad2deg(phi_rads))
    print(f"{epiboly_percentage}% epiboly = {round(phi_rads, 4)} radians or {round(np.rad2deg(phi_rads), 2)} degrees")
    return phi_rads

def get_state() -> dict:
    """In composite runs, save state of accumulated cell migration statistics"""
    return {"cumulative_from_edge": cumulative_from_edge,
            "cumulative_to_edge": cumulative_to_edge}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global cumulative_from_edge, cumulative_to_edge
    cumulative_from_edge = d["cumulative_from_edge"]
    cumulative_to_edge = d["cumulative_to_edge"]
