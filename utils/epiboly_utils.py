"""Epiboly utilities

These are utility functions specific to this simulation.
"""
from statistics import fmean

import tissue_forge as tf
import epiboly_globals as g

def reset_camera():
    """A good place to park the camera for epiboly
    
    Future note: I'd like to be able to enable lagging here, programmatically, but it's missing from the API.
    TJ will add it in a future release.
    """
    tf.system.camera_view_front()
    tf.system.camera_zoom_to(-12)

def embryo_phi(particle: tf.ParticleHandle) -> float:
    """phi relative to the animal/vegetal axis"""
    r, theta, phi = particle.sphericalPosition(particle=g.Big.particle(0))
    return phi

def embryo_theta(particle: tf.ParticleHandle) -> float:
    """theta relative to the animal/vegetal axis"""
    r, theta, phi = particle.sphericalPosition(particle=g.Big.particle(0))
    return theta

def embryo_coords(particle: tf.ParticleHandle) -> tuple[float, float]:
    """theta, phi relative to the animal/vegetal axis"""
    r, theta, phi = particle.sphericalPosition(particle=g.Big.particle(0))
    return theta, phi

def leading_edge_max_phi() -> float:
    """phi of the most progressed leading edge particle"""
    return max([embryo_phi(particle) for particle in g.LeadingEdge.items()])

def leading_edge_mean_phi() -> float:
    """mean phi for all leading edge particles"""
    phi_values = [embryo_phi(particle) for particle in g.LeadingEdge.items()]
    return fmean(phi_values)
    
def leading_edge_min_mean_max_phi() -> tuple[float, float, float]:
    """minimum, mean, and max phi for all leading edge particles"""
    phi_values = [embryo_phi(particle) for particle in g.LeadingEdge.items()]
    return min(phi_values), fmean(phi_values), max(phi_values)

def leading_edge_velocity_z() -> float:
    """mean z velocity of all leading edge particles"""
    p: tf.ParticleHandle
    veloc_z_values = [p.velocity.z() for p in g.LeadingEdge.items()]
    return fmean(veloc_z_values)

def internal_evl_max_phi() -> float:
    """phi of the most progressed Little (internal EVL) particle

    This is useful for plots that only consider the internal particles, like the binned tension plot
    """
    return max([embryo_phi(particle) for particle in g.Little.items()])
