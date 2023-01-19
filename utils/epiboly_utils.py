"""Epiboly utilities

These are utility functions specific to this simulation.
"""
import tissue_forge as tf
from epiboly_init import Big, LeadingEdge

def reset_camera():
    """A good place to park the camera for epiboly
    
    Future note: I'd like to be able to enable lagging here, programmatically, but it's missing from the API.
    TJ will add it in a future release.
    """
    tf.system.camera_view_front()
    tf.system.camera_zoom_to(-12)

def embryo_phi(particle: tf.ParticleHandle) -> float:
    """phi relative to the animal/vegetal axis"""
    r, theta, phi = particle.sphericalPosition(particle=Big.particle(0))
    return phi

def embryo_theta(particle: tf.ParticleHandle) -> float:
    """theta relative to the animal/vegetal axis"""
    r, theta, phi = particle.sphericalPosition(particle=Big.particle(0))
    return theta

def embryo_coords(particle: tf.ParticleHandle) -> tuple[float, float]:
    """theta, phi relative to the animal/vegetal axis"""
    r, theta, phi = particle.sphericalPosition(particle=Big.particle(0))
    return theta, phi

def leading_edge_max_phi() -> float:
    """phi of the most progressed leading edge particle"""
    return max([embryo_phi(particle) for particle in LeadingEdge.items()])
