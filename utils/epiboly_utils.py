"""Epiboly utilities

These are utility functions specific to this simulation.
"""
import tissue_forge as tf
from epiboly_init import Big

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
