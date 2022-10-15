"""Simulate the towing of the margin toward the vegetal pole by microtubule arrays in the yolk cells"""
import math

from epiboly_init import *
from utils import tf_utils as tfu

def update_tangent_forces(magnitude: int) -> None:
    """Note that once this has run, turning it off does not remove existing forces. Use remove_tangent_forces().
    
    Still to do! This needs a stopping criterion. Based on angle of phi? Or distance of particle
    from the vegetal pole? Needs criterion both for the individual particle, and for when all of them
    have arrived.
    """
    # For now, add a vector of fixed magnitude, in the tangent direction
    big_particle = Big.particle(0)
    for p in LeadingEdge.items():
        r, theta, phi = p.sphericalPosition(particle=big_particle)
        tangent_phi = phi + math.pi / 2
        tangent_force_vec: tf.fVector3 = tfu.cartesian_from_spherical([magnitude, theta, tangent_phi])

        # The assignment runs into the copy-constructor bug! So change to plain list
        p.force_init = tangent_force_vec.as_list()

def remove_tangent_forces() -> None:
    """Call this once to remove tangent forces from all particles, after turning off the updates."""
    for p in LeadingEdge.items():
        p.force_init = [0, 0, 0]
    print("Tangent forces removed")

