""""""
import math

from epiboly_init import *
import sharon_utils as su

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
        tangent_unit_vec = su.cartesian_from_spherical([1.0, theta, tangent_phi])

        # product is an fVector3, and the assignment runs into the copy-constructor bug! So change to plain list
        p.force_init = (tangent_unit_vec * magnitude).as_list()

def remove_tangent_forces() -> None:
    """Call this once to remove tangent forces from all particles, after turning off the updates."""
    for p in LeadingEdge.items():
        p.force_init = [0, 0, 0]
    print("Tangent forces removed")

