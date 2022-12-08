"""Simulate the towing of the margin toward the vegetal pole by microtubule arrays in the yolk cells"""
import math
from typing import NamedTuple

import tissue_forge as tf
from epiboly_init import Big, LeadingEdge
import config as cfg
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

def apply_even_tangent_forces(magnitude: int) -> None:
    class ParticleData(NamedTuple):
        phandle: tf.ParticleHandle
        theta: float
        phi: float
        
    def get_particle_data(p: tf.ParticleHandle) -> ParticleData:
        r, theta, phi = p.sphericalPosition(particle=big_particle)
        return ParticleData(p, theta, phi)
    
    p: tf.ParticleHandle
    big_particle = Big.particle(0)
    total_force: float = magnitude * len(LeadingEdge.items())
    particle_data_list: list[ParticleData] = [get_particle_data(p) for p in LeadingEdge.items()]
    sorted_on_theta: list[ParticleData] = sorted(particle_data_list, key=lambda data: data.theta)

    previous_particle_data: ParticleData = sorted_on_theta[-1]
    before_previous_particle_data: ParticleData = sorted_on_theta[-2]
    previous_theta: float = previous_particle_data.theta - cfg.two_pi
    before_previous_theta: float = before_previous_particle_data.theta - cfg.two_pi
    particle_data: ParticleData
    for particle_data in sorted_on_theta:
        arc: float = (particle_data.theta - before_previous_theta) / 2
        mag: float = total_force * arc / cfg.two_pi
        tangent_phi = previous_particle_data.phi + math.pi / 2
        tangent_force_vec: tf.fVector3 = tfu.cartesian_from_spherical([mag, previous_theta, tangent_phi])
        previous_particle_data.phandle.force_init = tangent_force_vec.as_list()
        
        before_previous_theta = previous_theta
        previous_particle_data = particle_data
        previous_theta = particle_data.theta
