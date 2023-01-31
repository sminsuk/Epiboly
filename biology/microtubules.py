"""Simulate the towing of the margin toward the vegetal pole by microtubule arrays in the yolk cells"""
from dataclasses import dataclass
import math

import tissue_forge as tf
from epiboly_init import LeadingEdge
import config as cfg
from utils import tf_utils as tfu,\
    epiboly_utils as epu

def remove_tangent_forces() -> None:
    """Call this once to remove tangent forces from all particles, after turning off the updates."""
    for p in LeadingEdge.items():
        p.force_init = [0, 0, 0]
    print("Tangent forces removed")

def apply_even_tangent_forces(total_force: int) -> None:
    """Note that once this has run, turning it off does not remove existing forces. Use remove_tangent_forces().
    
    To apply force evenly, must take into account not only the variation in density around the marginal ring, but
    also the effect of radius: if density is measured by delta theta between neighboring edge cells, then a region of
    cells will seem to get less "dense" as it approaches the vegetal pole because the theta between them will increase.
    """
    @dataclass
    class ParticleData:
        phandle: tf.ParticleHandle
        theta: float
        phi: float
        weight: float = 0
        
    def get_particle_data(p: tf.ParticleHandle) -> ParticleData:
        theta, phi = epu.embryo_coords(p)
        return ParticleData(p, theta, phi)
    
    p: tf.ParticleHandle
    particle_data_list: list[ParticleData] = [get_particle_data(p) for p in LeadingEdge.items()]
    sorted_on_theta: list[ParticleData] = sorted(particle_data_list, key=lambda data: data.theta)

    # First loop: collect (and sum) all the weights
    previous_particle_data: ParticleData = sorted_on_theta[-1]
    before_previous_particle_data: ParticleData = sorted_on_theta[-2]
    previous_theta: float = previous_particle_data.theta - cfg.two_pi
    before_previous_theta: float = before_previous_particle_data.theta - cfg.two_pi
    particle_data: ParticleData
    weight_total: float = 0
    for particle_data in sorted_on_theta:
        arc: float = (particle_data.theta - before_previous_theta)
        # The relevant "radius" of the leading edge "circle" is actually the distance (or arc) from the vegetal pole:
        radius: float = math.pi - previous_particle_data.phi
        weight: float = radius * arc
        previous_particle_data.weight = weight
        weight_total += weight
        
        before_previous_theta = previous_theta
        previous_particle_data = particle_data
        previous_theta = particle_data.theta
        
    # Second loop: now that we have all the weights and the total, we can calculate the forces
    for particle_data in sorted_on_theta:
        mag: float = total_force * particle_data.weight / weight_total
        tangent_phi = particle_data.phi + cfg.pi_over_2
        tangent_force_vec: tf.fVector3 = tfu.cartesian_from_spherical([mag, particle_data.theta, tangent_phi])
        
        # The assignment runs into the copy-constructor bug! So change to plain list
        particle_data.phandle.force_init = tangent_force_vec.as_list()

