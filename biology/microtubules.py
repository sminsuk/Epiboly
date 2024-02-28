"""Simulate the towing of the margin toward the vegetal pole by microtubule arrays in the yolk cells"""
from dataclasses import dataclass
import math

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import utils.tf_utils as tfu
import utils.epiboly_utils as epu

force_enabled: bool = True
_force_per_unit_length: float = 0.0

def initialize_tangent_forces() -> None:
    """Establish the linear relationship between total downward force on the leading edge, and its circumference
    
    This should be done after initial setup and equilibration but before any further timesteps, so that it's
    based on the initial circumference before any epiboly progression. (That's important so that the right value
    is used in the balanced-force control.) After that, never change it, but apply it to the gradually changing
    circumference in order to determine the amount of force to use.
    
    With the APPROACH_0 algorithm, the ratio of force per unit length of leading edge stays constant over time,
    which was expected to also result in constant speed of epiboly. (Not what it actually does, though!)
    """
    global _force_per_unit_length, force_enabled
    
    force_enabled = True
    external_force: float = 0 if cfg.run_balanced_force_control else cfg.external_force
    total_force_start: float = cfg.yolk_cortical_tension + external_force
    _force_per_unit_length = total_force_start / epu.leading_edge_circumference()

def remove_tangent_forces() -> None:
    """Call this once to remove tangent forces from all particles, after turning off the updates."""
    global force_enabled
    
    force_enabled = False
    for p in g.LeadingEdge.items():
        p.force_init = [0, 0, 0]
    print("Tangent forces removed")

def apply_even_tangent_forces() -> None:
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
    particle_data_list: list[ParticleData] = [get_particle_data(p) for p in g.LeadingEdge.items()]
    sorted_on_theta: list[ParticleData] = sorted(particle_data_list, key=lambda data: data.theta)

    # First loop: collect (and sum) all the weights
    previous_particle_data: ParticleData = sorted_on_theta[-1]
    before_previous_particle_data: ParticleData = sorted_on_theta[-2]
    previous_theta: float = previous_particle_data.theta - 2 * math.pi
    before_previous_theta: float = before_previous_particle_data.theta - 2 * math.pi
    particle_data: ParticleData
    weight_total: float = 0
    for particle_data in sorted_on_theta:
        # As we traverse the sorted list, we gather enough data to calculate the weight of a particle
        # when we reach the particle after that. So we are calculating the weight for previous_particle.
        previous_p_relative_cell_width: float = particle_data.theta - before_previous_theta
        previous_p_relative_distance_to_pole: float = math.pi - previous_particle_data.phi
        cell_relative_weight: float = previous_p_relative_distance_to_pole * previous_p_relative_cell_width
        previous_particle_data.weight = cell_relative_weight
        weight_total += cell_relative_weight
        
        before_previous_theta = previous_theta
        previous_particle_data = particle_data
        previous_theta = particle_data.theta
        
    # Second loop: now that we have all the weights and the total, we can calculate the forces
    leading_edge_circumference: float = epu.leading_edge_circumference()
    for particle_data in sorted_on_theta:
        normalized_cell_weight: float = particle_data.weight / weight_total
        effective_cell_width: float = leading_edge_circumference * normalized_cell_weight
        mag: float = _force_per_unit_length * effective_cell_width
        tangent_phi = particle_data.phi + math.pi / 2
        tangent_force_vec: tf.fVector3 = tfu.cartesian_from_spherical([mag, particle_data.theta, tangent_phi])
        
        # The assignment runs into the copy-constructor bug! So change to plain list
        particle_data.phandle.force_init = tangent_force_vec.as_list()

def current_total_force() -> float:
    """Calculate the total force to apply to the leading edge
    
    This function is used by the plotting module, which plots even when forces are disabled.
    Therefore need to return a value (0) in that case, so that the value is displayed correctly.
    """
    return _force_per_unit_length * epu.leading_edge_circumference() if force_enabled else 0
    
def get_state() -> dict:
    """generate state to be saved to disk"""
    return {"force_enabled": force_enabled,
            "force_per_unit_length": _force_per_unit_length}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global _force_per_unit_length, force_enabled
    force_enabled = d["force_enabled"]
    _force_per_unit_length = d["force_per_unit_length"]
