"""Simulate the towing of the margin toward the vegetal pole by microtubule arrays in the yolk cells"""
from dataclasses import dataclass
import math

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import utils.tf_utils as tfu
import utils.epiboly_utils as epu

force_enabled: bool = True

# Linear relationship between circumference and force, y = mx + b:
_m: float = 0.0
_b: float = 0.0

def initialize_tangent_forces() -> None:
    """Establish the linear relationship between total downward force on the leading edge, and its circumference
    
    This should be done after initial setup and equilibration but before any further timesteps, so that it's
    based on the initial circumference before any epiboly progression. (That's important so that the right value
    is used in the balanced-force control.) After that, never change it, but apply it to the gradually changing
    circumference in order to determine the amount of force to use.
    
    With the APPROACH_0 algorithm, the ratio of force per unit length of leading edge stays constant over time,
    which was expected to also result in constant speed of epiboly. (Not what it actually does, though!)
    """
    global _m, _b, force_enabled
    
    force_enabled = True
    external_force: int = 0 if cfg.run_balanced_force_control else cfg.external_force
    total_force_start: float = cfg.yolk_cortical_tension + external_force
    initial_force_circumference_ratio: float = total_force_start / epu.leading_edge_circumference()
    force_algorithm: cfg.ForceAlgorithm = cfg.force_algorithm
    
    if force_algorithm == cfg.ForceAlgorithm.CONSTANT:
        # Can fold this into the same formula, but for now I'm retaining the enums, so just leave it
        _m = 0
        _b = total_force_start
    elif force_algorithm == cfg.ForceAlgorithm.LINEAR:
        _m = (1 - cfg.force_target_fraction) * initial_force_circumference_ratio
        _b = cfg.force_target_fraction * total_force_start

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
        mag: float = current_total_force() * particle_data.weight / weight_total
        tangent_phi = particle_data.phi + math.pi / 2
        tangent_force_vec: tf.fVector3 = tfu.cartesian_from_spherical([mag, particle_data.theta, tangent_phi])
        
        # The assignment runs into the copy-constructor bug! So change to plain list
        particle_data.phandle.force_init = tangent_force_vec.as_list()

def current_total_force() -> float:
    """Calculate the total force to apply to the leading edge
    
    This function has a dual use. When called from within this module, the return value is interpreted to indicate
    how much force SHOULD be applied, and this will result in setting the forces on the particles. This
    only happens when forces are enabled.
    
    But when called from outside this module (i.e. from plotting module), the return value is interpreted to
    indicate hou much force IS being applied, and the reported value is displayed.
    Therefore still need to return a value (0) when forces are disabled, so that the value is displayed correctly.
    """
    # x = circumference, y = force = mx + b
    return (_m * epu.leading_edge_circumference() + _b) if force_enabled else 0
    
def get_state() -> dict:
    """generate state to be saved to disk"""
    return {"force_enabled": force_enabled,
            "m": _m,
            "b": _b}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global _m, _b, force_enabled
    force_enabled = d["force_enabled"]
    _m = d["m"]
    _b = d["b"]
