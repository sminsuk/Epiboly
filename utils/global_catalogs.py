""" Global particle and bond data. For storing stuff that is irretrievable once created

But, more generally, storing system state. So storing visibility here as well.

Maybe put this in a class?
Usage:
    Note: The original uses of this structure were to store data that was not retrievable from bonds and particles
    because of bugs in Tissue Forge. Those bugs have now been fixed. So currently these structures are not
    storing anything useful. However, this is good working infrastructure, so I'm keeping it in place for
    future use.
    
    Angles were added later, and that was also for a bug, which likely still exists in v0.1.0. (But that one
    never needed actual content, just keys, as described below.) If those anomalous Angle bonds get fixed (prevented)
    in a future release, then won't need this for Angles.
    
    Use the convenience functions below to create Bonds and Angles, and to add newly created Particles to the
    catalog; and to destroy any of those. Alternatively:
    
    Whenever creating a particle / bond / angle, add it to particles_by_id / bonds_by_id / angles_by_id, respectively
    Whenever destroying a particle / bond / angle, delete it from those dicts
    
    BondData currently empty.
    ParticleData currently empty.
    AngleData never needed any content: the idea is to identify anomalous Angle objects created due to a
        TF bug, by the fact that they are NOT in the dict. All we really need is the keys; using id as placeholder
        value. (We'll never need to look up the value.)
    
    Whenever a particle .becomes(LeadingEdge), set its visibility to True;
    Whenever a particle .becomes(Little) (interior particle), set its visibility according to the state flag here.
        (For now, change of state of the flag itself happens in module "interactive".)
"""
from typing import TypedDict

import tissue_forge as tf
import utils.tf_utils as tfu

class BondData(TypedDict, total=False):
    dummy: int

class ParticleData(TypedDict, total=False):
    dummy: int

bonds_by_id: dict[int, BondData] = {}
angles_by_id: dict[int, int] = {}
particles_by_id: dict[int, ParticleData] = {}

def create_bond(potential: tf.Potential, p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> tf.BondHandle:
    handle: tf.BondHandle = tf.Bond.create(potential, p1, p2)
    bond_values: BondData = {}
    bonds_by_id[handle.id] = bond_values
    return handle

def destroy_bond(bhandle: tf.BondHandle) -> None:
    del bonds_by_id[bhandle.id]
    bhandle.destroy()
    
def create_angle(potential: tf.Potential,
                 outer_p1: tf.ParticleHandle,
                 center_p: tf.ParticleHandle,
                 outer_p2: tf.ParticleHandle) -> tf.AngleHandle:
    handle: tf.AngleHandle = tf.Angle.create(potential, outer_p1, center_p, outer_p2)
    angles_by_id[handle.id] = handle.id
    return handle

def destroy_angle(angle_handle: tf.AngleHandle) -> None:
    del angles_by_id[angle_handle.id]
    angle_handle.destroy()
    
def add_particle(phandle: tf.ParticleHandle) -> None:
    """ 'Add' rather than 'Create' because TF routines handle the creation part"""
    particle_values: ParticleData = {}
    particles_by_id[phandle.id] = particle_values

def destroy_particle(phandle: tf.ParticleHandle) -> None:
    # If particle has any bonds, destroy them. TF would do that, but would not remove them from the catalog
    b: tf.BondHandle
    for b in tfu.bonds(phandle):
        destroy_bond(b)
        
    del particles_by_id[phandle.id]
    phandle.destroy()

def initialize_state() -> None:
    """Recreates the global catalogs based on imported data from a previous run
    
    Currently all three dicts contain no meaninful data, but we still need to create them and fill them
    with dummy values, so that when existing items are destroyed, their dict entries can be del'ed.
    
    Note that imported particle ids won't be the same as when they were exported, but we don't need
    the old ones and can simply retrieve new ones.
    """
    for bhandle in tf.BondHandle.items():
        bond_values: BondData = {}
        bonds_by_id[bhandle.id] = bond_values
        
    for handle in tf.AngleHandle.items():
        angles_by_id[handle.id] = handle.id
        
    for phandle in tf.Universe.particles:
        particle_values: ParticleData = {}
        particles_by_id[phandle.id] = particle_values
        
def clean_state() -> None:
    """Clean out all anomalous Angle bonds at once
    
    Because of the bug that required this dict to exist in the first place, we need a function to purge any
    anomalous angles that should not exist, BEFORE saving simulation state.
    """
    angle: tf.AngleHandle
    for angle in tf.AngleHandle.items():
        if angle.id not in angles_by_id:
            print(tfu.bluecolor + f"Destroying false angle (id={angle.id})" + tfu.endcolor)
            angle.destroy()

visibility_state: bool = True
