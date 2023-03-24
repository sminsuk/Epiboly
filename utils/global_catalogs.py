""" Global particle and bond data. For storing stuff that is irretrievable once created

But, more generally, storing system state. So storing visibility here as well.

Maybe put this in a class?
Usage:
    Note: The original uses of this structure were to store data that was not retrievable from bonds and particles
    because of bugs in Tissue Forge. Those bugs have now been fixed. So currently these structures are not
    storing anything useful. However, this is good working infrastructure, so I'm keeping it in place for
    future use.
    
    For particles: leaving this up to the caller, since it only happens in a couple of places, and is more convenient.
    For Bonds and Angles: use the convenience functions below.
    Whenever creating a particle / bond / angle, add it to particles_by_id / bonds_by_id / angles_by_id, respectively
    Whenever deleting a particle / bond / angle, delete it from those dicts
    BondData currently empty.
    AngleData doesn't even need any content: the idea is to identify anomalous Angle objects created due to a
        TF bug, by the fact that they are NOT in the dict. All we really need is the keys; using id as placeholder
        value. (We'll never need to look up the value.)
    ParticleData allows to retrieve ParticleHandle from an id.
    
    Whenever a particle .becomes(LeadingEdge), set its visibility to True;
    Whenever a particle .becomes(Little) (interior particle), set its visibility according to the state flag here.
        (For now, change of state of the flag itself happens in module "interactive".)
    
future: if those anomalous Angle bonds get fixed (prevented) in a future release, then won't need this for Angles.

future: storing particleHandles should not be necessary, as it should be possible to retrieve from particle.id.
Currently you can only do that if you also have the particle.type_id. Should be fixed in a future release.
"""
from typing import TypedDict

import tissue_forge as tf
import tf_utils as tfu

class BondData(TypedDict, total=False):
    dummy: int

class ParticleData(TypedDict):
    handle: tf.ParticleHandle

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
    
def initialize_state() -> None:
    """Recreates the global catalogs based on imported data from a previous run
    
    bonds_by_id: this only contains r0 for all the bonds, which CANNOT be recreated after an export/import,
    because in v. 0.0.1, r0 can't be read out of the bonds. If absolutely necessary, I could export these values
    and then re-import them, but it's not worth it right now because it's only used for the relaxation feature,
    which I'm currently not using. So skip it for now, and if I turn relaxation on, this will fail.
    
    However, still need to create a dict full of keys with dummy values, so that when existing bonds are destroyed,
    their entry can be destroyed too.
    
    angles_by_id: similarly, we need the keys so that angles can be destroyed. But also, see clean_state()
    
    particles_by_id: this needs to be reconstituted based on the imported particles. Note that their imported ids won't
    be the same as when they were exported, nor will their ParticleHandles, but we don't need the old ones and
    can simply retrieve new ones.
    """
    for bhandle in tf.BondHandle.items():
        bond_values: BondData = {}
        bonds_by_id[bhandle.id] = bond_values
        
    for handle in tf.AngleHandle.items():
        angles_by_id[handle.id] = handle.id
        
    for phandle in tf.Universe.particles:
        particle_values: ParticleData = {"handle": phandle}
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
