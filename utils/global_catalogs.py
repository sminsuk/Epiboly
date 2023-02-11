""" Global particle and bond data. For storing stuff that is irretrievable once created

But, more generally, storing system state. So storing visibility here as well.

Maybe put this in a class?
Usage:
    For particles: leaving this up to the caller, since it only happens in a couple of places, and is more convenient.
    For Bonds and Angles: use the convenience functions below.
    Whenever creating a particle / bond / angle, add it to particles_by_id / bonds_by_id / angles_by_id, respectively
    Whenever deleting a particle / bond / angle, delete it from those dicts
    BondData allows to retrieve the r0 from the potential attached to a given bond
    AngleData doesn't even need any content: the idea is to identify anomalous Angle objects created due to a
        TF bug, by the fact that they are NOT in the dict. All we really need is the keys; using id as placeholder
        value. (We'll never need to look up the value.)
    ParticleData allows to retrieve ParticleHandle from an id.
    
    Whenever a particle .becomes(LeadingEdge), set its visibility to True;
    Whenever a particle .becomes(Little) (interior particle), set its visibility according to the state flag here.
        (For now, change of state of the flag itself happens in module "interactive".)
    
future: storing r0 is not supposed to be necessary, because it should be retrievable from potential object.
Currently broken in harmonic, maybe others. May be able to do without this in future release.

future: if those anomalous Angle bonds get fixed (prevented) in a future release, then won't need this.

future: storing particleHandles should not be necessary, as it should be possible to retrieve from particle.id.
Currently you can only do that if you also have the particle.type_id. Should be fixed in a future release.
"""
from typing import TypedDict

import tissue_forge as tf

class BondData(TypedDict):
    r0: float

class ParticleData(TypedDict):
    handle: tf.ParticleHandle

bonds_by_id: dict[int, BondData] = {}
angles_by_id: dict[int, int] = {}
particles_by_id: dict[int, ParticleData] = {}

def create_bond(potential: tf.Potential, p1: tf.ParticleHandle, p2: tf.ParticleHandle, r0: float) -> tf.BondHandle:
    assert p1.id != p2.id, f"Bonding particle {p1.id} to itself!"
    handle: tf.BondHandle = tf.Bond.create(potential, p1, p2)
    bond_values: BondData = {"r0": r0}
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
    
visibility_state: bool = True
