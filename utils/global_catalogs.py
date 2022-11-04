""" Global particle and bond data. For storing stuff that is irretrievable once created

Maybe put this in a class?
Usage:
    For particles: leaving this up to the caller, since it only happens in a couple of places, and is more convenient.
    For bonds: use the convenience functions below.
    Whenever creating a particle / bond, add it to particles_by_id / bonds_by_id, respectively
    Whenever deleting a particle / bond, delete it from those dicts
    BondData allows to retrieve the r0 from the potential attached to a given bond
    ParticleData allows to retrieve ParticleHandle from an id.
    
future: storing r0 is not supposed to be necessary, because it should be retrievable from potential object.
Currently broken in harmonic, maybe others. May be able to do without this in future release.

future: storing particleHandles should not be necessary, as it should be possible to retrieve from particle.id.
Currently you can only do that if you also have the particle.type_id. Should be fixed in a future release.
"""
from typing import TypedDict

import tissue_forge as tf

class BondData(TypedDict):
    r0: float

class ParticleData(TypedDict):
    handle: tf.ParticleHandle
    blacklisted_ids: dict[int, float]   # [other id: expiration]

bonds_by_id: dict[int, BondData] = {}
particles_by_id: dict[int, ParticleData] = {}

def make_bond(potential: tf.Potential, p1: tf.ParticleHandle, p2: tf.ParticleHandle, r0: float) -> tf.BondHandle:
    handle: tf.BondHandle = tf.Bond.create(potential, p1, p2)
    bond_values: BondData = {"r0": r0}
    bonds_by_id[handle.id] = bond_values
    return handle

def break_bond(bhandle: tf.BondHandle) -> None:
    del bonds_by_id[bhandle.id]
    bhandle.destroy()
