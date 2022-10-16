""" Global particle and bond data. For storing stuff that is irretrievable once created

Maybe put this in a class?
Usage: for now, leaving it up to the caller, since these things only happen in a couple of places:
    Whenever creating a particle / bond, add it to particles_by_id / bonds_by_id, respectively
    Whenever deleting a particle / bond, delete it from those dicts
    BondData allows to retrieve the r0 from the potential attached to a given bond
    ParticleData allows to retrieve ParticleHandle from an id. (This may not work. May have to
        store actual Particles instead.)
"""
from typing import TypedDict

import tissue_forge as tf

class BondData(TypedDict):
    potential: tf.Potential
    r0: float

class ParticleData(TypedDict):
    handle: tf.ParticleHandle

bonds_by_id: dict[int, BondData] = {}
particles_by_id: dict[int, ParticleData] = {}
