"""Handle the remodeling of the bond network as the tissue changes shape"""
import random

from epiboly_init import *
from utils import tf_utils as tfu,\
    global_catalogs as gc

import neighbors as nbrs

def _make_bond(p1: tf.ParticleHandle, p2: tf.ParticleHandle, verbose: bool = False) -> None:
    """Return a potential tailored to these 2 particles: generates no force because r0 = their current distance.
    
    min = r0 so that this also generates no force if the particles are subsequently pushed towards each other;
    only generates force if they are pulled apart
    """
    distance: float = p1.distance(p2)
    # if particles are overlapping, set r0 so they won't:
    r0: float = max(distance, Little.radius * 2)
    potential: tf.Potential = tf.Potential.harmonic(r0=r0,
                                                    k=7.0,
                                                    min=r0,
                                                    max=6
                                                    )
    handle: tf.BondHandle = tf.Bond.create(potential, p1, p2)
    bond_values: gc.BondData = {"r0": r0}
    gc.bonds_by_id[handle.id] = bond_values
    if verbose:
        print(f"Making new bond {handle.id} between particles {p1.id} and {p2.id}")
        p1.style.setColor("lightgray")  # testing
        p2.style.setColor("white")  # testing

def make_bonds(phandle: tf.ParticleHandle, verbose=False) -> int:
    # Bond to all neighbors not already bonded to
    neighbors = nbrs.get_non_bonded_neighbors(phandle)
    for neighbor in neighbors:
        _make_bond(neighbor, phandle, verbose)
    return len(neighbors)

def _break_bonds(saturation_factor: float, max_prob: float) -> None:
    """Decide which bonds will be broken, then break them
    
    saturation_factor: multiple of r0 at which probability of breaking = max_prob
    max_prob: the max value that probability of breaking ever reaches. In range [0, 1].
    """
    def breaking_probability(bhandle: tf.BondHandle) -> float:
        """Probability of breaking bond. Bond must be active!"""
        gcdict = gc.bonds_by_id
        assert bhandle.id in gcdict, "Bond data missing from global catalog!"
        bond_data: gc.BondData = gcdict[bhandle.id]
        potential: tf.Potential = bhandle.potential
        r0: float = bond_data["r0"]
        # print(f"r0 = {r0}")
        r: float = tfu.bond_distance(bhandle)
        saturation_distance: float = saturation_factor * r0
        saturation_energy: float = potential(saturation_distance)
        
        # potential(r) should match the bond's energy property, though it won't be exact:
        assert abs(bhandle.energy - potential(r)) < 0.0001, \
            f"unexpected bond energy: property = {bhandle.energy}, calculated = {potential(r)}"
    
        p: float
        if r <= r0:
            p = 0
        elif r > saturation_distance:
            p = max_prob
        else:
            p = max_prob * bhandle.energy / saturation_energy
            
        return p

    print(f"Evaluating all {len(tf.BondHandle.items())} bonds, to maybe break")
    breaking_bonds = [bhandle for bhandle in tf.BondHandle.items()
                      if bhandle.active
                      if random.random() < breaking_probability(bhandle)
                      ]
    print(f"breaking {len(breaking_bonds)} bonds: {[bhandle.id for bhandle in breaking_bonds]}")
    
    bhandle: tf.BondHandle
    for bhandle in breaking_bonds:
        del gc.bonds_by_id[bhandle.id]
        bhandle.destroy()
    
def maintain_bonds() -> None:
    total: int = 0
    for ptype in [Little, LeadingEdge]:
        for p in ptype.items():
            total += make_bonds(p, verbose=True)
    print(f"Created {total} bonds.")

    _break_bonds(saturation_factor=3, max_prob=0.001)
