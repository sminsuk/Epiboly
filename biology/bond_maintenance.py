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

def _break_or_relax(saturation_factor: float, max_prob: float, viscosity: float) -> None:
    """Decide which bonds will be broken, then break them. Relax those that survive.
    
    saturation_factor: multiple of r0 at which probability of breaking = max_prob
    max_prob: the max value that probability of breaking ever reaches. In range [0, 1].
    viscosity: how much relaxation per timestep. In range [0, 1].
    """
    def breaking_probability(bhandle: tf.BondHandle, r0: float, r: float) -> float:
        """Probability of breaking bond. Bond must be active!"""
        potential: tf.Potential = bhandle.potential
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
    
    def relax_bond(bhandle: tf.BondHandle, r0: float, r: float, viscosity: float) -> None:
        """Relaxing a bond means to partially reduce the energy (hence the generated force) by changing
        the r0 toward the current r.

        viscosity: a value in the range [0, 1].
            v = 0 is completely elastic (no change to r0, ever; so if a force is applied that stretches the bond, and
                then released, the bond will recoil and tend to shrink back to its original length)
            v = 1 is completely plastic (r0 instantaneously takes the value of r; so if a force is applied that
                stretches the bond, and then released, there will be no recoil at all)
            0 < v < 1 means r0 will change each timestep, but only by that fraction of the difference (r-r0). So bonds
                will always be under some tension, but the longer a bond remains stretched, the less recoil there will
                be if the force is released.
        """
        pass
    
    assert 0 <= max_prob <= 1, "max_prob out of bounds"
    assert 0 <= viscosity <= 1, "viscosity out of bounds"

    breaking_bonds: list[tf.BondHandle] = []
    bhandle: tf.BondHandle
    gcdict: dict[int, gc.BondData] = gc.bonds_by_id

    print(f"Evaluating all {len(tf.BondHandle.items())} bonds, to either break, or relax")
    for bhandle in tf.BondHandle.items():
        if bhandle.active:
            assert bhandle.id in gcdict, "Bond data missing from global catalog!"
            bond_data: gc.BondData = gcdict[bhandle.id]
            potential: tf.Potential = bhandle.potential
            r0: float = bond_data["r0"]
            # print(f"r0 = {r0}")
            r: float = tfu.bond_distance(bhandle)
            
            if random.random() < breaking_probability(bhandle, r0, r):
                breaking_bonds.append(bhandle)
            else:
                relax_bond(bhandle, r0, r, viscosity)

    print(f"breaking {len(breaking_bonds)} bonds: {[bhandle.id for bhandle in breaking_bonds]}")
    for bhandle in breaking_bonds:
        del gcdict[bhandle.id]
        bhandle.destroy()
    
def maintain_bonds() -> None:
    total: int = 0
    for ptype in [Little, LeadingEdge]:
        for p in ptype.items():
            total += make_bonds(p, verbose=True)
    print(f"Created {total} bonds.")

    _break_or_relax(saturation_factor=3, max_prob=0.001, viscosity=0)
