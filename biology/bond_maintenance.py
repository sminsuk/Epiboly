"""Handle the remodeling of the bond network as the tissue changes shape"""
import random

from epiboly_init import *
from utils import tf_utils as tfu,\
    global_catalogs as gc

import neighbors as nbrs

def _maintain_bond(phandle: tf.ParticleHandle) -> None:
    # print(f"Bond maintenance invoked for particle {phandle.id} of type {phandle.type().name}")
    # Any bonds here, to break?
    p1 = phandle
    bonded_neighbors = p1.getBondedNeighbors()
    if not bonded_neighbors:
        # empty, no bonds here
        return
    # particle's longest bond, because it's the most likely to break? Simplistic. Keep this for now, for
    # testing reasons: increase likelihood that I see something happening at all. But correct way to do it
    # is to select one of them at random, and then break according to energy considerations.
    p2 = max(bonded_neighbors, key=lambda other_p: p1.distance(other_p))
    bond = tfu.bond_between(p1, p2)
    # print("p1.id, p2.id, bond.parts =", p1.id, p2.id, bond.parts)
    # for testing, have a 10% chance of changing color
    if random.random() < 0.1:
        p1.style.setColor("lightgray")  # testing
        p2.style.setColor("white")  # testing
    
    # Now we have two particles and the bond between. Search nearby for particles (not necessarily bonded neighbors
    # of these two, or immediate neighbors at all), that are able to make any new bonds.
    # For simplicity, just use p1 as the search center. Was going to use both and combine them, but maybe not necessary?
    vicinity = nbrs.find_neighbors(p1, distance_factor=3.0)
    # vicinity += nbrs.find_neighbors(p2, distance_factor=3.0)
    # I waiver on whether to do this: if we break this bond, exclude these two particles from being involved in the new one:
    if p2 in vicinity:
        vicinity.remove(p2)
    
    ######## This seems wrong, though. Maybe we only want to find a single pair?
    ######## Actually that seems wrong too; if we always break one and make one, then it enforces that the
    # total number of bonds in the system must be static. I highly doubt it! Rethink this. Maybe completely
    # decouple breaking from making? Or maybe remake all bonds each time from scratch?
    # Or what about this: any qualified, unbonded particles nearby, just bond to them; any bonds out of range,
    # break them? The latter can be automatic for now. This is at least based on energy (because energy<==>distance).
    # Stochasticity from movements? Or maybe don't do every one, every time? The problem: in a stretching tissue,
    # bonds are getting on average longer, and it would be horrible. It will be necessary to increase bond length!
    bondable_pairs = []
    # Since vicinity is sorted by distance, we're finding the closest particle capable of making a new bond
    # NOTE this is ungodly slow. Gotta fix this.
    #
    # print(f"vicinity of p1 has {len(vicinity)} particles, each with the following number of bondable neighbors:")
    # for particle in vicinity:
    #     # find potential bonding partners (close neighbors not already bonded to)
    #     neighbors = nbrs.get_non_bonded_neighbors(particle)
    #     print(f"{len(neighbors)} bondable neighbors")
    #     # Several insights from running this print statement: 1) a lot of wasted time, because there are so
    #     # few bondable neighbors, so this is a terribly inefficient way to find them; 2) vicinity is large,
    #     # with typically 14-18 particles each; 3) So this results in usually *no* bondable pairs in the vicinity,
    #     # occasionally 1 or 2 or 3. Also, 4) consider the algorithm. This is every time step, 700 particles
    #     # request neighbors and sort; * 15 resulting vicinity particles; * 6 of their neighbors loop to find
    #     # non-bonded ones. 700 * 15 * 6. So each vicinity is 90 times through the two innermost loops, to
    #     # get easily less than one bondable pair per vicinity. This is ridiculous. So need to kill this
    #     # algorithm not only because it's probably wrong in terms of logic, but also because it sucks!
    #     for neighbor in neighbors:
    #         bondable_pairs.append((particle, neighbor))
    # #             tf.Bond.create(small_small_attraction, neighbor, particle)
    #
    # print(":", end="", flush=True)
    # # print(f"yielding {len(bondable_pairs)} bondable pairs")
    #
    # Did we find any at all?
    if not bondable_pairs:
        # if we can't make any bonds, then we can't break any, either
        return
    
    # Next: compare energies and make/break bonds. But first, rethink...

def _make_bond(p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> None:
    """Return a potential tailored to these 2 particles: generates no force because r0 = their current distance.
    
    min = r0 so that this also generates no force if the particles are subsequently pushed towards each other;
    only generates force if they are pulled apart
    """
    distance: float = p1.distance(p2)
    assert distance > 0.2, f"Making bond with r0 = {distance}"
    potential: tf.Potential = tf.Potential.harmonic(r0=distance,
                                                    k=7.0,
                                                    min=distance,
                                                    max=6
                                                    )
    handle: tf.BondHandle = tf.Bond.create(potential, p1, p2)
    bond_values: gc.BondData = {"r0": distance}
    gc.bonds_by_id[handle.id] = bond_values

def make_bonds(phandle: tf.ParticleHandle, verbose=False) -> int:
    # Bond to all neighbors not already bonded to
    neighbors = nbrs.get_non_bonded_neighbors(phandle)
    for neighbor in neighbors:
        if verbose:
            print(f"Making new bond between particles {neighbor.id} and {phandle.id}")
            neighbor.style.setColor("lightgray")  # testing
            phandle.style.setColor("white")  # testing
        _make_bond(neighbor, phandle)
    return len(neighbors)

def _break_bonds(saturation_factor: int) -> None:
    """Decide which bonds will be broken, then break them
    
    saturation_factor: multiple of r0 at which probability of breaking = 1
    """
    def breaking_probability(bhandle: tf.BondHandle) -> float:
        potential: tf.Potential = bhandle.potential
        assert hasattr(potential, "r0"), f"Potential {potential} has no r0 attribute!"
        r0: float = potential.r0
        print(f"r0 = {r0}")
        # assert r0 > 0.2, f"Found potential with r0 = {r0}"
        r: float = tfu.bond_distance(bhandle)
        saturation_distance: float = saturation_factor * r0
        # ###### ToDo: This fails because r0 is being read as 0.0, hence every bond breaks!
        # ###### Also sometimes potential is None, and throws error when trying to access r0.
        # ###### Furthermore, pausing the simulator reproducibly causes that to happen!
        # ###### ToDo: Need to implement the speed-up approach, as well
        saturation_energy: float = potential(saturation_distance)
        
        # potential(r) should match the bond's energy property, though it won't be exact:
        assert abs(bhandle.energy - potential(r)) < 0.0001, \
            f"unexpected bond energy: property = {bhandle.energy}, calculated = {potential(r)}"
    
        p: float
        if r <= r0:
            p = 0
        elif r > saturation_distance:
            p = 1
        else:
            p = bhandle.energy / saturation_energy
            
        return p

    print(f"Evaluating all {len(tf.BondHandle.items())} bonds, to maybe break")
    breaking_bonds = [bhandle for bhandle in tf.BondHandle.items()
                      if random.random() < breaking_probability(bhandle)
                      ]
    initial_count = len(breaking_bonds)
    print(f"breaking {initial_count} bonds")
    
    bhandle: tf.BondHandle
    for bhandle in breaking_bonds:
        del gc.bonds_by_id[bhandle.id]
        bhandle.destroy()
        # Okay to destroy items while iterating over the list?
    final_count = len(breaking_bonds)
    
    assert initial_count == final_count, \
        f"len(list) changed during iteration, from {initial_count} to {final_count}"
    # i.e. if length changes, then shouldn't destroy while iterating (use pop instead!)

def maintain_bonds_old_version(ptypes: list[tf.ParticleType]) -> None:
    """Worrying: this seems really slow. Might need, instead of visiting every particle and deciding whether
    to process it, decide in advance how many to process, and only visit those, randomly selected of course.
    Could also go less often than every dt."""
    for ptype in ptypes:
        particles = [p for p in ptype.items()]
        for p in particles:
            _maintain_bond(p)

def maintain_bonds() -> None:
    total: int = 0
    for ptype in [Little, LeadingEdge]:
        for p in ptype.items():
            total += make_bonds(p, verbose=True)
    print(f"Created {total} bonds.")

    _break_bonds(saturation_factor=3)
