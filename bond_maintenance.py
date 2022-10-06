"""Handle the remodeling of the bond network as the tissue changes shape"""
import random

from epiboly_init import *
import sharon_utils as su

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
    bond = su.bond_between(p1, p2)
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

def maintain_bonds(ptypes: list[tf.ParticleType]) -> None:
    """Worrying: this seems really slow. Might need, instead of visiting every particle and deciding whether
    to process it, decide in advance how many to process, and only visit those, randomly selected of course.
    Could also go less often than every dt."""
    for ptype in ptypes:
        particles = [p for p in ptype.items()]
        for p in particles:
            _maintain_bond(p)
