"""Set up dynamical changes to the system.

Creates events that are run every simulation step, completely separate from the
execution queue (module exec_queue). By default, they do nothing, but let you turn
actions on or off, either interactively, or from within the execution queue, as
appropriate.

The on_time event is for actions that involve iterating over multiple particles at once.
The on_particle events are for actions that involve a single particle.
on_particle events only take a *single* ParticleType. So if you want to run an event on
more than one type, you have to turn on that event for each one.
"""

import math
import random

from epiboly_init import *
import sharon_utils as su

import neighbors as nbrs

_set_tangent_allowed = False
_set_bond_maint_allowed = False


def set_tangent_forces():
    # guard against calling from Jupyter when running all cells. In ipython, call twice the first time
    global _set_tangent_allowed

    global _tangent_forces_on
    if _set_tangent_allowed:
        _tangent_forces_on = True
        print("Tangent forces enabled")
    else:
        print("First invoke, outside Jupyter call twice")
        _set_tangent_allowed = True


def unset_tangent_forces():
    global _tangent_forces_on
    _tangent_forces_on = False

    for p in LeadingEdge.items():
        p.force_init = [0, 0, 0]
    print("Tangent forces disabled")


def set_bond_maintenance(types):
    """Turn bond maintenance on, for the ParticleTypes listed in the dict
    
    types: dictionary of ParticleType objects as keys, bool as values
    Note: internal dict keys will be the names of the ParticleType objects, not the objects themselves
    """
    # guard against calling from Jupyter when running all cells. In ipython, call twice the first time
    global _set_bond_maint_allowed

    global _bond_maintenance
    if _set_bond_maint_allowed:
        # 
        _bond_maintenance = {ptype.name: types[ptype] for ptype in types}
        print("Bond maintenance enabled for", [typename for typename in _bond_maintenance
                                               if _bond_maintenance[typename]])
    else:
        print("First invoke, outside Jupyter call twice")
        _set_bond_maint_allowed = True


def unset_bond_maintenance():
    """Convenience method. Could also call the setter with False/False."""
    global _bond_maintenance
    _bond_maintenance = {Little.name: False, LeadingEdge.name: False}
    print("Bond maintenance disabled")


######## Private ########
_tangent_forces_on = False
_bond_maintenance = {Little.name: False, LeadingEdge.name: False}


def _update_tangent_forces():
    """Still to do! This needs a stopping criterion. Based on angle of phi? Or distance of particle
    from the vegetal pole? Needs criterion both for the individual particle, and for when all of them
    have arrived.
    """
    # For now, add a vector of fixed magnitude, in the tangent direction
    magnitude = 5
    big_particle = Big.particle(0)
    for p in LeadingEdge.items():
        r, theta, phi = p.sphericalPosition(particle=big_particle)
        tangent_phi = phi + math.pi / 2
        tangent_unit_vec = su.cartesian_from_spherical([1.0, theta, tangent_phi])

        # product is an fVector3, and the assignment runs into the copy-constructor bug! So change to plain list
        p.force_init = (tangent_unit_vec * magnitude).as_list()


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


def _maintain_bonds(particle_type: tf.ParticleType) -> None:
    """Worrying: this seems really slow. Might need, instead of visiting every particle and deciding whether
    to process it, decide in advance how many to process, and only visit those, randomly selected of course.
    Could also go less often than every dt."""
    particles = [p for p in particle_type.items()]
    for p in particles:
        _maintain_bond(p)


def _master_event(evt):
    """This is intended to be run every simulation step. For now, ignoring evt and letting it run forever."""
    global _tangent_forces_on, _bond_maintenance

    if _tangent_forces_on:
        try:
            _update_tangent_forces()
        except Exception as e:
            su.exception_handler(e, _update_tangent_forces.__name__)

    # For now, hard-coding that this acts on the two types, Little and LeadingEdge.
    # This will be much cleaner if I improved the interface for this as planned.
    # Then func name and list of types will be passed; no need for typename keys to enable.
    if _bond_maintenance[Little.name]:
        try:
            _maintain_bonds(Little)
        except Exception as e:
            su.exception_handler(e, _master_event.__name__)
    if _bond_maintenance[LeadingEdge.name]:
        try:
            _maintain_bonds(LeadingEdge)
        except Exception as e:
            su.exception_handler(e, _master_event.__name__)


#     _bond_maintenance = {Little.name: False, LeadingEdge.name: False} # for testing, do this just once!

# def _master_particle_event(evt):
#     """This runs every simulation step. Ignore evt and let it run forever, but pass the particle to the invoked actions.
#
#     Use for actions that act on a single particle.
#     evt contains the particle information.
#     For now, hard-coding that this acts on the two types, Little and LeadingEdge. I'll probably generalize
#     this, and set up a way to launch events for any generic set of ParticleTypes.
#     Also for now, using same master event for all types. If their behaviors get very divergent, might end up
#     being cleaner to have one master_particle_event for each type. Alternatively, different actions for each type.
#     """
#     global _bond_maintenance
#     if _bond_maintenance[evt.targetType.name]:
#         try:
#             _maintain_bonds(evt.targetParticle)
#         except Exception as e:
#             su.exception_handler(e, _master_particle_event.__name__)


# This will happen as soon as it's imported (after the tf.init)
tf.event.on_time(period=tf.Universe.dt, invoke_method=lambda event: _master_event(event))

# These will happen as soon as they're imported (after the tf.init)
# tf.event.on_particletime(period=10 * tf.Universe.dt, invoke_method=_master_particle_event, ptype=Little)
# tf.event.on_particletime(period=10 * tf.Universe.dt, invoke_method=_master_particle_event, ptype=LeadingEdge)
