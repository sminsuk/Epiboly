"""bond_maintenance.py - Handle the remodeling of the bond network as the tissue changes shape"""
import math
import random
from statistics import fmean
import time

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
from utils import tf_utils as tfu,\
    epiboly_utils as epu,\
    global_catalogs as gc

import neighbors as nbrs

def is_edge_bond(p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> bool:
    return p1.type_id == p2.type_id == g.LeadingEdge.id

def make_bond(p1: tf.ParticleHandle, p2: tf.ParticleHandle, verbose: bool = False) -> None:
    """Return a potential tailored to these 2 particles
    
    [generates no force because r0 = their current distance.]
    ^^^ (This is not true anymore, but I may change my mind on it yet again)
    
    Accepts the TF default min (half of r0). Note that repulsion due to the bond, and repulsion due to
    type-based repulsive potential, may either overlap (both be active at certain distances) or have a gap
    (neither be active at certain intermediate distances), depending on how r0 of the bond has changed over
    the course of the simulation.
    
    Also note that this allows overlap if particles happen to be very close, but that should be minimal if
    particles are well equilibrated before making the initial bonds, and unlikely for bonds created later,
    since particles should bond before getting that close.
    """
    k: float = cfg.harmonic_edge_spring_constant if is_edge_bond(p1, p2) else cfg.harmonic_spring_constant
    
    # r0: float = p1.distance(p2)
    r0: float = 2 * g.Little.radius
    potential: tf.Potential = tf.Potential.harmonic(r0=r0,
                                                    k=k,
                                                    max=cfg.max_potential_cutoff
                                                    )
    handle: tf.BondHandle = gc.create_bond(potential, p1, p2)

    if verbose:
        distance: float = p1.distance(p2)
        print(f"Making new bond {handle.id} between particles {p1.id} and {p2.id},",
              f"distance = (radius * {distance/g.Little.radius})")
        # p1.style.color = tfu.gray   # testing
        # p2.style.color = tfu.white  # testing

def make_all_bonds(phandle: tf.ParticleHandle, verbose=False) -> None:
    # Bond to all neighbors not already bonded to
    # Except ToDo: This needs to take into account cfg.max_edge_neighbor_count
    existing_neighbor_count: int = len(phandle.bonded_neighbors)
    additional_neighbors_needed: int = max(cfg.min_neighbor_count - existing_neighbor_count, 0)
    neighbors: list[tf.ParticleHandle]
    neighbors = nbrs.get_nearest_non_bonded_neighbors(phandle,
                                                      min_neighbors=additional_neighbors_needed,
                                                      min_distance=cfg.min_neighbor_initial_distance_factor)
    for neighbor in neighbors:
        make_bond(neighbor, phandle, verbose)
        
    assert len(phandle.bonded_neighbors) >= cfg.min_neighbor_count, \
        "Failed particle bonding: particle can't find enough nearby neighbors to bond to."

def harmonic_angle_equilibrium_value() -> float:
    """A function rather than just a config constant because it depends on the number of particles in the ring"""
    # Equilibrium angle might look like π from within the plane of the leading edge, but the actual angle is
    # different. And, it changes if the number of leading edge particles changes. Hopefully it won't need to be
    # dynamically updated to be that precise. If the number of particles changes, they'll "try" to reach a target angle
    # that is not quite right, but will be opposed by the same force acting on the neighbor particles, so hopefully
    # it all balances out. (For the same reason, π would probably also work, but this value is closer to the real one.)
    return math.pi - (2 * math.pi / len(g.LeadingEdge.items()))

def test_ring_is_fubar():
    """For debugging. Set breakpoints at the indicated locations. Stop there and examine these values."""
    particles: list[tf.ParticleHandle] = [p for p in g.LeadingEdge.items()]
    neighbor_lists: list[tf.ParticleList] = [nbrs.getBondedNeighbors(p) for p in particles]
    leading_edge_neighbor_lists: list[list[tf.ParticleHandle]] = (
            [[n for n in neighbor_list if n.type_id == g.LeadingEdge.id]
             for neighbor_list in neighbor_lists])
    leading_edge_counts: list[int] = [len(neighbor_list) for neighbor_list in leading_edge_neighbor_lists]
    neighbor_fubar: list[bool] = [length != 2 for length in leading_edge_counts]
    if any(neighbor_fubar):
        # At least one LeadingEdge particle doesn't have exactly two LeadingEdge neighbors, as it should
        break_point = 0
    if not cfg.angle_bonds_enabled:
        return
    angle_lists: list[list[tf.AngleHandle]] = [p.angles for p in particles]
    angle_counts: list[int] = [len(angle_list) for angle_list in angle_lists]
    angle_fubar: list[bool] = [length != 3 for length in angle_counts]
    if any(angle_fubar):
        # At least one LeadingEdge particle doesn't have exactly three Angle bonds, as it should
        break_point = 1
    return

def _make_break_or_become(k_neighbor_count: float, k_angle: float,
                          k_edge_neighbor_count: float, k_edge_angle: float) -> None:
    """
    All the "k": coefficient, like lambda for each energy term in the Potts model, but "lambda" is python reserved word.
    Energy terms: neighbor-count constraint (like Potts model volume constraint), angle constraint.
    k_edge_neighbor_count, k_edge_angle: same, for the leading-edge transformations, so they can be tuned separately.
    """

    def accept(main_particle: tf.ParticleHandle,
               making_particle: tf.ParticleHandle = None,
               breaking_particle: tf.ParticleHandle = None,
               becoming: bool = False) -> bool:
        """Decide whether the indicated transformation between these particles is valid and energetically permissible
        
        making_particle and breaking_particle - but not both - can be None
        
        :param main_particle: the particle being visited; is making and/or breaking bonds to other particles
        :param making_particle: if not None, the particle that main_particle is making a bond with
        :param breaking_particle: if not None, the particle that main_particle is breaking a bond with
        :param becoming: if True, this is one of the leading edge transformations, flagging some special case behavior
        :return: True if the transformation is accepted
        """
        def delta_energy_neighbor_count(main_particle: tf.ParticleHandle,
                                        making_particle: tf.ParticleHandle,
                                        breaking_particle: tf.ParticleHandle) -> float:
            """Return the neighbor-count term of the energy of this bond-remodeling event.
            
            :param main_particle: the main particle
            :param making_particle: the particle that main_particle is making a bond with
            :param breaking_particle: the particle that main_particle is breaking a bond with
            :return: neighbor-count term of the energy of this remodeling event
            """
            if making_particle and breaking_particle:
                # If main_particle is both making a bond and breaking a bond, then its total number of bonds will not
                # change; so the energy change is 0 and there's nothing to calculate
                # (Oops: this was not correct! The energy of main_particle indeed won't change; however, the energy
                # of making_ and breaking_particle will both change, and not necessarily by amounts that exactly cancel,
                # so that needs to be taken into account. Currently I'm not using the coupled-event feature (i.e.,
                # having a making_ and breaking_particle at the same time), but if I decide to try it again, then:
                # ToDo: fix this! It just means that we were miscalculating the energy change for such events, so,
                #  may have accepted or rejected some of these events erroneously.)
                return 0
            
            p1: tf.ParticleHandle = main_particle
            p2: tf.ParticleHandle = making_particle or breaking_particle

            k_neighbor_count_energy: float = k_edge_neighbor_count if is_edge_bond(p1, p2) else k_neighbor_count
            if k_neighbor_count_energy == 0:
                return 0
            
            p1current_count: int = len(p1.bonded_neighbors)
            p2current_count: int = len(p2.bonded_neighbors)
            delta_count: int = -1 if breaking_particle else 1
            p1final_count: int = p1current_count + delta_count
            p2final_count: int = p2current_count + delta_count
    
            # Simple for now, but this will probably get more complex later. I think LeadingEdge target count
            # needs to gradually increase as edge approaches vegetal pole, because of the geometry (hence float).
            p1target_count: float = (6 if p1.type_id == g.Little.id else 4)
            p2target_count: float = (6 if p2.type_id == g.Little.id else 4)
    
            p1current_energy: float = (p1current_count - p1target_count) ** 2
            p2current_energy: float = (p2current_count - p2target_count) ** 2
    
            p1final_energy: float = (p1final_count - p1target_count) ** 2
            p2final_energy: float = (p2final_count - p2target_count) ** 2
    
            delta_energy: float = (p1final_energy + p2final_energy) - (p1current_energy + p2current_energy)
            return k_neighbor_count_energy * delta_energy
        
        def delta_energy_angle(main_particle: tf.ParticleHandle,
                               making_particle: tf.ParticleHandle,
                               breaking_particle: tf.ParticleHandle) -> float:
            """Return the bond-angle term of the energy of this bond-remodeling event.
            
            :param main_particle: the main particle
            :param making_particle: the particle that main_particle is making a bond with
            :param breaking_particle: the particle that main_particle is breaking a bond with
            :return: bond-angle term of the energy of this remodeling event
            """
            class ConstrainedAngle:
                def __init__(self, value: float, target: float):
                    self.value = value
                    self.target = target
    
                def energy(self) -> float:
                    return (self.value - self.target) ** 2

            class AngleStateChange:
                def __init__(self, before: list[ConstrainedAngle], after: list[ConstrainedAngle]):
                    """Contains the bond angles on a given particle that will change during a bond remodeling event

                    :param before: all the angles that are present before the event, that will disappear as a result
                    of the change.
                    :param after: all the angles that will appear as a result of the change, that didn't exist before.
                    """
                    self.before = before
                    self.after = after
    
                def delta_energy(self) -> float:
                    energy_before: float = sum([angle.energy() for angle in self.before])
                    energy_after: float = sum([angle.energy() for angle in self.after])
                    return energy_after - energy_before

            def get_angle_state_change(vertex_particle: tf.ParticleHandle,
                                       making_particle: tf.ParticleHandle | None,
                                       breaking_particle: tf.ParticleHandle | None) -> AngleStateChange:
                """Returns the angles that will change on vertex_particle as a result of this bond-remodeling event
                
                making_particle or breaking_particle – but not both - can be None. If one of them is None,
                then it means vertex_particle is undergoing a pure bond-breaking, or bond-making, event.
                If neither of them is None, then vertex_particle is breaking one bond, and making another.
                
                (Note that when an edge bond is being made or broken (accept()'s "becoming" == True),
                we will never be making and breaking a bond at the same time, so either making_particle
                or breaking_particle will be None.)
                
                Return value contains a list of constrained bond angles present before the remodeling
                (when a bond to breaking_particle still exists and a bond to making_particle has not yet
                been created), and a list of bond angles present after the remodeling (when the bond to
                breaking_particle has been broken and the bond to making_particle has been created).
                The lengths of each list can vary, as follows:
                
                If breaking_particle is None: we are only making a bond, so the "before" list contains a
                    single angle, and the "after" list contains two angles;
                If making_particle is None: we are only breaking a bond, so the "before" list contains two
                    angles, and the "after" list contains a single angle;
                If there is both a breaking_particle and a making_particle, and they are adjacent (angularly
                    situated between the same two other bonded particles around vertex_particle; not
                    separated by any other bonds on vertex_particle), then the "before" and "after" lists
                    will each contain two angles;
                If there is both a breaking_particle and a making_particle, and they are NOT adjacent (they are
                    angularly separated by at least one other bond on vertex_particle), then the "before" and
                    "after" lists will each contain three angles.
                    
                Each of these constrained bond angles has a target value, which depends on whether the bond
                represents the EVL leading edge.
                """
                def get_flanking_angles(other_p: tf.ParticleHandle,
                                        leading_index: int,
                                        trailing_index: int) -> list[ConstrainedAngle]:
                    """Angles flanking the other (non-vertex) particle, and their composite
                    
                    :param other_p: the making_ or breaking_ particle, as the case may be
                    :param leading_index: location in ordered_neighbor_list of the first flanking particle
                    :param trailing_index: location in ordered_neighbor_list of the second flanking particle
                    :return: the leading, trailing, and composite angle. Leading and trailing are the angles
                    that exist when the bond to other_p is present (has been made, or has not yet been broken),
                    and composite is the angle that exists when the bond to other_p is absent (has not yet been
                    made, or has been broken).
                    """
                    theta_leading: float
                    theta_trailing: float
                    target_leading: float
                    target_trailing: float
                    target_composite: float
                    
                    theta_leading = tfu.angle_from_particles(ordered_neighbor_list[leading_index],
                                                             vertex_particle,
                                                             other_p)
                    theta_trailing = tfu.angle_from_particles(other_p,
                                                              vertex_particle,
                                                              ordered_neighbor_list[trailing_index])
    
                    target_leading = target_trailing = target_composite = cfg.target_neighbor_angle
                    if becoming:
                        # Assume(?) that the two component angles are very different sizes; and that the
                        # larger one is the EVL leading edge, so should have a bigger target.
                        # (Note: is this assumption correct?)
                        if theta_leading > theta_trailing:
                            target_leading = cfg.target_edge_angle
                        else:
                            target_trailing = cfg.target_edge_angle
        
                        # And when the bond is absent, the composite angle will be the EVL leading edge
                        target_composite = cfg.target_edge_angle
                    
                    return [ConstrainedAngle(value=theta_leading, target=target_leading),
                            ConstrainedAngle(value=theta_trailing, target=target_trailing),
                            ConstrainedAngle(value=theta_leading + theta_trailing, target=target_composite)]

                ordered_neighbor_list: list[tf.ParticleHandle] = nbrs.get_ordered_bonded_neighbors(
                        vertex_particle,
                        extra_neighbor=making_particle
                        )

                # Note, with newer versions of Tissue Forge, the list.index() function should work for finding
                # ParticleHandles, so we no longer need to traverse the list looking for them as previously,
                # but should be able to go right to them.
                breaking_index: int = -1
                making_index: int = -1
                if breaking_particle:
                    assert breaking_particle in ordered_neighbor_list, "Breaking particle missing from neighbors!"
                    breaking_index = ordered_neighbor_list.index(breaking_particle)
                if making_particle:
                    assert making_particle in ordered_neighbor_list, "Making particle missing from neighbors!"
                    making_index = ordered_neighbor_list.index(making_particle)

                # If there are both a making_ and a breaking_ particle, determine whether they are adjacent in
                # ordered_neighbor_list, because it will affect the algorithm and the shape of the return value.
                adjacent: bool = False
                index_diff: int = 0
                if breaking_particle and making_particle:
                    index_diff = abs(making_index - breaking_index)
                    adjacent = (index_diff == 1 or
                                index_diff == len(ordered_neighbor_list) - 1)

                # Find the bond angles on vertex_particle that are changing. I.e., the angles
                # that will disappear when bonds are made/broken, and those that will appear when
                # bonds are made/broken. These are the angles with vertex_particle at the vertex,
                # one ray along the bond being made/broken, and the other ray through that particle's
                # two ordered neighbors.
                before: list[ConstrainedAngle] = []
                after: list[ConstrainedAngle] = []

                leading_index: int
                trailing_index: int
                leading: ConstrainedAngle
                trailing: ConstrainedAngle
                composite: ConstrainedAngle
                if adjacent:
                    # There are *both* making_ and breaking_ particles; and becoming == False
                    # The before and after angles we want are overlapping pairs, with one or the
                    # other particle present.
                
                    if index_diff == 1:
                        # expected, normally (the two particles are adjacent in the list)
                        leading_index = min(breaking_index, making_index) - 1
                        # works even for 0, because of negative indexing
                    else:
                        # when one is at index 0 and the other is at the end of the list
                        # (so they are not adjacent in the linear list, but are considered adjacent
                        # in the circularized sequence of ordered neighbors)
                        leading_index = -2
                    trailing_index = (leading_index + 3) % len(ordered_neighbor_list)
                    
                    # Get the "before" angles: the angles when breaking_ is present and making_ is absent
                    # (we won't use the composite angle, since it doesn't exist either before or after, in this case)
                    leading, trailing, composite = get_flanking_angles(breaking_particle, leading_index, trailing_index)
                    before.extend([leading, trailing])
                    
                    # Get the "after" angles: the angles when breaking_ is absent and making_ is present
                    leading, trailing, composite = get_flanking_angles(making_particle, leading_index, trailing_index)
                    after.extend([leading, trailing])
                else:
                    # Each of the two particles (if present) is flanked by particles whose bonds
                    # are not changing, so the before and after component angles for each particle are
                    # a single angle, and a pair of angles.
                    if breaking_particle:
                        leading_index = breaking_index - 1  # works even for 0, because of negative indexing
                        trailing_index = (breaking_index + 1) % len(ordered_neighbor_list)
                        leading, trailing, composite = get_flanking_angles(breaking_particle,
                                                                           leading_index,
                                                                           trailing_index)
                        before.extend([leading, trailing])
                        after.append(composite)
                    if making_particle:
                        leading_index = making_index - 1
                        trailing_index = (making_index + 1) % len(ordered_neighbor_list)
                        leading, trailing, composite = get_flanking_angles(making_particle,
                                                                           leading_index,
                                                                           trailing_index)
                        before.append(composite)
                        after.extend([leading, trailing])

                return AngleStateChange(before, after)
            
            k_angle_energy: float = k_edge_angle if becoming else k_angle
            if k_angle_energy == 0:
                return 0
            
            main_p_state_change: AngleStateChange
            breaking_p_state_change = AngleStateChange(before=[], after=[])
            making_p_state_change = AngleStateChange(before=[], after=[])
            
            main_p_state_change = get_angle_state_change(vertex_particle=main_particle,
                                                         making_particle=making_particle,
                                                         breaking_particle=breaking_particle)
            if breaking_particle:
                breaking_p_state_change = get_angle_state_change(vertex_particle=breaking_particle,
                                                                 making_particle=None,
                                                                 breaking_particle=main_particle)
            if making_particle:
                making_p_state_change = get_angle_state_change(vertex_particle=making_particle,
                                                               making_particle=main_particle,
                                                               breaking_particle=None)

            delta_energy: float = (main_p_state_change.delta_energy() +
                                   making_p_state_change.delta_energy() +
                                   breaking_p_state_change.delta_energy())
            return k_angle_energy * delta_energy
            
        # Test for basic validity: screen for configurations that shouldn't happen at all
        assert making_particle or breaking_particle, "Making and breaking particles both equal None!"
        
        bonded_neighbor_ids: list[int]
        if breaking_particle:
            bonded_neighbor_ids = [phandle.id for phandle in nbrs.getBondedNeighbors(breaking_particle)]
            assert main_particle.id in bonded_neighbor_ids,\
                f"Attempting to break bond between non-bonded particles:" \
                f" id={main_particle.id} ({main_particle.type()})," \
                f" id={breaking_particle.id} ({breaking_particle.type()})," \
                f" particle identity = {main_particle == breaking_particle}"
            
        if making_particle:
            bonded_neighbor_ids = [phandle.id for phandle in nbrs.getBondedNeighbors(making_particle)]
            assert main_particle.id not in bonded_neighbor_ids,\
                f"Attempting to make bond between already bonded particles: {main_particle.id}, {making_particle.id}"
            
        # Test for illegal changes: these things will happen, but we reject them:
        
        # No particle may go below the minimum threshold for number of bonds
        if making_particle and breaking_particle:
            # Only need to test breaking_particle; main_particle will gain a bond and lose a bond, so no net change
            breaking_particle_current_count: int = len(breaking_particle.bonded_neighbors)
            if breaking_particle_current_count <= cfg.min_neighbor_count:
                return False
        elif breaking_particle:
            # both main_particle and breaking_particle will lose a bond
            main_particle_current_count: int = len(main_particle.bonded_neighbors)
            breaking_particle_current_count: int = len(breaking_particle.bonded_neighbors)
            if (main_particle_current_count <= cfg.min_neighbor_count or
                    breaking_particle_current_count <= cfg.min_neighbor_count):
                return False
        
        # Internal particles may not acquire more than a maximum threshold of bonds to the leading edge
        if making_particle:
            if main_particle.type_id != making_particle.type_id:
                phandle: tf.ParticleHandle
                p_internal: tf.ParticleHandle = (main_particle if main_particle.type_id == g.Little.id
                                                 else making_particle)
                edge_neighbor_count: int = len([phandle for phandle in nbrs.getBondedNeighbors(p_internal)
                                                if phandle.type_id == g.LeadingEdge.id])
                if edge_neighbor_count >= cfg.max_edge_neighbor_count:
                    return False

        # If we get this far, the change is valid and legal; now we can test for energetic favorability
        delta_energy: float = (delta_energy_neighbor_count(main_particle, making_particle, breaking_particle)
                               + delta_energy_angle(main_particle, making_particle, breaking_particle))
    
        if delta_energy <= 0:
            return True
        else:
            probability: float = math.exp(-delta_energy)
            return random.random() < probability

    def find_breakable_bond(p: tf.ParticleHandle, allowed_types: list[tf.ParticleType]) -> tf.BondHandle | None:
        """Randomly select a bond on p that can be broken"""
        breakable_bonds: list[tf.BondHandle] = nbrs.bonds_to_neighbors_of_types(p, allowed_types)

        if not breakable_bonds:
            # can be empty if p is not bonded to any particle of the allowed types
            return None
        
        # select one at random to break:
        return random.choice(breakable_bonds)

    def attempt_break_bond(p: tf.ParticleHandle) -> int:
        """For internal, break any bond; for leading edge, break any bond to an internal particle
        
        returns: number of bonds broken
        """
        # Don't break bond between two LeadingEdge particles
        allowed_types: list[tf.ParticleType] = [g.Little] if p.type() == g.LeadingEdge else [g.Little, g.LeadingEdge]
        
        bhandle: tf.BondHandle = find_breakable_bond(p, allowed_types)
        if not bhandle:
            # can be None if p is a LeadingEdge particle and is *only* bonded to other LeadingEdge particles
            return 0
        
        other_p: tf.ParticleHandle = tfu.other_particle(p, bhandle)
        if accept(p, breaking_particle=other_p):
            gc.destroy_bond(bhandle)
            return 1
        return 0
    
    def find_bondable_neighbor(p: tf.ParticleHandle, allowed_types: list[tf.ParticleType]) -> tf.ParticleHandle | None:
        """Find nearby candidate particles to which p can bond, and randomly select one of them"""
        if cfg.cell_division_enabled or cfg.bondable_neighbor_discovery == cfg.BondableNeighborDiscovery.NEAREST:
            # With cell division, just get nearest unbonded neighbor to bond to.
            # The approach used below to prevent holes in the absence of cell division, was not helpful for the
            # cell division case (which didn't actually have holes in the first place), and resulted in particle
            # crowding (negative tension) at the animal pole. So, handling this case separately.
            return nbrs.get_nearest_non_bonded_neighbor(p, allowed_types)
        else:
            # When cell division disabled, need to grab particles from further away sometimes, to prevent holes.
            # (And after all this effort, it seems this doesn't quite salvage it. Works only sometimes!)
            #
            # Get a bunch of nearby candidates.
            # A few approaches:
        
            bondable_neighbors: list[tf.ParticleHandle]
            match cfg.bondable_neighbor_discovery:
                case cfg.BondableNeighborDiscovery.OPEN_ENDED:
                    # request minimum 1, but length of returned list commonly ranges from 1 up to 12, or more.
                    # This approach worked okay, though may be a bit harder to explain in paper, so prefer the
                    # approach below.
                    bondable_neighbors = nbrs.get_nearest_non_bonded_neighbors(p,
                                                                               allowed_types,
                                                                               min_neighbors=1)
                    # print(f"Asked for minimum 1 particles, got {len(bondable_neighbors)}")
                case cfg.BondableNeighborDiscovery.BOUNDED:
                    # Get between min and max nearby candidates.
                    # This approach did not work when min = max, regardless of the value! It seems to be important
                    # to have variation in how far out we look for candidate particles. It also didn't work to have
                    # min = 5, max = 8, which captured the bulk of the distribution from the open-ended method; but it
                    # did work with min = 1, max = 7. So it seems it's important to have a lower min, so that we
                    # frequently stick close to home, even though it's the larger values that prevent the holes. So
                    # it requires a balance: SOMETIMES have the possibility to bond with distant candidate particles,
                    # in order to prevent holes; but not so often that you create too much tension in the EVL.
                    minimum: int = cfg.bondable_neighbors_min_candidates
                    maximum: int = cfg.bondable_neighbors_max_candidates
                    bondable_neighbors = nbrs.get_nearest_non_bonded_neighbors_constrained(p,
                                                                                           allowed_types,
                                                                                           min_neighbors=minimum,
                                                                                           max_neighbors=maximum)
                    # print(f"Asked for strictly {minimum}-{maximum} particles, got {len(bondable_neighbors)}")
                case cfg.BondableNeighborDiscovery.UNIFORM:
                    # Get between min and max nearby candidates, but with a known uniform distribution, by deciding
                    # in advance how many we'll get this time.
                    # Note the difference between BOUNDED and UNIFORM: with BOUNDED, we will always get between min
                    # and max candidate neighbors, but there's no telling how many on any given call. In contrast,
                    # with UNIFORM, we will always get between min and max candidate neighbors, but we decide each
                    # time, exactly how many we will request, and we know we'll get exactly that many.
                    # This doesn't seem to have been terribly helpful, however: cleaner, but not more successful.
                    num_neighbors: int = random.randrange(cfg.bondable_neighbors_min_candidates,
                                                          cfg.bondable_neighbors_max_candidates)
                    bondable_neighbors = nbrs.get_nearest_non_bonded_neighbors_constrained(p,
                                                                                           allowed_types,
                                                                                           min_neighbors=num_neighbors,
                                                                                           max_neighbors=num_neighbors)
                    # print(f"Asked for exactly {num_neighbors} particles, got {len(bondable_neighbors)}")
                case _:
                    bondable_neighbors = []
        
            # select one at random to bond to:
            return None if not bondable_neighbors else random.choice(bondable_neighbors)
    
    def attempt_make_bond(p: tf.ParticleHandle) -> int:
        """For internal, bond to a particle selected from nearby unbonded neighbors (either type); for leading edge,
        select from unbonded *internal* neighbors only.
        
        returns: number of bonds created
        """
        # Don't make a bond between two LeadingEdge particles
        allowed_types: list[tf.ParticleType] = [g.Little] if p.type() == g.LeadingEdge else [g.Little, g.LeadingEdge]

        other_p: tf.ParticleHandle = find_bondable_neighbor(p, allowed_types)
        if not other_p:
            # Possible in theory, but with the iterative approach to distance_factor, it seems this never happens.
            # You can always find a non-bonded neighbor.
            return 0
        if accept(p, making_particle=other_p):
            make_bond(p, other_p, verbose=False)
            return 1
        return 0
    
    def attempt_coupled_make_break_bond(p: tf.ParticleHandle) -> int:
        """Break a bond to one internal particle, and make a bond to a different internal particle
        
        :param p: the particle that will have an existing bond broken and a new one made
        :return: number of bond pairs modified (1 or 0)
        """
        # For now, only break and make bonds to internal particles, regardless of whether p is internal or edge
        allowed_types: list[tf.ParticleType] = [g.Little] if p.type() == g.LeadingEdge else [g.Little, g.LeadingEdge]

        # Find a bond to break
        bhandle: tf.BondHandle = find_breakable_bond(p, allowed_types)
        if not bhandle:
            # Can be None if p is a LeadingEdge particle and is *only* bonded to other LeadingEdge particles.
            # If there's no bond to break, we can't do this coupled operation.
            return 0
        breaking_particle: tf.ParticleHandle = tfu.other_particle(p, bhandle)

        # Find an unbonded neighbor particle to bond to
        making_particle: tf.ParticleHandle = find_bondable_neighbor(p, allowed_types)
        if not making_particle:
            # Possible in theory, but with the iterative approach to distance_factor, it seems this never happens.
            # You can always find a non-bonded neighbor.
            # If there's no particle to bond to, we can't do this coupled operation.
            return 0

        if accept(p, making_particle, breaking_particle):
            gc.destroy_bond(bhandle)
            make_bond(p, making_particle, verbose=False)
            return 1
        return 0

    def remodel_angles(p1: tf.ParticleHandle, p2: tf.ParticleHandle, p_becoming: tf.ParticleHandle, add: bool) -> None:
        """Handle the transformation of the Angle bonds accompanying the transformation of a leading edge particle
        
        p1 and p2 flank p_becoming, which is either entering (add == True) or leaving (add == False) the leading edge.
        """
        def cleanup(p: tf.ParticleHandle) -> None:
            """This is to deal with a TF bug which randomly creates garbage AngleHandles out of thin air.
            
            If this particle has any such Angles on it, destroy them. Unclear whether they cause any damage to
            the sim, but, they certainly cause my asserts to trigger and they shouldn't be here. After this,
            the asserts shouldn't fire anymore unless my own code is doing something wrong.
            
            Use TF native destroy(), not my gc.destroy_angle(), because these are already not in the global
            catalog, hence can't be del'ed.
            """
            angle: tf.AngleHandle
            false_angles: list[tf.AngleHandle] = [angle for angle in p.angles
                                                  if angle.id not in gc.angles_by_id]
            for angle in false_angles:
                print(tfu.bluecolor +
                      f"Destroying false angle (id={angle.id}) on particle {p.id}, leaving {len(p.angles) - 1}" +
                      tfu.endcolor)
                angle.destroy()
        
        def get_pivot_angle(p: tf.ParticleHandle) -> tf.AngleHandle | None:
            """Return the particle's pivot Angle (the Angle that has this particle as the CENTER particle) - if any"""
            angle: tf.AngleHandle
            angles: list[tf.AngleHandle] = [angle for angle in p.angles
                                            if p == angle.parts[1]]
            assert len(angles) < 2, f"Found 2 or more pivot Angles on particle {p.id}"
            return None if not angles else angles[0]
        
        def exchange_particles(angle: tf.AngleHandle, bonded_p: tf.ParticleHandle, new_p: tf.ParticleHandle,
                               potential: tf.Potential) -> None:
            """Make this angle connect to the new particle instead of to the previously bonded neighbor.
            
            Conceptually, delete the old outer particle and replace it with the new one. But AngleHandle.parts is
            not writeable, so actually create a new Angle with the proper connection, then delete the old Angle.
            """
            old_outer_p1: tf.ParticleHandle
            new_outer_p1: tf.ParticleHandle
            center_p: tf.ParticleHandle
            old_outer_p2: tf.ParticleHandle
            new_outer_p2: tf.ParticleHandle

            old_outer_p1, center_p, old_outer_p2 = angle.parts
            assert bonded_p in (old_outer_p1, old_outer_p2), f"bonded_p ({bonded_p}) is not part of this Angle!"
            if bonded_p == old_outer_p1:
                new_outer_p1 = new_p
                new_outer_p2 = old_outer_p2
            else:
                new_outer_p1 = old_outer_p1
                new_outer_p2 = new_p

            gc.create_angle(potential, new_outer_p1, center_p, new_outer_p2)

            gc.destroy_angle(angle)

        if not cfg.angle_bonds_enabled:
            return
        
        cleanup(p1)
        cleanup(p2)
        cleanup(p_becoming)
        
        assert len(p1.angles) == 3 and len(p2.angles) == 3,\
            f"While {'joining' if add else 'leaving'} the leading edge, " \
            f"particles {p1.id}, {p2.id} have {len(p1.angles)}, {len(p2.angles)} Angles, respectively (should have 3)."\
            f" p_becoming (id={p_becoming.id}) has {len(p_becoming.angles)} Angles (should have {0 if add else 3})."
        a1: tf.AngleHandle = get_pivot_angle(p1)
        a2: tf.AngleHandle = get_pivot_angle(p2)
        assert a1, f"Particle {p1.id} has no pivot Angle!"
        assert a2, f"Particle {p2.id} has no pivot Angle!"

        k: float = cfg.harmonic_angle_spring_constant
        theta0: float = harmonic_angle_equilibrium_value()
        tol: float = cfg.harmonic_angle_tolerance
        edge_angle_potential: tf.Potential = tf.Potential.harmonic_angle(k=k, theta0=theta0, tol=tol)
        assert edge_angle_potential is not None, f"Failed harmonic_angle potential, k={k}, theta0={theta0}, tol={tol}"

        if add:
            assert len(p_becoming.angles) == 0, f"Recruited particle (id={p_becoming.id}) already has" \
                                                f" {len(p_becoming.angles)} Angle bonds on it! Should have zero!"
            exchange_particles(a1, bonded_p=p2, new_p=p_becoming, potential=edge_angle_potential)
            exchange_particles(a2, bonded_p=p1, new_p=p_becoming, potential=edge_angle_potential)
            # Trouble-shooting new bug (with TF v 0.1.0) – ended up with 2 instead of 3 Angle bonds on it.
            # This assert is excessive, can be removed once that's solved. Just want to find out, did it happen during
            # exchange_particles(), or during gc.create_angle()? (Note, probably on the back end because TF did
            # generate an error in the log. In which case, doesn't matter when it happened, both use tf.Angle.create())
            assert len(p_becoming.angles) == 2, f"Recruited particle (id={p_becoming.id}) now has" \
                                                f" {len(p_becoming.angles)} Angle bonds on it! Should have 2!"
            gc.create_angle(edge_angle_potential, p1, p_becoming, p2)
            assert len(p_becoming.angles) == 3, f"Recruited particle (id={p_becoming.id}) ended up with" \
                                                f" {len(p_becoming.angles)} Angle bonds on it! Should have 3!"
        else:
            assert len(p_becoming.angles) == 3, f"Particle becoming internal (id={p_becoming.id}) starting with" \
                                                f" {len(p_becoming.angles)} Angle bonds on it! Should have 3!"
            exchange_particles(a1, bonded_p=p_becoming, new_p=p2, potential=edge_angle_potential)
            exchange_particles(a2, bonded_p=p_becoming, new_p=p1, potential=edge_angle_potential)
            gc.destroy_angle(p_becoming.angles[0])
            assert len(p_becoming.angles) == 0, f"Particle becoming internal (id={p_becoming.id}) ended up with" \
                                                f" {len(p_becoming.angles)} Angle bonds on it! Should have zero!"

    def attempt_become_internal(p: tf.ParticleHandle) -> int:
        """For LeadingEdge particles only. Become internal, and let its two bonded leading edge neighbors
        bond to one another.
        
        This MAKES a bond.
        returns: number of bonds created
        """
        # #### Bypass:
        # return attempt_make_bond(p)
        
        # #### Actual implementation:
        phandle: tf.ParticleHandle
        neighbor1: tf.ParticleHandle
        neighbor2: tf.ParticleHandle
        
        bonded_neighbors: list[tf.ParticleHandle] = [phandle for phandle in nbrs.getBondedNeighbors(p)
                                                     if phandle.type_id == g.LeadingEdge.id]
        assert len(bonded_neighbors) == 2, f"Leading edge particle {p.id} has {len(bonded_neighbors)}" \
                                           f" leading edge neighbors??? Should always be exactly 2!"
        
        neighbor1, neighbor2 = bonded_neighbors
        p_phi: float = epu.embryo_phi(p)
        if p_phi >= epu.embryo_phi(neighbor1) or p_phi >= epu.embryo_phi(neighbor2):
            # Overly strict test for validity of doing this operation on this particle. We only want to do it
            # if the edge of the EVL is concave here. If phi of the particle is less than phi of the other two
            # particles, it's definitely concave. If it's greater than the other two, it's definitely convex and
            # we reject the operation. If it's between the other two, then it may be either convex or concave,
            # but I don't know how to detect the difference without using slow trigonometry; so for now just reject
            # the operation. (Of course, embryo_phi() also uses trig, but at least tf handles it in C++.)
            return 0
        
        if accept(neighbor1, making_particle=neighbor2, becoming=True):
            # test_ring_is_fubar()
            make_bond(neighbor1, neighbor2, verbose=False)
            p.become(g.Little)
            p.style.color = g.Little.style.color
            p.style.visible = gc.visibility_state
            p.force_init = [0, 0, 0]

            # test_ring_is_fubar()
            remodel_angles(neighbor1, neighbor2, p_becoming=p, add=False)

            # test_ring_is_fubar()
            return 1
        return 0
    
    def attempt_recruit_from_internal(p: tf.ParticleHandle) -> tuple[int, int]:
        """For LeadingEdge particles only. Break the bond with one bonded leading edge neighbor, but only
        if there is an internal particle bonded to both of them. That internal particle becomes leading edge.
        If there are more than one such particle, pick the one with the shortest combined path.
        
        This BREAKS a bond.
        returns: number of bonds that became edge, number of bonds broken
        """
        # #### Bypass:
        # return 0, attempt_break_bond(p)
        
        # #### Actual implementation:
        leading_edge_neighbors: list[tf.ParticleHandle] = [phandle for phandle in nbrs.getBondedNeighbors(p)
                                                           if phandle.type_id == g.LeadingEdge.id]
        assert len(leading_edge_neighbors) == 2, f"Leading edge particle {p.id} has {len(leading_edge_neighbors)}" \
                                                 f" leading edge neighbors??? Should always be exactly 2!"
        
        # Select one neighbor at random; if it doesn't work with that one, try the other one. First, randomize the
        # order of the list (of 2 particles), so I can simply try them in list order without introducing a bias.
        if random.random() < 0.5:
            leading_edge_neighbors.reverse()
            
        other_leading_edge_p: tf.ParticleHandle | None = None
        shared_internal_bonded_neighbors: list[tf.ParticleHandle] = []
        for other_leading_edge_p in leading_edge_neighbors:
            shared_internal_bonded_neighbors = nbrs.get_shared_bonded_neighbors(p, other_leading_edge_p)
            if shared_internal_bonded_neighbors:
                break
                
        # If there was more than one shared neighbor, we'll use the one closest to these 2 edge particles
        # If there were none, we'll do nothing
        recruit: tf.ParticleHandle = min(shared_internal_bonded_neighbors,
                                         key=lambda n: nbrs.distance(n, p, other_leading_edge_p),
                                         default=None)
        
        if not recruit:
            return 0, 0

        # Prevent runaway edge recruitment.
        # Hoped that algorithm improvements would make this unnecessary. Would prefer to understand the cause,
        # but we can at least prevent by a rule. Disallow recruitment if the recruited particle is too far from
        # the leading edge.
        # (Note, currently, radii of particles are all the same. Use of mean radius was anticipating that I might
        # change that. So far I have not.)
        leading_edge_baseline_phi: float = max(epu.embryo_phi(p), epu.embryo_phi(other_leading_edge_p))
        mean_particle_radius: float = fmean([p.radius, other_leading_edge_p.radius, recruit.radius])
        leading_edge_recruitment_limit_distance: float = cfg.leading_edge_recruitment_limit * mean_particle_radius
        # ratio of any arc's length in distance units, to the radians it represents, is always:
        # distance_to_radians_ratio =
        # full circumference of circle (i.e. 2*pi*r) / radians in a full circle (i.e. 2*pi), which = r;
        # hence if we know the distance, then the radians = distance / r
        embryo_radius: float = g.Big.radius + mean_particle_radius
        leading_edge_recruitment_limit_radians: float = leading_edge_recruitment_limit_distance / embryo_radius
        if epu.embryo_phi(recruit) < leading_edge_baseline_phi + leading_edge_recruitment_limit_radians:
            return 0, 0

        # In case recruit is bonded to any additional *other* LeadingEdge particles, disallow this
        # (instead of simply breaking those extra bonds, like before, which led to problems).
        num_bonds_to_leading_edge: int = nbrs.count_neighbors_of_types(recruit, [g.LeadingEdge])
        if num_bonds_to_leading_edge > 2:
            return 0, 0

        if accept(p, breaking_particle=other_leading_edge_p, becoming=True):
            # In case recruit is bonded to an *internal* particle that already has the maximum edge bonds,
            # break the bond with that particle. Recruit will become LeadingEdge, which means bonded neighbors
            # will get an additional bond to the edge. So, if internal neighbor is already bonded to the maximum,
            # we should not make one more such bond, so delete it.
            def too_many_edge_neighbors(p: tf.ParticleHandle) -> bool:
                return nbrs.count_neighbors_of_types(p, [g.LeadingEdge]) >= cfg.max_edge_neighbor_count
                
            phandle: tf.ParticleHandle
            saturated_internal_neighbors: list[tf.ParticleHandle] = [phandle
                                                                     for phandle in nbrs.getBondedNeighbors(recruit)
                                                                     if phandle.type_id == g.Little.id
                                                                     if too_many_edge_neighbors(phandle)]
            for phandle in saturated_internal_neighbors:
                gc.destroy_bond(tfu.bond_between(recruit, phandle))
            # test_ring_is_fubar()
            
            gc.destroy_bond(tfu.bond_between(p, other_leading_edge_p))
            recruit.become(g.LeadingEdge)
            recruit.style.color = g.LeadingEdge.style.color
            recruit.style.visible = True
            recruit.force_init = [0, 0, 0]  # remove particle diffusion force
            
            remodel_angles(p, other_leading_edge_p, p_becoming=recruit, add=True)
            
            # test_ring_is_fubar()
            return 1, 1 + len(saturated_internal_neighbors)
        return 0, 0

    assert k_neighbor_count >= 0 and k_angle >= 0, f"k values must be non-negative; " \
                                                   f"k_neighbor_count = {k_neighbor_count}, k_angle = {k_angle}"
    assert k_edge_neighbor_count >= 0 and k_edge_angle >= 0, f"k values must be non-negative; " \
                                                             f"k_edge_neighbor_count = {k_edge_neighbor_count}, " \
                                                             f"k_edge_angle = {k_edge_angle}"
    total_bonded: int = 0
    total_broken: int = 0
    total_coupled: int = 0
    total_to_internal: int = 0
    total_to_edge: int = 0
    result: int
    p: tf.ParticleHandle
    ran: float
    
    # Constrain to between 0 and 1
    uncoupled_bond_remodeling_freq: float = 1 - max(0.0, min(1.0, cfg.coupled_bond_remodeling_freq))
    
    start = time.perf_counter()
    for p in g.Little.items():
        ran = random.random()
        if ran < uncoupled_bond_remodeling_freq / 2:
            total_bonded += attempt_make_bond(p)
        elif ran < uncoupled_bond_remodeling_freq:
            total_broken += attempt_break_bond(p)
        else:
            result = attempt_coupled_make_break_bond(p)
            total_broken += result
            total_bonded += result
            total_coupled += result
        
    for p in g.LeadingEdge.items():
        ran = random.random()
        if ran < uncoupled_bond_remodeling_freq / 4:
            total_bonded += attempt_make_bond(p)
        elif ran < uncoupled_bond_remodeling_freq / 2:
            total_broken += attempt_break_bond(p)
        elif ran < 0.5:
            result = attempt_coupled_make_break_bond(p)
            total_broken += result
            total_bonded += result
            total_coupled += result
        elif ran < 0.75:
            result = attempt_become_internal(p)
            total_bonded += result
            total_to_internal += result
        else:
            became_edge, got_broken = attempt_recruit_from_internal(p)
            total_broken += got_broken
            total_to_edge += became_edge
    end = time.perf_counter()

    print(f"Created {total_bonded} bonds and broke {total_broken} bonds, in {round(end - start, 2)} sec.; "
          f"of those, {total_coupled} were coupled; "
          f"{total_to_edge} became edge; {total_to_internal} became internal")
    epu.cumulative_to_edge += total_to_edge
    epu.cumulative_from_edge += total_to_internal
    
def _move_toward_open_space() -> None:
    """Prevent gaps from opening up by giving particles a nudge to move toward open space.
    
    Where "toward open space" means: away from the centroid of all the bonded neighbors. This should prevent the
    situation where particles look for potential bonding partners but can't find one in the direction in which
    they are most needed (the direction of an incipient hole, i.e., toward areas where particles are most
    stretched apart).
    
    This applies only to internal particles. Furthermore, for particles bonded directly to the leading edge, a
    modified version of this is applied, to prevent anomalies near the edge. (The empty area below the leading edge
    would be seen as a "hole", and edge-bonded internal particles would get ejected into it.) For those edge-bonded
    particles, whenever the centroid-based force vector would push the particle closer to the leading edge,
    remove the edge-directed (vegetalward) component of the vector, leaving only the horizontal (circumferential)
    component, and use that.
    """
    phandle: tf.ParticleHandle
    for phandle in g.Little.items():
        bonded_neighbors: tf.ParticleList = nbrs.getBondedNeighbors(phandle)
        bonded_neighbor_positions: tuple[tf.fVector3] = bonded_neighbors.positions
        vecsum: tf.fVector3 = sum(bonded_neighbor_positions, start=tf.fVector3([0, 0, 0]))
        centroid: tf.fVector3 = vecsum / len(bonded_neighbor_positions)
        force: tf.fVector3 = (phandle.position - centroid) * cfg.k_particle_diffusion

        neighbor: tf.ParticleHandle
        bonded_edge_neighbors: list[tf.ParticleHandle] = [neighbor for neighbor in bonded_neighbors
                                                          if neighbor.type() == g.LeadingEdge]
        if len(bonded_edge_neighbors) > 0:
            # We want to add the diffusion force whenever possible, because it helps maintain tissue integrity;
            # but it doesn't work as-is for particles bound to the leading edge, so need special case behavior:
            # Adding force based on the centroid can cause these cells to go careening into the open vegetal
            # space. Particularly particles immediately after transforming from edge to internal type; or when
            # particles are closely packed (as in the balanced-force control or when cell division is enabled).
            # So, refrain from pushing edge-bonded particles toward the leading edge / vegetal pole. Remove
            # the component of the force that is pushing in that direction.
            
            # ToDo: Hmm, in the calculations below, it might be more correct to use centroid rather than phandle
            #  to get the initial coordinates? (Or maybe even, a point mid-way between the two?) But I've done a ton
            #  of testing using phandle, and if it ain't broke don't fix it, so stick with this for now and maybe
            #  consider that later. The two points are very close to one another, so how much does it matter?
            
            # Find the vertical tangent direction (along the longitude line) at this point, to decide whether the
            # force vector is pointing toward or away from vegetal; this will be more reliable, when the leading
            # edge is near the vegetal pole, than just using the z component of the vector.
            tangent_point_theta, tangent_point_phi = epu.embryo_coords(phandle)
            polar_theta: float = tangent_point_theta
            animalward_phi: float = tangent_point_phi - math.pi / 2
            animalward_tangent_direction: tf.fVector3 = tfu.cartesian_from_spherical([1, polar_theta, animalward_phi])
            force_toward_vegetal: bool = force.dot(animalward_tangent_direction) < 0
            if force_toward_vegetal:
                # If the force vector points toward animal, then it's fine; just use it. But if it points
                # toward vegetal, project it onto the horizontal. I.e., just use the horizonal component of
                # the force, and not the vegetelward component. (Vegetalward is not the same as "down", so
                # not as simple as just setting z=0.)
                # Find the horizontal tangent direction (latitude line), to project the calculated force vector onto:
                horizontal_theta: float = tangent_point_theta + math.pi / 2
                horizontal_phi: float = math.pi / 2
                horizontal_tangent_direction = tfu.cartesian_from_spherical([1, horizontal_theta, horizontal_phi])
                force = force.projectedOntoNormalized(horizontal_tangent_direction)
            
        phandle.force_init = force.as_list()
        

def maintain_bonds(k_neighbor_count: float = 0.4, k_angle: float = 2,
                   k_edge_neighbor_count: float = 2, k_edge_angle: float = 2) -> None:
    _make_break_or_become(k_neighbor_count, k_angle,
                          k_edge_neighbor_count, k_edge_angle)
    if cfg.space_filling_enabled:
        _move_toward_open_space()
