"""Handle the remodeling of the bond network as the tissue changes shape"""
import math
import random
import time
from typing import Optional

import tissue_forge as tf
from epiboly_init import Little, LeadingEdge
import config as cfg
from utils import tf_utils as tfu,\
    epiboly_utils as epu,\
    global_catalogs as gc

import neighbors as nbrs

def is_edge_bond(p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> bool:
    return p1.type_id == p2.type_id == LeadingEdge.id

def _make_bond(p1: tf.ParticleHandle, p2: tf.ParticleHandle, verbose: bool = False) -> None:
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
    r0: float = 2 * Little.radius
    potential: tf.Potential = tf.Potential.harmonic(r0=r0,
                                                    k=k,
                                                    max=cfg.max_potential_cutoff
                                                    )
    handle: tf.BondHandle = gc.create_bond(potential, p1, p2, r0)

    if verbose:
        distance: float = p1.distance(p2)
        print(f"Making new bond {handle.id} between particles {p1.id} and {p2.id},",
              f"distance = (radius * {distance/Little.radius})")
        # p1.style.color = tfu.gray   # testing
        # p2.style.color = tfu.white  # testing

def make_all_bonds(phandle: tf.ParticleHandle, verbose=False) -> int:
    # Bond to all neighbors not already bonded to
    neighbors: list[tf.ParticleHandle]
    neighbors = nbrs.get_nearest_non_bonded_neighbors(phandle,
                                                      min_neighbors=cfg.min_neighbor_count,
                                                      min_distance=cfg.min_neighbor_initial_distance_factor)
    for neighbor in neighbors:
        _make_bond(neighbor, phandle, verbose)
    return len(neighbors)

def harmonic_angle_equilibrium_value() -> float:
    """A function rather than just a config constant because it depends on the number of particles in the ring"""
    # Equilibrium angle might look like π from within the plane of the leading edge, but the actual angle is
    # different. And, it changes if the number of leading edge particles changes. Hopefully it won't need to be
    # dynamically updated to be that precise. If the number of particles changes, they'll "try" to reach a target angle
    # that is not quite right, but will be opposed by the same force acting on the neighbor particles, so hopefully
    # it all balances out. (For the same reason, π would probably also work, but this value is closer to the real one.)
    return math.pi - (cfg.two_pi / len(LeadingEdge.items()))

def test_ring_is_fucked_up():
    """For debugging. Set breakpoints at the indicated locations. Stop there and examine these values."""
    particles: list[tf.ParticleHandle] = [p for p in LeadingEdge.items()]
    neighbor_lists: list[list[tf.ParticleHandle]] = [p.getBondedNeighbors() for p in particles]
    leading_edge_neighbor_lists: list[list[tf.ParticleHandle]] = (
            [[n for n in neighbor_list if n.type_id == LeadingEdge.id]
             for neighbor_list in neighbor_lists])
    leading_edge_counts: list[int] = [len(neighbor_list) for neighbor_list in leading_edge_neighbor_lists]
    neighbor_fuckedness: list[bool] = [length != 2 for length in leading_edge_counts]
    if any(neighbor_fuckedness):
        # At least one LeadingEdge particle doesn't have exactly two LeadingEdge neighbors, as it should
        break_point = 0
    angle_lists: list[list[tf.AngleHandle]] = [p.angles for p in particles]
    angle_counts: list[int] = [len(angle_list) for angle_list in angle_lists]
    angle_fuckedness: list[bool] = [length != 3 for length in angle_counts]
    if any(angle_fuckedness):
        # At least one LeadingEdge particle doesn't have exactly three Angle bonds, as it should
        break_point = 1
    return

def _make_break_or_become(k_neighbor_count: float, k_angle: float,
                          k_edge_neighbor_count: float, k_edge_angle: float, verbose: bool = False) -> None:
    """
    All the "k": coefficient, like lambda for each energy term in the Potts model, but "lambda" is python reserved word.
    Energy terms: neighbor-count constraint (like Potts model volume constraint), angle constraint.
    k_edge_neighbor_count, k_edge_angle: same, for the leading-edge transformations, so they can be tuned separately.
    """

    # @profile
    def accept(p1: tf.ParticleHandle, p2: tf.ParticleHandle, breaking: bool, becoming: bool = False) -> bool:
        """Decide whether the bond between these two particles may be made/broken
        
        breaking: if True, decide whether to break a bond; if False, decide whether to make a new one
        becoming: if True, this is one of the leading edge transformations, flagging some special case behavior
        """
        def delta_energy_neighbor_count(p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> float:
            k_neighbor_count_energy: float = k_edge_neighbor_count if is_edge_bond(p1, p2) else k_neighbor_count
            if k_neighbor_count_energy == 0:
                return 0
            
            p1current_count: int = len(p1.bonds)
            p2current_count: int = len(p2.bonds)
            delta_count: int = -1 if breaking else 1
            p1final_count: int = p1current_count + delta_count
            p2final_count: int = p2current_count + delta_count
    
            # Simple for now, but this will probably get more complex later. I think LeadingEdge target count
            # needs to gradually increase as edge approaches vegetal pole, because of the geometry (hence float).
            p1target_count: float = (6 if p1.type_id == Little.id else 4)
            p2target_count: float = (6 if p2.type_id == Little.id else 4)
    
            p1current_energy: float = (p1current_count - p1target_count) ** 2
            p2current_energy: float = (p2current_count - p2target_count) ** 2
    
            p1final_energy: float = (p1final_count - p1target_count) ** 2
            p2final_energy: float = (p2final_count - p2target_count) ** 2
    
            delta_energy: float = (p1final_energy + p2final_energy) - (p1current_energy + p2current_energy)
            return k_neighbor_count_energy * delta_energy
        
        def delta_energy_angle(p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> float:
            def get_component_angles(vertex_particle: tf.ParticleHandle,
                                     ordered_neighbor_list: list[tf.ParticleHandle],
                                     other_p: tf.ParticleHandle) -> tuple[tuple[float, float],
                                                                          tuple[float, float],
                                                                          float]:
                """returns the angles that will change when a bond is made/broken:
                
                first sub-tuple: the two component angles
                second sub-tuple: the target angle for each respective component angle,
                    which will differ depending on whether it represents the leading edge.
                    
                I.e., the two angles (bonded_neighbor -> vertex_particle -> other_p) that will come into
                existence if a bond to other_p is added, or that will fuse into a larger angle
                (bonded_neighbor -> vertex_particle -> consecutive_bonded_neighbor) if an existing bond to
                other_p is broken.
                
                final element of tuple: the target angle for the fused angle (i.e. if the bond in
                    question gets broken, or doesn't get made)
                """
                # Find the two angles before and after other_p. I.e., the angles between other_p and
                # its two ordered neighbors, each with vertex_particle as the vertex
                
                # I don't trust doing this until the next release, because it's not
                # entirely clear that .index() would recognize the one in the list is "equal" to other_p,
                # even if it actually "is" other_p. For now, loop and compare the ids instead.
                # other_p_index: int = ordered_neighbor_list.index(other_p)
                # previous_neighbor_index: int = other_p_index - 1    # works even for 0, because of negative indexing
                # next_neighbor_index: int = (other_p_index + 1) % len(ordered_neighbor_list)
                # theta1: float = angle(ordered_neighbor_list[previous_neighbor_index],
                #                       vertex_particle,
                #                       other_p)
                # theta2: float = angle(other_p,
                #                       vertex_particle,
                #                       ordered_neighbor_list[next_neighbor_index])

                theta1: Optional[float] = None
                theta2: Optional[float] = None
                previous_neighbor: tf.ParticleHandle = ordered_neighbor_list[-1]
                for p in ordered_neighbor_list:
                    if p.id == other_p.id:
                        theta1 = tfu.angle_from_particles(previous_neighbor, vertex_particle, p)
                        # Previously tried here, and might come back to it again: if p and previous_neighbor
                        # are both LeadingEdge, then set edge target angle based on that. But, it works
                        # for breaking a bond and turning a Little into a LeadingEdge; it does not work
                        # for making a bond and turning a LeadingEdge into a Little, because all the particles
                        # are LeadingEdge and you can't tell the cases apart. Hence, created the "becoming"
                        # parameter and handled it separately below, instead. And either way, I never got
                        # the accept/reject criterion really working for edge transformations. (Hence the
                        # addition of the ad hoc criterion, leading_edge_baseline, in recruit_from_internal().)
                    elif previous_neighbor.id == other_p.id:
                        theta2 = tfu.angle_from_particles(previous_neighbor, vertex_particle, p)
                        
                    if theta1 is not None and theta2 is not None:
                        break
                        
                    previous_neighbor = p
                    
                target1: float
                target2: float
                fused_target: float
                target1 = target2 = fused_target = cfg.target_neighbor_angle
                if becoming:
                    # Assume(?) that the two component angles are very different sizes; and that the
                    # larger one is the leading edge, so should have a bigger target.
                    # (Note: probably a WRONG assumption, because I'm getting very bad behaviors.)
                    if theta1 > theta2:
                        target1 = cfg.target_edge_angle
                    else:
                        target2 = cfg.target_edge_angle
                    
                    # And if the bond gets broken / does not get made, then the fused angle will be the leading edge
                    fused_target = cfg.target_edge_angle

                return (theta1, theta2), (target1, target2), fused_target
            
            k_angle_energy: float = k_edge_angle if becoming else k_angle
            if k_angle_energy == 0:
                return 0
            
            p1_extra: tf.ParticleHandle = None if breaking else p2
            p2_extra: tf.ParticleHandle = None if breaking else p1
            p1_neighbors: list[tf.ParticleHandle] = nbrs.get_ordered_bonded_neighbors(p1, extra_neighbor=p1_extra)
            p2_neighbors: list[tf.ParticleHandle] = nbrs.get_ordered_bonded_neighbors(p2, extra_neighbor=p2_extra)
            
            p1_angles: tuple[float, float]
            p1_targets: tuple[float, float]
            p2_angles: tuple[float, float]
            p2_targets: tuple[float, float]
            p1_fused_target: float
            p2_fused_target: float

            p1_angles, p1_targets, p1_fused_target = get_component_angles(vertex_particle=p1,
                                                                          ordered_neighbor_list=p1_neighbors,
                                                                          other_p=p2)
            p2_angles, p2_targets, p2_fused_target = get_component_angles(vertex_particle=p2,
                                                                          ordered_neighbor_list=p2_neighbors,
                                                                          other_p=p1)
            
            p1_component_energy: float = ((p1_angles[0] - p1_targets[0]) ** 2 +
                                          (p1_angles[1] - p1_targets[1]) ** 2)
            p2_component_energy: float = ((p2_angles[0] - p2_targets[0]) ** 2 +
                                          (p2_angles[1] - p2_targets[1]) ** 2)
            p1_fused: float = p1_angles[0] + p1_angles[1]
            p2_fused: float = p2_angles[0] + p2_angles[1]
            p1_fused_energy: float = (p1_fused - p1_fused_target) ** 2
            p2_fused_energy: float = (p2_fused - p2_fused_target) ** 2
            
            delta_energy_making: float = ((p1_component_energy + p2_component_energy) -
                                          (p1_fused_energy + p2_fused_energy))
            delta_energy_breaking: float = -delta_energy_making
            
            if breaking:
                return k_angle_energy * delta_energy_breaking
            else:
                return k_angle_energy * delta_energy_making
            
        bonded_neighbor_ids: list[int] = [phandle.id for phandle in p2.getBondedNeighbors()]
        if breaking:
            assert p1.id in bonded_neighbor_ids,\
                f"Attempting to break bond between non-bonded particles: {p1.id}, {p2.id}"
        else:
            assert p1.id not in bonded_neighbor_ids,\
                f"Attempting to make bond between already bonded particles: {p1.id}, {p2.id}"
            
        # Neither particle may go below the minimum threshold for number of bonds
        p1current_count: int = len(p1.bonds)
        p2current_count: int = len(p2.bonds)
        if breaking and (p1current_count <= cfg.min_neighbor_count or
                         p2current_count <= cfg.min_neighbor_count):
            if verbose:     # and p1.type_id == LeadingEdge.id and p2.type_id == LeadingEdge.id:
                print(f"Rejecting break because particles have {p1current_count} and {p2current_count} bonds")
            return False
        
        # Internal particles may not acquire more than a maximum threshold of bonds to the leading edge
        if p1.type_id != p2.type_id and not breaking:
            phandle: tf.ParticleHandle
            p_internal: tf.ParticleHandle = p1 if p1.type_id == Little.id else p2
            edge_neighbor_count: int = len([phandle for phandle in p_internal.getBondedNeighbors()
                                            if phandle.type_id == LeadingEdge.id])
            if edge_neighbor_count >= cfg.max_edge_neighbor_count:
                if verbose:
                    print(f"Rejecting new bond between internal and leading edge because internal particle"
                          f" is already bonded to {edge_neighbor_count} LeadingEdge particles")
                return False

        delta_energy: float = (delta_energy_neighbor_count(p1, p2)
                               + delta_energy_angle(p1, p2))
    
        if delta_energy <= 0:
            return True
        else:
            probability: float = math.exp(-delta_energy)
            if random.random() < probability:
                return True
            else:
                if verbose:     # and p1.type_id == LeadingEdge.id and p2.type_id == LeadingEdge.id:
                    print(f"Rejecting {'break' if breaking else 'make'} because unfavorable; particles have"
                          f" {p1current_count} and {p2current_count} bonds")
                return False

    # @profile
    def attempt_break_bond(p: tf.ParticleHandle) -> int:
        """For internal, break any bond; for leading edge, break any bond to an internal particle
        
        returns: number of bonds broken
        """
        # profiling: just run this once; mprof output might be easier to understand:
        # global _profiled_break
        if _profiled_break:
            return 0
        
        bhandle: tf.BondHandle
        breakable_bonds: list[tf.BondHandle] = p.bonds
        if p.type_id == LeadingEdge.id:
            # Don't break bond between two LeadingEdge particles
            breakable_bonds = [bhandle for bhandle in breakable_bonds
                               if tfu.other_particle(p, bhandle).type_id == Little.id]
        if not breakable_bonds:
            # can be empty if p is a LeadingEdge particle and is *only* bonded to other LeadingEdge particles
            return 0
        
        # select one at random to break:
        bhandle = random.choice(breakable_bonds)
        other_p: tf.ParticleHandle = tfu.other_particle(p, bhandle)
        # _profiled_break = True
        if accept(p, other_p, breaking=True):
            # _profiled_break = True
            if verbose:
                print(f"Breaking bond {bhandle.id} between particles {p.id} and {other_p.id}")
            gc.destroy_bond(bhandle)
            return 1
        return 0
    
    # @profile
    def attempt_make_bond(p: tf.ParticleHandle) -> int:
        """For internal, bond to the closest unbonded neighbor (either type); for leading edge, bond to
        the closest unbonded *internal* neighbor only.
        
        returns: number of bonds created
        """
        # profiling: just run this once; mprof output might be easier to understand:
        global _profiled_make
        if _profiled_make:
            return 0

        other_p: tf.ParticleHandle
        if p.type_id == LeadingEdge.id:
            # Don't make a bond between two LeadingEdge particles
            other_p = nbrs.get_nearest_non_bonded_neighbor(p, [Little])
        else:
            other_p = nbrs.get_nearest_non_bonded_neighbor(p, [Little, LeadingEdge])
        
        if not other_p:
            # Possible in theory, but with the iterative approach to distance_factor, it seems this never happens.
            # You can always find a non-bonded neighbor.
            if verbose:
                print("Can't make bond: No particles available")
            return 0
        _profiled_make = True
        if accept(p, other_p, breaking=False):
            # _profiled_make = True
            _make_bond(p, other_p, verbose=verbose)
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
                print(f"Destroying false angle (id={angle.id}) on particle {p.id}, leaving {len(p.angles) - 1}")
                angle.destroy()
        
        def get_pivot_angle(p: tf.ParticleHandle) -> Optional[tf.AngleHandle]:
            """Return the particle's pivot Angle (the Angle that has this particle as the CENTER particle) - if any"""
            angle: tf.AngleHandle
            angles: list[tf.AngleHandle] = [angle for angle in p.angles
                                            if p.id == angle.parts[1]]
            assert len(angles) < 2, f"Found 2 or more pivot Angles on particle {p.id}"
            return None if not angles else angles[0]
        
        def exchange_particles(angle: tf.AngleHandle, bonded_p: tf.ParticleHandle, new_p: tf.ParticleHandle,
                               potential: tf.Potential) -> None:
            """Make this angle connect to the new particle instead of to the previously bonded neighbor.
            
            Conceptually, delete the old outer particle and replace it with the new one. But AngleHandle.parts is
            not writeable, so actually create a new Angle with the proper connection, then delete the old Angle.
            """
            outer_p1id: int
            outer_p2id: int
            center_pid: int
            old_outer_p1: tf.ParticleHandle
            new_outer_p1: tf.ParticleHandle
            center_p: tf.ParticleHandle
            old_outer_p2: tf.ParticleHandle
            new_outer_p2: tf.ParticleHandle

            outer_p1id, center_pid, outer_p2id = angle.parts
            center_p = gc.particles_by_id[center_pid]["handle"]
            old_outer_p1 = gc.particles_by_id[outer_p1id]["handle"]
            old_outer_p2 = gc.particles_by_id[outer_p2id]["handle"]
            assert bonded_p.id in (outer_p1id, outer_p2id), f"bonded_p (id = {bonded_p.id}) is not part of this Angle!"
            if bonded_p.id == outer_p1id:
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

    # @profile
    def attempt_become_internal(p: tf.ParticleHandle) -> int:
        """For LeadingEdge particles only. Become internal, and let its two bonded leading edge neighbors
        bond to one another.
        
        This MAKES a bond.
        returns: number of bonds created
        """
        # profiling: just run this once; mprof output might be easier to understand:
        # global _profiled_become
        if _profiled_become:
            return 0

        # #### Bypass:
        # return attempt_make_bond(p)
        
        # #### Actual implementation:
        phandle: tf.ParticleHandle
        neighbor1: tf.ParticleHandle
        neighbor2: tf.ParticleHandle
        
        bonded_neighbors: list[tf.ParticleHandle] = [phandle for phandle in p.getBondedNeighbors()
                                                     if phandle.type_id == LeadingEdge.id]
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

        # _profiled_become = True
        if accept(neighbor1, neighbor2, breaking=False, becoming=True):
            # _profiled_become = True
            # test_ring_is_fucked_up()
            _make_bond(neighbor1, neighbor2, verbose=verbose)
            p.become(Little)
            p.style.color = Little.style.color
            p.style.visible = gc.visibility_state
            p.force_init = [0, 0, 0]

            # test_ring_is_fucked_up()
            remodel_angles(neighbor1, neighbor2, p_becoming=p, add=False)

            # test_ring_is_fucked_up()
            return 1
        return 0
    
    # @profile
    def attempt_recruit_from_internal(p: tf.ParticleHandle) -> tuple[int, int]:
        """For LeadingEdge particles only. Break the bond with one bonded leading edge neighbor, but only
        if there is an internal particle bonded to both of them. That internal particle becomes leading edge.
        If there are more than one such particle, pick the one with the shortest combined path.
        
        This BREAKS a bond.
        returns: number of bonds that became edge, number of bonds broken
        """
        # profiling: just run this once; mprof output might be easier to understand:
        # global _profiled_recruit
        if _profiled_recruit:
            return 0, 0

        # #### Bypass:
        # return 0, attempt_break_bond(p)
        
        # #### Actual implementation:
        leading_edge_neighbors: list[tf.ParticleHandle] = [phandle for phandle in p.getBondedNeighbors()
                                                           if phandle.type_id == LeadingEdge.id]
        assert len(leading_edge_neighbors) == 2, f"Leading edge particle {p.id} has {len(leading_edge_neighbors)}" \
                                                 f" leading edge neighbors??? Should always be exactly 2!"
        
        # Select one neighbor at random; if it doesn't work with that one, try the other one.
        # By doing it this way, I can simply try them in list order without introducing a bias.
        if random.random() < 0.5:
            leading_edge_neighbors.reverse()
            
        other_leading_edge_p: Optional[tf.ParticleHandle] = None
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

        # # ##### Alternative criterion for preventing runaway edge recruitment #####
        # # For awhile this seemed like the more natural approach, and worked well, at least
        # # when Angle bonds were included. After changing the number and size of EVL particles,
        # # it needed adjusting (had worked with cfg.leading_edge_recruitment_min_angle = pi/3.5; now
        # # needed to make it more stringent, with pi/2.5). But I did not really like the arbitrariness
        # # of this, or understand why it would be impacted by particle size, so probably will just
        # # stick with the old way, below. That way is based on a multiple of radius, and still works
        # # robustly even after the particle size change.
        # # Also, deprecated because now I'm disabling Angle bonds while tuning setup and equilibration,
        # # and this method never worked in the absence of Angle bonds. Algorithm used here needs to work
        # # the same while tuning as in the regular sim, so really can't use this version anymore.
        # # Keeping this comment only until I do the final needed fix below.
        # if cfg.angle_bonds_enabled:
        #     recruit_angle: float = tfu.angle_from_particles(p1=p, p_vertex=recruit, p2=other_leading_edge_p)
        #     if recruit_angle < cfg.leading_edge_recruitment_min_angle:
        #         return 0, 0
        
        # Prevent runaway edge recruitment.
        # Hoped that algorithm improvements would make this unnecessary. Would prefer to understand the cause,
        # but we can at least prevent by a rule. Disallow recruitment if the recruited particle is too far from
        # the leading edge.
        # ToDo: this really should be based on phi rather than on z though, because as epiboly progresses,
        #  the difference in z becomes smaller and smaller, and less relevant. (Once that has been done and
        #  validated, finally remove altogether the alternative method above.)
        pos: tf.fVector3
        leading_edge_baseline: float = min([pos.z() for pos in LeadingEdge.items().positions])
        leading_edge_recruitment_zone: float = cfg.leading_edge_recruitment_limit * LeadingEdge.radius
        if recruit.position.z() > leading_edge_baseline + leading_edge_recruitment_zone:
            return 0, 0

        # In case recruit is bonded to any additional *other* LeadingEdge particles, disallow this
        # (instead of simply breaking those extra bonds, like before, which led to problems).
        num_bonds_to_leading_edge: int = nbrs.count_neighbors_of_type(recruit, ptype=LeadingEdge)
        if num_bonds_to_leading_edge > 2:
            return 0, 0

        # _profiled_recruit = True
        if accept(p, other_leading_edge_p, breaking=True, becoming=True):
            # In case recruit is bonded to an *internal* particle that already has the maximum edge bonds,
            # break the bond with that particle. Recruit will become LeadingEdge, which means bonded neighbors
            # will get an additional bond to the edge. So, if internal neighbor is already bonded to the maximum,
            # we should not make one more such bond, so delete it.
            def too_many_edge_neighbors(p: tf.ParticleHandle) -> bool:
                return nbrs.count_neighbors_of_type(p, ptype=LeadingEdge) >= cfg.max_edge_neighbor_count
                
            # _profiled_recruit = True
            phandle: tf.ParticleHandle
            saturated_internal_neighbors: list[tf.ParticleHandle] = [phandle for phandle in recruit.getBondedNeighbors()
                                                                     if phandle.type_id == Little.id
                                                                     if too_many_edge_neighbors(phandle)]
            for phandle in saturated_internal_neighbors:
                gc.destroy_bond(tfu.bond_between(recruit, phandle))
            # test_ring_is_fucked_up()
            
            gc.destroy_bond(tfu.bond_between(p, other_leading_edge_p))
            recruit.become(LeadingEdge)
            recruit.style.color = LeadingEdge.style.color
            recruit.style.visible = True
            recruit.force_init = [0, 0, 0]  # remove particle diffusion force
            
            remodel_angles(p, other_leading_edge_p, p_becoming=recruit, add=True)
            
            # test_ring_is_fucked_up()
            return 1, 1 + len(saturated_internal_neighbors)
        return 0, 0

    assert k_neighbor_count >= 0 and k_angle >= 0, f"k values must be non-negative; " \
                                                   f"k_neighbor_count = {k_neighbor_count}, k_angle = {k_angle}"
    assert k_edge_neighbor_count >= 0 and k_edge_angle >= 0, f"k values must be non-negative; " \
                                                             f"k_edge_neighbor_count = {k_edge_neighbor_count}, " \
                                                             f"k_edge_angle = {k_edge_angle}"
    total_bonded: int = 0
    total_broken: int = 0
    total_to_internal: int = 0
    total_to_edge: int = 0
    result: int
    p: tf.ParticleHandle
    
    start = time.perf_counter()
    for p in Little.items():
        if random.random() < 0.5:
            total_bonded += attempt_make_bond(p)
        else:
            total_broken += attempt_break_bond(p)
        
    for p in LeadingEdge.items():
        ran: float = random.random()
        if ran < 0.25:
            total_bonded += attempt_make_bond(p)
        elif ran < 0.5:
            total_broken += attempt_break_bond(p)
        elif ran < 0.75:
            result = attempt_become_internal(p)
            total_bonded += result
            total_to_internal += result
        else:
            became_edge, got_broken = attempt_recruit_from_internal(p)
            total_broken += got_broken
            total_to_edge += became_edge
    end = time.perf_counter()

    print(f"Created {total_bonded} bonds and broke {total_broken} bonds, in {end - start} sec. "
          f"{total_to_edge} became edge; {total_to_internal} became internal")
    
def _relax(relaxation_saturation_factor: float, viscosity: float) -> None:
    def relax_bond(bhandle: tf.BondHandle, r0: float, r: float, viscosity: float,
                   p1: tf.ParticleHandle, p2: tf.ParticleHandle
                   ) -> None:
        """Relaxing a bond means to partially reduce the energy (hence the generated force) by changing
        the r0 toward the current r.

        viscosity: how much relaxation per timestep. In range [0, 1].
            v = 0 is completely elastic (no change to r0, ever; so if a force is applied that stretches the bond, and
                then released, the bond will recoil and tend to shrink back to its original length)
            v = 1 is completely plastic (r0 instantaneously takes the value of r; so if a force is applied that
                stretches the bond, and then released, there will be no recoil at all)
            0 < v < 1 means r0 will change each timestep, but only by that fraction of the difference (r-r0). So bonds
                will always be under some tension, but the longer a bond remains stretched, the less recoil there will
                be if the force is released.
        """
        # Because existing bonds can't be modified, we destroy it and replace it with a new one, with new properties
        gc.destroy_bond(bhandle)
        
        delta_r0: float
        saturation_distance: float = relaxation_saturation_factor * Little.radius
        if r > r0 + saturation_distance:
            delta_r0 = viscosity * saturation_distance
        elif r < r0 - saturation_distance:
            # ToDo: this case is completely wrong; it needs its own, different saturation_distance.
            # Minor issue but should fix this.
            delta_r0 = viscosity * -saturation_distance
        else:
            delta_r0 = viscosity * (r - r0)
        new_r0: float = r0 + delta_r0

        k: float = cfg.harmonic_edge_spring_constant if is_edge_bond(p1, p2) else cfg.harmonic_spring_constant
        potential: tf.Potential = tf.Potential.harmonic(r0=new_r0,
                                                        k=k,
                                                        max=cfg.max_potential_cutoff
                                                        )
        gc.create_bond(potential, p1, p2, new_r0)
    
    assert 0 <= viscosity <= 1, "viscosity out of bounds"
    assert relaxation_saturation_factor > 0, "relaxation_saturation_factor out of bounds"
    if viscosity == 0:
        # 0 is the off-switch. No relaxation. (Calculations work, but are just an expensive no-op.)
        return
    
    bhandle: tf.BondHandle
    gcdict: dict[int, gc.BondData] = gc.bonds_by_id

    print(f"Relaxing all {len(tf.BondHandle.items())} bonds")
    for bhandle in tf.BondHandle.items():
        # future: checking .active is not supposed to be needed; those are supposed to be filtered out before you
        # see them. Possibly the flag may not even be accessible in future versions.
        if bhandle.active:
            assert bhandle.id in gcdict, "Bond data missing from global catalog!"
            p1: tf.ParticleHandle
            p2: tf.ParticleHandle
            p1, p2 = tfu.bond_parts(bhandle)
            bond_data: gc.BondData = gcdict[bhandle.id]
            r0: float = bond_data["r0"]
            # print(f"r0 = {r0}")
            r: float = p1.distance(p2)
            
            relax_bond(bhandle, r0, r, viscosity, p1, p2)
            
@profile
def _move_toward_open_space(k_particle_diffusion: float) -> None:
    """Prevent gaps from opening up by giving particles a nudge to move toward open space.
    
    This should prevent the situation where particles look for potential bonding partners but can't find one
    in the direction in which they are most needed (the direction of a hole). For the moment, this applies only
    to internal particles.
    """
    phandle: tf.ParticleHandle
    for phandle in Little.items():
        neighbor: tf.ParticleHandle
        if any([neighbor.type_id == LeadingEdge.id for neighbor in phandle.getBondedNeighbors()]):
            # Can't add diffusion force for particles bound to leading edge, or they'll go careening into that
            # open space. Particularly particles immediately after transforming from edge to internal type.
            phandle.force_init = [0, 0, 0]
        else:
            bonded_neighbor_positions: list[tf.fVector3] = \
                [neighbor.position for neighbor in phandle.getBondedNeighbors()]
            vecsum: tf.fVector3 = sum(bonded_neighbor_positions, start=tf.fVector3([0, 0, 0]))
            centroid: tf.fVector3 = vecsum / len(bonded_neighbor_positions)
            force: tf.fVector3 = (phandle.position - centroid) * k_particle_diffusion
            phandle.force_init = force.as_list()
        

def maintain_bonds(k_neighbor_count: float = 0.4, k_angle: float = 2,
                   k_edge_neighbor_count: float = 2, k_edge_angle: float = 2,
                   k_particle_diffusion: float = 20,
                   relaxation_saturation_factor: float = 2, viscosity: float = 0) -> None:
    _make_break_or_become(k_neighbor_count, k_angle,
                          k_edge_neighbor_count, k_edge_angle, verbose=False)
    _move_toward_open_space(k_particle_diffusion)
    _relax(relaxation_saturation_factor, viscosity)
    
    # Notes on parameters: with relaxation disabled (viscosity=0), k_particle_diffusion=20 works well.
    # When relaxation is enabled (viscosity=0.001), surprisingly, we get more holes, not fewer. A higher
    # value of k_particle_diffusion seems to be needed, then. 40 works okay, might try a little higher,
    # but 50 was way too much and produced instability. Also, a higher viscosity may be needed because
    # the recoil after disabling the external force seems like still too much.
    #
    # To be revisited later after I reconsider/retool the particle diffusion algorithm.

_profiled_break: bool = True
_profiled_make: bool = False
_profiled_recruit: bool = True
_profiled_become: bool = True

def profile_finished() -> bool:
    return _profiled_become and _profiled_recruit and _profiled_make and _profiled_break
