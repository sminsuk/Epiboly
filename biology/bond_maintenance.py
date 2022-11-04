"""Handle the remodeling of the bond network as the tissue changes shape"""
import math
import random
import time

import tissue_forge

from epiboly_init import *
import config as cfg
from utils import tf_utils as tfu,\
    global_catalogs as gc

import neighbors as nbrs

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
    # r0: float = p1.distance(p2)
    r0: float = 2 * Little.radius
    potential: tf.Potential = tf.Potential.harmonic(r0=r0,
                                                    k=cfg.harmonic_spring_constant,
                                                    max=cfg.max_potential_cutoff
                                                    )
    handle: tf.BondHandle = gc.make_bond(potential, p1, p2, r0)

    if verbose:
        distance: float = p1.distance(p2)
        print(f"Making new bond {handle.id} between particles {p1.id} and {p2.id},",
              f"distance = (radius * {distance/Little.radius})")
        p1.style.setColor("lightgray")  # testing
        p2.style.setColor("white")  # testing

def make_all_bonds(phandle: tf.ParticleHandle, verbose=False) -> int:
    # Bond to all neighbors not already bonded to
    neighbors = nbrs.get_non_bonded_neighbors(phandle)
    for neighbor in neighbors:
        _make_bond(neighbor, phandle, verbose)
    return len(neighbors)

def _attempt_closest_bond(phandle: tf.ParticleHandle, making_search_distance: float,
                          making_prob_dropoff: float, max_prob: float, verbose=False) -> int:
    """making_search_distance is for the exponential algorithm; max_prob is for the linear algorithm"""
    # Get all neighbors not already bonded to, within a certain fairly permissive radius. Bond to at most one.
    neighbors: list[tf.ParticleHandle] = nbrs.get_non_bonded_neighbors(phandle, distance_factor=making_search_distance,
                                                                       sort=False)
    # Three ways to do this, none really really giving me what I want...
    # Note that the one that needs sorting is slow, which is why I tried the other two, but they are all equally
    # slow. Turns out that's not the problem. I traced the slowdown to the call to get_non_bonded_neighbors(); it
    # slows down greatly when you call it with a wider area search. With the default search area of 1.5, this
    # is all essentially as fast as make_all_bonds().
    # random_neighbor: tf.ParticleHandle = None if not neighbors else random.choice(neighbors)
    # closest_neighbor: tf.ParticleHandle = None if not neighbors else neighbors[0]     # For this, set sort=True!
    closest_neighbor: tf.ParticleHandle = min(neighbors, key=lambda neighbor: phandle.distance(neighbor), default=None)
    neighbor = closest_neighbor
    bonded = False
    if neighbor:
        # probability falls off with distance
        r0: float = 2 * Little.radius
        distance: float = phandle.distance(neighbor)
        # bonding_probability: float = math.exp(-(distance - r0)/making_prob_dropoff)
        
        # alternative way: linear
        # at distance = r0, probability = 1; at distance = (making_search_distance +1) * r0, probability = 0. So:
        m: float = -max_prob / ((making_search_distance - 1) * Little.radius)
        b: float = max_prob * (making_search_distance + 1) / (making_search_distance - 1)
        bonding_probability = m * distance + b
        assert 0 <= bonding_probability <= b,\
            f"Bonding probability calculation isn't right." \
            f" m, b, distance, probability = {(m, b, distance, bonding_probability)}"
        
        if random.random() < bonding_probability:
            if verbose:
                print(f"distance = {distance}, probability = {bonding_probability}, max_prob = {max_prob}")
            _make_bond(neighbor, phandle, verbose)
            bonded = True
    return 1 if bonded else 0

def _break(breaking_saturation_factor: float, max_prob: float) -> None:
    """Decide which bonds will be broken, then break them.
    
    saturation_factor: multiple of r0 at which probability of breaking = max_prob
    max_prob: the max value that probability of breaking ever reaches. In range [0, 1].
    """
    alert_once = True
    
    def breaking_probability(bhandle: tf.BondHandle, r0: float, r: float) -> float:
        """Probability of breaking bond"""
        nonlocal alert_once
        potential: tf.Potential = bhandle.potential
        saturation_distance: float = breaking_saturation_factor * r0
        saturation_energy: float = potential(saturation_distance)
        
        # Note, in extreme cases, i.e. after alert_once has been tripped, this assert will fail, but is meaningless.
        # potential(r) should match the bond's energy property, though it won't be exact:
        assert abs(bhandle.energy - potential(r)) < 0.0001, \
            f"unexpected bond energy: property = {bhandle.energy}, calculated = {potential(r)}"
    
        p: float
        if r <= r0:
            p = 0
        elif r > saturation_distance:
            p = max_prob
        elif saturation_energy == 0:
            # This is to trap the error that would otherwise occur in extreme cases; if this happens, most likely
            # r0 has gotten so big that saturation_distance is well beyond potential.max, hence
            # potential() evaluates to 0. Just go ahead and break the bond.
            # "alert_once" means once per event invocation.
            p = max_prob
            if alert_once:
                alert_once = False
                print(tfu.bluecolor + "WARNING: potential.max exceeded" + tfu.endcolor)
        else:
            p = max_prob * bhandle.energy / saturation_energy
            
        return p
    
    assert 0 <= max_prob <= 1, "max_prob out of bounds"
    assert breaking_saturation_factor > 0, "breaking_saturation_factor out of bounds"

    breaking_bonds: list[tf.BondHandle] = []
    bhandle: tf.BondHandle
    gcdict: dict[int, gc.BondData] = gc.bonds_by_id

    print(f"Evaluating all {len(tf.BondHandle.items())} bonds, to maybe break")
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
            
            if random.random() < breaking_probability(bhandle, r0, r):
                breaking_bonds.append(bhandle)

    print(f"breaking {len(breaking_bonds)} bonds: {[bhandle.id for bhandle in breaking_bonds]}")
    for bhandle in breaking_bonds:
        gc.break_bond(bhandle)
        
def _make_break_or_become(search_distance: float, k: float, verbose: bool = False) -> None:
    """
    search_distance: the maximum distance within which to make new bonds
    k: lambda of the neighbor-count constraint, like lambda of the Potts model volume constraint, but "lambda" is
        python reserved, so call it k by analogy with k of the harmonic potential, which is basically the same formula.
    verbose: controls the printing of rejected operations only. printing of accepted operations always goes.
    """

    def accept(p1: tf.ParticleHandle, p2: tf.ParticleHandle, breaking: bool) -> bool:
        """Decide whether the bond between these two particles may be made/broken
        
        breaking: if True, decide whether to break a bond; if False, decide whether to make a new one
        """
        bonded_neighbor_ids: list[int] = [phandle.id for phandle in p2.getBondedNeighbors()]
        if breaking:
            assert p1.id in bonded_neighbor_ids,\
                f"Attempting to break bond between non-bonded particles: {p1.id}, {p2.id}"
        else:
            assert p1.id not in bonded_neighbor_ids,\
                f"Attempting to make bond between already bonded particles: {p1.id}, {p2.id}"
            
        p1current_count: int = len(p1.bonds)
        p2current_count: int = len(p2.bonds)
        delta_count: int = -1 if breaking else 1
        p1final_count: int = p1current_count + delta_count
        p2final_count: int = p2current_count + delta_count

        # Neither particle may go below the minimum threshold for number of bonds
        if p1final_count < 3 or p2final_count < 3:
            if verbose:
                print(f"Rejecting break because particles have {p1current_count} and {p2current_count} bonds")
            return False

        # Simple for now, but this will probably get more complex later. I think LeadingEdge target count
        # needs to gradually increase as edge approaches vegetal pole, because of the geometry (hence float).
        p1target_count: float = (6 if p1.type_id == Little.id else 4)
        p2target_count: float = (6 if p2.type_id == Little.id else 4)
    
        p1current_energy: float = (p1current_count - p1target_count) ** 2
        p2current_energy: float = (p2current_count - p2target_count) ** 2
        
        p1final_energy: float = (p1final_count - p1target_count) ** 2
        p2final_energy: float = (p2final_count - p2target_count) ** 2

        delta_energy: float = (p1final_energy + p2final_energy) - (p1current_energy + p2current_energy)
    
        if delta_energy <= 0:
            return True
        else:
            probability: float = math.exp(-k * delta_energy)
            if random.random() < probability:
                return True
            else:
                if verbose:
                    print(f"Rejecting {'break' if breaking else 'make'} because unfavorable; particles have"
                          f" {p1current_count} and {p2current_count} bonds")
                return False

    def attempt_break_bond(p: tf.ParticleHandle) -> int:
        """For internal, break any bond; for leading edge, break any bond to an internal particle
        
        returns: number of bonds broken
        """
        bhandle: tf.BondHandle
        breakable_bonds: list[tf.BondHandle] = p.bonds
        if p.type_id == LeadingEdge.id:
            # Don't break bond between two LeadingEdge particles
            breakable_bonds = [bhandle for bhandle in breakable_bonds
                               if tfu.other_particle(p, bhandle).type_id == Little.id]
        # select one at random to break:
        bhandle = random.choice(breakable_bonds)
        other_p: tf.ParticleHandle = tfu.other_particle(p, bhandle)
        if accept(p, other_p, breaking=True):
            print(f"Breaking bond {bhandle.id} between particles {p.id} and {other_p.id}")
            gc.break_bond(bhandle)
            expiration: float = tf.Universe.time + cfg.bonding_ban_duration * tissue_forge.Universe.dt
            gcdict: dict[int, gc.ParticleData] = gc.particles_by_id
            gcdict[p.id]["blacklisted_ids"][other_p.id] = expiration
            gcdict[other_p.id]["blacklisted_ids"][p.id] = expiration
            return 1
        return 0
    
    def attempt_make_bond(p: tf.ParticleHandle) -> int:
        """For internal, bond to the closest unbonded neighbor (either type); for leading edge, bond to
        the closest unbonded *internal* neighbor only.
        
        returns: number of bonds created
        """
        start: float = time.perf_counter()
        
        neighbors: list[tf.ParticleHandle] = []
        distance_factor: float = 1
        gcdict: dict[int, gc.ParticleData] = gc.particles_by_id
        blacklisted_ids: dict[int, float] = gcdict[p.id]["blacklisted_ids"]
        while not neighbors and distance_factor < cfg.max_distance_factor:
            # Get all neighbors not already bonded to, within the given radius. (There may be none.)
            # Don't bond to a neighbor you broke a bond with, unless its banning has expired.
            # Don't make a bond between two LeadingEdge particles
            neighbors = nbrs.get_non_bonded_neighbors(p, distance_factor)
            neighbors = [neighbor for neighbor in neighbors
                         if p.type_id == Little.id or neighbor.type_id == Little.id
                         if (neighbor.id not in blacklisted_ids
                             or blacklisted_ids[neighbor.id] < tissue_forge.Universe.time)]
            distance_factor += 1
            
        # Note on performance-tuning of this neighbor-finding algorithm, by modifying the increment on distance_factor
        # at the end of the while loop: either extreme caused significantly worse results: With increment of 19, big
        # slow-down, because searching very far, even though particles are found in either the first or second
        # iteration. With increment of 0.05, also big slow-down, because lots of iterations needed to find particles,
        # even though the distance of the search is kept to a minimum. But, difference between using 1 and 0.5
        # was not large enough to distinguish from noise, and was a definite speed-up over using a single search
        # (no loop) with a distance_factor of 5. Probably there is an optimum that could be discovered with more
        # careful peformance profiling.
        
        other_p: tf.ParticleHandle = min(neighbors, key=lambda neighbor: p.distance(neighbor), default=None)
        elapsed: float = time.perf_counter() - start
        # print(f"Neighbor-finding time = {elapsed}, final distance_factor = {distance_factor}")
        
        if not other_p:
            # With the iterative approach to distance_factor, it seems this never happens
            if verbose:
                print("Can't make bond: No particles available")
            return 0
        if accept(p, other_p, breaking=False):
            _make_bond(p, other_p, verbose=True)
            return 1
        return 0
    
    def attempt_become_internal(p: tf.ParticleHandle) -> int:
        """For LeadingEdge particles only. Become internal, and let its two bonded leading edge neighbors
        bond to one another.
        
        This MAKES a bond.
        returns: number of bonds created
        """
        # Temporary, until this is implemented
        return attempt_make_bond(p)
    
    def attempt_recruit_from_internal(p: tf.ParticleHandle) -> int:
        """For LeadingEdge particles only. Break the bond with one bonded leading edge neighbor, but only
        if there is an internal particle bonded to both of them. That internal particle becomes leading edge.
        If there are more than one such particle, pick the one with the shortest combined path.
        
        This BREAKS a bond.
        returns: number of bonds broken
        """
        # Temporary, until this is implemented
        return attempt_break_bond(p)
    
    assert k > 0 and search_distance > 0, f"Both args must be positive; k = {k}, search_distance = {search_distance}"
    total_bonded: int = 0
    total_broken: int = 0
    p: tf.ParticleHandle
    
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
            total_bonded += attempt_become_internal(p)
        else:
            total_broken += attempt_recruit_from_internal(p)
            
    print(f"Created {total_bonded} bonds and broke {total_broken} bonds.")
    
    
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
        gc.break_bond(bhandle)
        
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
        
        potential: tf.Potential = tf.Potential.harmonic(r0=new_r0,
                                                        k=cfg.harmonic_spring_constant,
                                                        max=6
                                                        )
        gc.make_bond(potential, p1, p2, new_r0)
    
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

def maintain_bonds_deprecated(
        making_search_distance: float = 5, making_prob_dropoff: float = 0.01, making_max_prob: float = 1e-4,
        breaking_saturation_factor: float = 3, max_prob: float = 0.001,
        relaxation_saturation_factor: float = 2, viscosity: float = 0) -> None:
    total: int = 0
    for ptype in [Little, LeadingEdge]:
        for p in ptype.items():
            total += _attempt_closest_bond(p, making_search_distance, making_prob_dropoff, making_max_prob,
                                           verbose=True)
    print(f"Created {total} bonds.")

    _break(breaking_saturation_factor, max_prob)
    
    _relax(relaxation_saturation_factor, viscosity)
    
def maintain_bonds(search_distance: float = 5, k: float = 1,
                   relaxation_saturation_factor: float = 2, viscosity: float = 0) -> None:
    _make_break_or_become(search_distance, k, verbose=True)
    _relax(relaxation_saturation_factor, viscosity)
