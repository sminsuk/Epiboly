"""neighbors module (for now, this is based on operations within EVL)"""

import math
import time
from typing import Optional

import tissue_forge as tf
import epiboly_globals as g
from utils import tf_utils as tfu
import config as cfg

def getBondedNeighbors(p: tf.ParticleHandle) -> tf.ParticleList:
    """Just a wrapper. Cruft.
    
    Was previously used while beta testing, in order to be able to run in either v0.0.1, or in later versions.
    It included a version test and a different syntax for the older version. Support for 0.0.1 has now been removed.
    This is called from a LOT of places, so leaving in place, at least for now.
    ToDO? Maybe remove opportunistically as I come across them, until they're gone.
    """
    # Property introduced (in 0.0.2, I think), to replace function p.getBondedNeighbors(), to address issues,
    # though not documented in the docs (yet?)
    return p.bonded_neighbors

def find_neighbors(p: tf.ParticleHandle, distance_factor: float, sort: bool = False) -> list[tf.ParticleHandle]:
    """Find neighbors of particle p
    
    distance_factor: search out to this multiple of particle radius
    sort: if True, return results ordered by increasing distance from p.
    """
    # Get all the particles within the threshold distance of p.
    neighbors = p.neighbors(distance_factor * p.radius, [].extend([g.Little, g.LeadingEdge]))
    if sort:
        neighbors = sorted(neighbors, key=lambda neighbor: neighbor.distance(p))
    return neighbors

def get_non_bonded_neighbors(phandle: tf.ParticleHandle,
                             distance_factor: float, sort: bool = False) -> list[tf.ParticleHandle]:
    """Return list of neighbors, but excluding any that the particle is already bonded to
    
    (Sort of the inverse of particleHandle.getBondedNeighbors().)
    
    distance_factor: search out to this multiple of particle radius
    sort: return results ordered by increasing distance from particle.
    """
    my_bonded_neighbor_ids: list[int]
    neighbors: list[tf.ParticleHandle]
    non_bonded_neighbors: list[tf.ParticleHandle]
    
    # Who am I already bonded to?
    my_bonded_neighbor_ids = [neighbor.id for neighbor in getBondedNeighbors(phandle)]

    # Who are all my neighbors? (bonded or not)
    neighbors = find_neighbors(phandle, distance_factor, sort)
    non_bonded_neighbors = [neighbor for neighbor in neighbors
                            if neighbor.id not in my_bonded_neighbor_ids]
    return non_bonded_neighbors

def get_nearest_non_bonded_neighbors(phandle: tf.ParticleHandle,
                                     ptypes: list[tf.ParticleType] = None,
                                     min_neighbors: int = 1,
                                     min_distance: float = 1.0) -> list[tf.ParticleHandle]:
    """Use an iterative approach to search over larger and larger distances until you find enough non-bonded neighbors
    
    ptypes: list of allowed particle types to search for
    min_neighbors: search until at least this many are found
    min_distance: search out to at least this multiple of radius
    Starts the search at the specified distance, and proceeds outward until the specified number of neighbors is found.
    Thus, the returned list will satisfy both minimums.
    
    Can return empty list if none found, but
    with this iterative approach to distance_factor, it seems this never happens.
    You can always find non-bonded neighbors, long before hitting the max allowable distance.
    """
    ptype: tf.ParticleType
    if ptypes is None:
        ptypes = [g.LeadingEdge, g.Little]
    type_ids: list[int] = [ptype.id for ptype in ptypes]

    start: float = time.perf_counter()

    neighbors: list[tf.ParticleHandle] = []
    distance_factor: float = min_distance
    # Huge maximum that should never be reached, just insurance against a weird infinite loop:
    max_distance_factor: float = cfg.max_potential_cutoff / g.Little.radius
    while len(neighbors) < min_neighbors and distance_factor < max_distance_factor:
        # Get all neighbors not already bonded to, within the given radius. (There may be none.)
        neighbors = get_non_bonded_neighbors(phandle, distance_factor)
        
        # Exclude unwanted neighbor types:
        # (After next release, this might work without ids, just test for the type in ptypes)
        neighbors = [neighbor for neighbor in neighbors
                     if neighbor.type_id in type_ids]
        
        distance_factor += 1
    
    # Note on performance-tuning of this neighbor-finding algorithm, by modifying the increment on distance_factor
    # at the end of the while loop: both extremes caused significantly worse results: With increment of 19, big
    # slow-down, because searching very far, even though particles are found in either the first or second
    # iteration. With increment of 0.05, also big slow-down, because lots of iterations needed to find particles,
    # even though the distance of the search is kept to a minimum. But, difference between using 1 and 0.5
    # was not large enough to distinguish from noise, and was a definite speed-up over using a single search
    # (no loop) with a distance_factor of 5. Probably there is an optimum that could be discovered with more
    # careful peformance profiling.
    
    elapsed: float = time.perf_counter() - start
    # print(f"Neighbor-finding time = {elapsed}, final distance_factor = {distance_factor}")

    return neighbors

def get_nearest_non_bonded_neighbor(phandle: tf.ParticleHandle,
                                    ptypes: list[tf.ParticleType] = None) -> Optional[tf.ParticleHandle]:
    """Find the nearest non-bonded neighbor
    
    Can return None (hence "Optional" in typing of function return value), but
    with the iterative approach to search distance, it seems this never happens.
    You can always find a nearest non-bonded neighbor, long before hitting the max allowable distance.
    """
    neighbors: list[tf.ParticleHandle] = get_nearest_non_bonded_neighbors(phandle, ptypes, min_neighbors=1)
    nearest_neighbor: tf.ParticleHandle = min(neighbors, key=lambda neighbor: phandle.distance(neighbor), default=None)
    return nearest_neighbor

def get_shared_bonded_neighbors(p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> list[tf.ParticleHandle]:
    """If there are none, returns empty list"""
    phandle: tf.ParticleHandle
    p1_ids: list[int] = [phandle.id for phandle in getBondedNeighbors(p1)]
    shared_neighbors: list[tf.ParticleHandle] = [phandle for phandle in getBondedNeighbors(p2)
                                                 if phandle.id in p1_ids]
    return shared_neighbors

def distance(p: tf.ParticleHandle, neighbor1: tf.ParticleHandle, neighbor2: tf.ParticleHandle) -> float:
    """Get total distance of p from its two neighbors"""
    return p.distance(neighbor1) + p.distance(neighbor2)

def bonds_to_neighbors_of_type(p: tf.ParticleHandle, ptype: tf.ParticleType) -> list[tf.BondHandle]:
    bhandle: tf.BondHandle
    return [bhandle for bhandle in p.bonds
            if tfu.other_particle(p, bhandle).type_id == ptype.id]

def count_neighbors_of_type(p: tf.ParticleHandle, ptype: tf.ParticleType) -> int:
    return len(bonds_to_neighbors_of_type(p, ptype))

def get_ordered_bonded_neighbors(p: tf.ParticleHandle,
                                 extra_neighbor: tf.ParticleHandle = None) -> list[tf.ParticleHandle]:
    """Get bonded neighbors, ordered according to their relative angles, so that iterating over the result
    would trace a simple closed polygon around particle p
    
    extra_neighbor: a particle not currently bonded, but for which a bond might be created. So that we
        can get its order relative to the existing bonds.
    """
    
    def disambiguate(original_angles: list[float],
                     neighbor_unit_vectors: list[tf.fVector3],
                     reference_vector: tf.fVector3,
                     p: tf.ParticleHandle) -> list[float]:
        """Use a cross product to generate assymetry, followed by dot products to determine which angles
        are on which side of the polygon.

        Original angles are in the range [0, π], so there's ambiguity over which side of the polygon
        they are on. Simply ordering by that angle would therefore trace both sides of the polygon in
        parallel, meeting up at the opposite side. We need to instead trace down one side of the polygon
        and back up the other side, to the point where we started."""
        def corrected_angle(theta: float, neighbor_vec: tf.fVector3, reference_cross: tf.fVector3) -> float:
            """Angle in range [0, π], corrected to range [0, 2π), depending which side of the reference vector it is on

            neighbor_vec: points from the particle to one of its neighbors
            theta: angle between the given vector, and the reference vector. (In the range [0, π].)
            reference_cross: a vector pointing to one side of the polygon, perpendicular to the reference vector.
            """
            # It appears that doing the projection is not actually needed, and getting rid of it gives maybe
            # a 10% speed-up. Delete this once I'm sure it works (no weirdness at angles close to pi/2).
            # projection: tf.fVector3 = neighbor_vec.projected(reference_cross)
            # dotprod: float = projection.dot(reference_cross)
            dotprod: float = neighbor_vec.dot(reference_cross)
            if dotprod >= 0:
                # > 0: vector angle < pi/2; particle on the same side as reference_cross;
                # == 0: vector angle == pi/2 (neighbor_vec perpendicular to reference_cross);
                #       either neighbor_vec is the reference_vector, or it's pointing exactly 180 deg from it.
                return theta
            else:
                # < 0: vector angle > pi/2; particle on the opposite side from reference_cross;
                return cfg.two_pi - theta
    
        big_particle: tf.ParticleHandle = g.Big.items()[0]
        normal_vector: tf.fVector3 = p.position - big_particle.position
        reference_cross: tf.fVector3 = tfu.cross(reference_vector, normal_vector)
        corrected_angles: list[float] = [corrected_angle(theta, neighbor_unit_vectors[i], reference_cross)
                                         for i, theta in enumerate(original_angles)]
        return corrected_angles

    neighbors: tf.ParticleList = getBondedNeighbors(p)
    neighbors_id_list = [neighbor.id for neighbor in neighbors]   # #### Until next release lets me fix the following
    if extra_neighbor:
        # doesn't work (until next release?)
        # assert extra_neighbor not in neighbors, f"Extra neighbor id={extra_neighbor.id} is already bonded"
        # Try this instead:
        assert extra_neighbor.id not in neighbors_id_list, f"Extra neighbor id={extra_neighbor.id} is already bonded"
        neighbors.insert(extra_neighbor)
        
    if len(neighbors) < 4:
        # It does not matter what order you traverse these. The existing order is fine.
        return list(neighbors)

    neighbor_unit_vectors: list[tf.fVector3] = [(position - p.position).normalized()
                                                for position in neighbors.positions]
    reference_vector: tf.fVector3 = neighbor_unit_vectors[0]
    # Note: speed-ups I tried, all skipping the function call and doing the work right here instead:
    # 1) control: do it here with a single list comprehension (just no func call overhead);
    # 2) just get the dotprods in a list comprehension, then pass the list to numpy.arccos;
    # 3) same, but use math.acos() here; i.e. 2 separate list comprehensions
    # 4) Using angle proxies here in this function, instead of the actual angles, avoiding the use of acos().
    #       Surprisingly, no apparent speed-up at all!
    # 5) Substituted a table look-up for calling math.acos() (how the hell is this not faster?)
    # None of these had any appreciable effect, so sticking with the function call; math.acos() inside.
    angles_to_reference: list[float] = [tfu.angle_from_unit_vectors(uvec, reference_vector)
                                        for uvec in neighbor_unit_vectors]

    corrected_angles: list[float] = disambiguate(angles_to_reference,
                                                 neighbor_unit_vectors,
                                                 reference_vector,
                                                 p)
    
    # To do the final sort, must package up the particleHandle and the angle and sort them together
    neighbors_and_angles_to_reference: list[tuple[tf.ParticleHandle, float]] = list(zip(neighbors, corrected_angles))
    sorted_tuples: list[tuple[tf.ParticleHandle, float]] = sorted(neighbors_and_angles_to_reference,
                                                                  key=lambda tup: tup[1])
    return [tup[0] for tup in sorted_tuples]

def paint_neighbors():
    """Test of neighbors() functionality by painting neighbors different colors"""
    # Get the two sets of particles. There should be about 2200, and a bit over 100, respectively. Note
    # these two lists are live. Instead of assigning to a variable, make a new list from each of them,
    # that's not live? Thought it might make a difference in memory management. Doesn't seem to help, though.
    little_particles = tf.ParticleList(g.Little.items())
    edge_particles = tf.ParticleList(g.LeadingEdge.items())
    print("little, edge particles contain:", len(little_particles), len(edge_particles))
    print(little_particles.thisown)
    
    little_step = round(len(little_particles) / 15)
    edge_step = round(len(edge_particles) / 10)
    
    found_color = tfu.white
    neighbor_color = tfu.gray
    found_particles = []
    # Iterate over each list with a step, to pick a small subset of particles
    for i in range(0, len(little_particles), little_step):
        found_particles.append(little_particles.item(i))
    
    for i in range(0, len(edge_particles), edge_step):
        found_particles.append(edge_particles.item(i))
    
    for p in found_particles:
        
        # Set color on particle
        p.style.color = found_color
        
        # Find the neighbors of this particle
        neighbors = find_neighbors(p, distance_factor=cfg.min_neighbor_initial_distance_factor)
        
        for neighbor in neighbors:
            if neighbor not in found_particles:
                neighbor.style.color = neighbor_color
