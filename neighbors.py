"""neighbors.py - neighbors module (for now, this is based on operations within EVL)"""

import math
import time

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import utils.epiboly_utils as epu
import utils.global_catalogs as gc
import utils.tf_utils as tfu

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

def get_non_bonded_neighbors(phandle: tf.ParticleHandle,
                             ptypes: list[tf.ParticleType],
                             distance_factor: float) -> list[tf.ParticleHandle]:
    """Return list of neighbors, but excluding any that the particle is already bonded to
    
    (Sort of the inverse of particleHandle.bonded_neighbors.)
    
    distance_factor: search out to this multiple of cell radius (not particle radius). But, its the INITIAL
        cell radius, not the radius of the individual cell. Because we want all searches to be over the same
        distance, regardless of cell size, because small cells (their centers) will generally be nearer
        than large ones. Defining search space by individual cell radius would therefore bias the search on
        smaller cells, to find other smaller cells.
    """
    neighbors: tf.ParticleList
    non_bonded_neighbors: list[tf.ParticleHandle]
    
    search_distance: float = distance_factor * epu.initial_cell_radius
    neighbors = phandle.neighbors(search_distance, ptypes)
    non_bonded_neighbors = [neighbor for neighbor in neighbors
                            if neighbor not in phandle.bonded_neighbors]
    return non_bonded_neighbors

def get_nearest_non_bonded_neighbors(phandle: tf.ParticleHandle,
                                     ptypes: list[tf.ParticleType] = None,
                                     min_neighbors: int = 1,
                                     min_distance: float = 1.0) -> list[tf.ParticleHandle]:
    """Use an iterative approach to search over larger and larger distances until you find enough non-bonded neighbors
    
    ptypes: list of allowed particle types to search for
    min_neighbors: search until at least this many are found
    min_distance: search out to at least this multiple of radius (cell radius, not particle radius)
    Starts the search at the specified distance, and proceeds outward until the specified number of neighbors is found.
    Thus, the returned list will satisfy both minimums.
    
    Can return empty list if none found, but
    with this iterative approach to distance_factor, it seems this never happens.
    You can always find non-bonded neighbors, long before hitting the max allowable distance.
    
    NOTE: min_neighbors may be 0. In this case we still iterate the search at least once, to satisfy
    the distance criterion. This is the one situation in which this function may return an empty list.
    The caller is indicating that it's okay not to find any, if there are none within min_distance.
    """
    if ptypes is None:
        ptypes = [g.LeadingEdge, g.Evl]

    start: float = time.perf_counter()

    neighbors: list[tf.ParticleHandle]
    distance_factor: float = min_distance
    # Huge maximum that should never be reached, just insurance against a weird infinite loop:
    max_distance_factor: float = cfg.max_potential_cutoff / gc.get_cell_radius(phandle)
    while True:
        # Get all neighbors not already bonded to, of the specified types, within the given radius. (There may be none.)
        neighbors = get_non_bonded_neighbors(phandle, ptypes, distance_factor)
        
        distance_factor += 1
        
        if len(neighbors) >= min_neighbors or distance_factor > max_distance_factor:
            break
    
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

def get_nearest_non_bonded_neighbors_constrained(phandle: tf.ParticleHandle,
                                                 ptypes: list[tf.ParticleType] = None,
                                                 min_neighbors: int = 1,
                                                 max_neighbors: int = 1) -> list[tf.ParticleHandle]:
    """Find an exactly specified number of nearest neighbors
    
    Will return a number of neighbors between min_neighbors and max_neighbors, inclusive (where max ≥ min).
    """
    if max_neighbors < min_neighbors:
        max_neighbors = min_neighbors
    neighbors: list[tf.ParticleHandle] = get_nearest_non_bonded_neighbors(phandle, ptypes, min_neighbors)
    neighbors.sort(key=phandle.distance)
    return neighbors[:max_neighbors]

def get_nearest_non_bonded_neighbor(phandle: tf.ParticleHandle,
                                    ptypes: list[tf.ParticleType] = None) -> tf.ParticleHandle | None:
    """Find the nearest non-bonded neighbor
    
    Can return None, but with the iterative approach to search distance, it seems this never happens.
    You can always find a nearest non-bonded neighbor, long before hitting the max allowable distance.
    """
    neighbors: list[tf.ParticleHandle] = get_nearest_non_bonded_neighbors(phandle, ptypes, min_neighbors=1)
    nearest_neighbor: tf.ParticleHandle = min(neighbors, key=phandle.distance, default=None)
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

def bonds_to_neighbors_of_types(p: tf.ParticleHandle, ptypes: list[tf.ParticleType]) -> list[tf.BondHandle]:
    """Note. As of TF v. 0.1.0, p.bonds MAY contain some phantom bonds.
    
    As a workaround, this code was changed from using .bonds, to the more indirect method of using
    .bonded_neighbors, and getting the bonds from that. If only needing the length of the result and not
    the bonds themselves, just use bonded_neighbors_of_types() instead. (I.e., use count_neighors_of_types().)
    """
    neighbor: tf.ParticleHandle
    return [tfu.bond_between(p, neighbor)
            for neighbor in bonded_neighbors_of_types(p, ptypes)]

def bonded_neighbors_of_types(p: tf.ParticleHandle, ptypes: list[tf.ParticleType]) -> list[tf.ParticleHandle]:
    """This function uses p.bonded_neighbors, which is not subject to the TF phantom-bonds bug."""
    neighbor: tf.ParticleHandle
    return [neighbor for neighbor in p.bonded_neighbors
            if neighbor.type() in ptypes]

def count_neighbors_of_types(p: tf.ParticleHandle, ptypes: list[tf.ParticleType]) -> int:
    return len(bonded_neighbors_of_types(p, ptypes))

def get_ordered_neighbors_and_angles(p: tf.ParticleHandle,
                                     neighbors: tf.ParticleList | list[tf.ParticleHandle]
                                     ) -> list[tuple[tf.ParticleHandle, float]]:
    """Return a sorted list of neighbors and angles (relative to a reference vector)
    
    Returned list is sorted in order of those relative angles, so that iterating over the result would trace
    a simple closed polygon around particle p.
    
    Uses the vegetalward direction from p as the reference vector. Therefore, the returned list starts with the most
    vegetalward neighbor on one side of vertical, and ends with the most vegetalward neighbor on the other side.
    
    neighbors: list of neighbors to sort (really any set of particles, not including p)
    return: list of tuples, in which the first element is the neighbor, and the second element is the angle that
        neighbor forms with the vegetalward direction, in the range [0..2pi).
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
                return 2 * math.pi - theta
    
        normal_vector: tf.fVector3 = epu.embryo_cartesian_coords(p)
        reference_cross: tf.fVector3 = tfu.cross(reference_vector, normal_vector)
        corrected_angles: list[float] = [corrected_angle(theta, neighbor_unit_vectors[i], reference_cross)
                                         for i, theta in enumerate(original_angles)]
        return corrected_angles

    neighbor: tf.ParticleHandle
    neighbor_unit_vectors: list[tf.fVector3] = [(neighbor.position - p.position).normalized()
                                                for neighbor in neighbors]
    reference_vector: tf.fVector3 = epu.vegetalward(p)
    
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
    return sorted_tuples

def get_ordered_bonded_neighbors_and_angles(p: tf.ParticleHandle,
                                            extra_neighbor: tf.ParticleHandle = None
                                            ) -> list[tuple[tf.ParticleHandle, float]]:
    """Get bonded neighbors of p, and their angles (relative to a reference vector), sorted on those angles
    
    Calls get_ordered_neighbors_and_angles(); see that function for details.
    Selects specifically the bonded neighbors of p (plus extra_neighbor) to sort.
    
    extra_neighbor: a particle not currently bonded to p, but for which a bond might be created. So that we
        can get its order relative to the existing bonds.
    return: list of tuples, in which the first element is the neighbor (all bonded neighbors, plus extra_neighbor),
        and the second element is the angle that neighbor forms with the vegetalward direction, in the range [0..2pi).
    """
    neighbors: tf.ParticleList = getBondedNeighbors(p)
    if extra_neighbor:
        assert extra_neighbor not in neighbors, f"Extra neighbor id={extra_neighbor.id} is already bonded"
        neighbors.insert(extra_neighbor)
        
    return get_ordered_neighbors_and_angles(p, neighbors)

def get_ordered_bonded_neighbors(p: tf.ParticleHandle,
                                 extra_neighbor: tf.ParticleHandle = None
                                 ) -> list[tf.ParticleHandle]:
    """Get bonded neighbors of p, ordered according to their relative angles
    
    Pass-through. See get_ordered_bonded_neighbors_and_angles() for details. From the result, discards
    the angles and just returns the neighbors.
    
    extra_neighbor: a particle not currently bonded to p, but for which a bond might be created. So that we
        can get its order relative to the existing bonds.
    return: sorted list of neighbors
    """
    # pass-through, get the ordered list of neighbors, and their angles from vegetalward
    neighbors_and_angles: list[tuple[tf.ParticleHandle, float]]
    neighbors_and_angles = get_ordered_bonded_neighbors_and_angles(p, extra_neighbor)
    
    # Return only the neighbor particles
    return [tup[0] for tup in neighbors_and_angles]

def paint_neighbors():
    """Test of neighbors() functionality by painting neighbors different colors"""
    # Get the two sets of particles. There should be about 2200, and a bit over 100, respectively. Note
    # these two lists are live. Instead of assigning to a variable, make a new list from each of them,
    # that's not live? Thought it might make a difference in memory management. Doesn't seem to help, though.
    internal_particles = tf.ParticleList(g.Evl.items())
    edge_particles = tf.ParticleList(g.LeadingEdge.items())
    print("internal, edge particles contain:", len(internal_particles), len(edge_particles))
    print(internal_particles.thisown)
    
    internal_step = round(len(internal_particles) / 15)
    edge_step = round(len(edge_particles) / 10)
    
    found_color = tfu.white
    neighbor_color = tfu.gray
    found_particles = []
    # Iterate over each list with a step, to pick a small subset of particles
    for i in range(0, len(internal_particles), internal_step):
        found_particles.append(internal_particles.item(i))
    
    for i in range(0, len(edge_particles), edge_step):
        found_particles.append(edge_particles.item(i))
    
    for p in found_particles:
        
        # Set color on particle
        p.style.color = found_color
        
        # Find the neighbors of this particle
        neighbors = p.neighbors(cfg.min_neighbor_initial_distance_factor)
        
        for neighbor in neighbors:
            if neighbor not in found_particles:
                neighbor.style.color = neighbor_color
