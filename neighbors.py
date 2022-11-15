"""neighbors module (for now, this is based on operations within EVL)"""

import math
import time
from typing import Optional

import tissue_forge as tf
import config as cfg
from epiboly_init import Little, LeadingEdge, Big

def _unshadowed_neighbors(p: tf.ParticleHandle, distance_factor: float) -> list[tf.ParticleHandle]:
    """Not nearest neighbors, but best neighbors.
    
    p: particle_handle
    distance_factor: when multiplied by the particle radius, gives the distance to search.
    returns: neighbors of p in a plain python list, ordered by increasing distance from p
    
    Bridges across gaps.
    Avoids having no neighbors in a particular direction.
    Avoids having multiple tiers of neighbors all in the same direction.
    """
    
    def is_in_shadow(candidate, neighbor, particle):
        """Is the candidate "hidden" from particle by the shadow cast by neighbor?"""
        
        # Where p, n, c stand for particle, neighbor, candidate, respectively:
        pn_vector = neighbor.position - p.position
        pn_distance = p.distance(neighbor)
        pc_distance = p.distance(candidate)
        cn_distance_ratio = pc_distance / pn_distance
        
        # and s stands for shadow:
        # center of shadow cone, at same distance as the candidate:
        ps_vector = pn_vector * cn_distance_ratio
        s_position = p.position + ps_vector
        s_radius = Little.radius * cn_distance_ratio
        
        # how far is the candidate from the shadow center?
        cx, cy, cz = candidate.position
        sx, sy, sz = s_position
        cs_distance = math.sqrt((sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2)
        return cs_distance < s_radius
        
        # Dropping down distance_factor to 1.5 got rid
    
    # of all the crossings (2.0 reduced the number, but did not eliminate).
    # And at this distance, I *still* got occasional false positives, which maybe are worse than
    # false negatives? So I guess I will stick with the simpler method for now, the native one.
    search_distance = Little.radius * distance_factor
    
    # Capture all potential neighbors within the region; ordered by distance from p (nearest neighbor last)
    candidates = p.neighbors(search_distance, [].extend([Little, LeadingEdge]))
    candidates = sorted(candidates, key=lambda candidate: candidate.distance(p), reverse=True)
    
    # testing
    #     print("particle.position = ", p.position.as_list())
    #     for candidate in candidates:
    #         print("distance =", candidate.distance(p))
    
    neighbors = []
    while candidates:
        # accept the nearest of the candidates as a neighbor
        neighbors.append(candidates.pop())
        
        # discard any remaining candidates that are in the shadow of an accepted neighbor
        surviving_candidates = []
        for candidate in candidates:
            shadowed = False
            for neighbor in neighbors:
                if is_in_shadow(candidate, neighbor, p):
                    shadowed = True
                    break
            if not shadowed:
                surviving_candidates.append(candidate)
        candidates = surviving_candidates
    
    # We now have the neighbors of particle p. These are ordered by increasing distance from p.
    return neighbors

def _native_neighbors(p: tf.ParticleHandle, distance_factor: float, sort: bool = False) -> list[tf.ParticleHandle]:
    """Native method of Tissue Forge for finding neighbors of particle p
    
    p: particle_handle
    distance_factor: when multiplied by the particle radius, gives the distance to search.
    sort: if True, return results ordered by increasing distance from p.
    returns: neighbors of p in a plain python list, ordered by increasing distance from p
    """
    # Get all the particles within the threshold distance of p.
    neighbors = p.neighbors(distance_factor * p.radius, [].extend([Little, LeadingEdge]))
    if sort:
        neighbors = sorted(neighbors, key=lambda neighbor: neighbor.distance(p))
    return neighbors

# noinspection PyUnreachableCode
def find_neighbors(p: tf.ParticleHandle, distance_factor: float = 1.5, sort: bool = False) -> list[tf.ParticleHandle]:
    """A central place to keep the decision of which neighbor algorithm to use, consistently throughout the program.
    
    p: particle_handle
    distance_factor: when multiplied by the particle radius, gives the distance to search.
        Default value works as basic threshold for making bonds while minimizing crossings.
    sort: if True, return results ordered by increasing distance from p. Applies only to _native_neighbors(),
        since sorting is non-optional in _unshadowed_neighbors().
    returns: neighbors of p in a plain python list. If sort == True, list is ordered by increasing distance from p
    """
    neighbors: list[tf.ParticleHandle]
    
    # Pick one of these methods (both return neighbors ordered by increasing distance from particle):
    if False:
        neighbors = _unshadowed_neighbors(p, distance_factor)
    else:
        neighbors = _native_neighbors(p, distance_factor, sort)
    return neighbors

def get_non_bonded_neighbors(phandle: tf.ParticleHandle,
                             distance_factor: float = 1.5, sort: bool = False) -> list[tf.ParticleHandle]:
    """Not quite the inverse of particleHandle.getBondedNeighbors()
    
    phandle: particleHandle
    distance_factor: when multiplied by the particle radius, gives the distance to search.
    sort: if True, return results ordered by increasing distance from p.
    returns: list of neighbors, using one of the neighbor algorithms in this module, but excluding any that
        the particle is already bonded to.
    """
    my_bonded_neighbor_ids: list[int]
    neighbors: list[tf.ParticleHandle]
    non_bonded_neighbors: list[tf.ParticleHandle]
    
    # Who am I already bonded to?
    my_bonded_neighbor_ids = [neighbor.id for neighbor in phandle.getBondedNeighbors()]
    
    # Who are all my neighbors? (bonded or not)
    neighbors = find_neighbors(phandle, distance_factor, sort)
    non_bonded_neighbors = [neighbor for neighbor in neighbors
                            if neighbor.id not in my_bonded_neighbor_ids]
    return non_bonded_neighbors

def get_nearest_non_bonded_neighbor(phandle: tf.ParticleHandle,
                                    ptypes: list[tf.ParticleType] = None) -> Optional[tf.ParticleHandle]:
    """Use an iterative approach to search over larger and larger distances until you find a non-bonded neighbor
    
    ptypes: list of allowed particle types to search for
    
    Can return None (hence "Optional" in typing of function return value), but
    with this iterative approach to distance_factor, it seems this never happens.
    You can always find a nearest non-bonded neighbor, long before hitting the max allowable distance.
    """
    ptype: tf.ParticleType
    if ptypes is None:
        ptypes = [LeadingEdge, Little]
    type_ids: list[int] = [ptype.id for ptype in ptypes]

    start: float = time.perf_counter()

    neighbors: list[tf.ParticleHandle] = []
    distance_factor: float = 1
    while not neighbors and distance_factor < cfg.max_distance_factor:
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
    
    nearest_neighbor: tf.ParticleHandle = min(neighbors, key=lambda neighbor: phandle.distance(neighbor), default=None)
    elapsed: float = time.perf_counter() - start
    # print(f"Neighbor-finding time = {elapsed}, final distance_factor = {distance_factor}")

    return nearest_neighbor

def get_ordered_bonded_neighbors(p: tf.ParticleHandle,
                                 extra_neighbor: tf.ParticleHandle = None) -> list[tf.ParticleHandle]:
    """Get bonded neighbors, ordered according to their relative angles, so that iterating over the result
    would trace a simple closed polygon around particle p
    
    extra_neighbor: a particle not currently bonded, but for which a bond might be created. So that we
        can get its order relative to the existing bonds.
    """
    
    def cross(v1: tf.fVector3, v2: tf.fVector3) -> tf.fVector3:
        return tf.fVector3([v1.y() * v2.z() - v1.z() * v2.y(),
                            v1.z() * v2.x() - v1.x() * v2.z(),
                            v1.x() * v2.y() - v1.y() * v2.x()])

    def truncate(dotprod: float) -> float:
        """Restrict dot product to the range [-1, 1]
        
        Dot products can be anything; but in this case (dot product of two unit vectors), result should never
        be outside this range, and the angle should be retrievable by taking acos(dot_product).
        If dot product is outside that range, acos() will throw an exception.
        
        This issue arises in particular when taking the dot product of a unit vector with itself, or when the two
        unit vectors are exactly 180 deg apart. These should come out to exactly +/- 1.0. But in these cases,
        tf.fVector3.dot() produces an imprecise result that can be too large, and this will crash acos().
        """
        if dotprod > 1.0:
            return 1.0
        elif dotprod < -1.0:
            return -1.0
        else:
            return dotprod
            
    def deprecated_disambiguate(original_angles: list[float],
                                neighbor_unit_vectors: list[tf.fVector3],
                                reference_vector: tf.fVector3) -> list[float]:
        """Use cross products to determine which angles are on which side of the polygon.
        
        Original angles are in the range [0, π], so there's ambiguity over which side of the polygon
        they are on. Simply ordering by that angle would therefore trace both sides of the polygon in
        parallel, meeting up at the opposite side. We need to instead trace down one side of the polygon
        and back up the other side, to the point where we started."""
        def normalized_cross(v1: tf.fVector3, v2: tf.fVector3) -> tf.fVector3:
            """Cross product (order matters!), normalized to a unit vector (when possible)

            returns: a unit vector in the correct direction, or, when cross product is 0, just that 0 vector

            Note: for angles close to (but not exactly) 180 deg, this seems to perform poorly, resulting in
            a vector that's off on some unexpected angle. However, close to 180 deg this may not matter that much.
            (See corrected_angle().)
            """
            result: tf.fVector3 = cross(v1, v2)
            if result.length() > 0:
                result = result.normalized()
            return result
    
        def corrected_angle(theta: float, reference_cross: tf.fVector3, cross: tf.fVector3) -> float:
            """Angle in range [0, π), corrected to range [0, 2π), depending which side of the reference vector it is on

            theta: angle between a given vector, and the reference vector. (In the range [0, π).)
            reference_cross: (normalized) cross product of arbitrarily selected neighbor vector, with the reference
                vector. (Possible magnitudes: 1 and 0. Possible direction: toward or away from the yolk center, roughly.)
            cross: (normalized) cross product of the reference vector with the given vector
            """
            vector_sum: tf.fVector3 = reference_cross + cross
            if vector_sum.length() > 0.5:
                # Either it's roughly 2 (the two cross products are pointing in the same direction);
                # or it's roughly 1 (the 2nd cross product is either reference_cross, or it's exactly 180 deg from it)
                # Note, for angles near (but not exactly) 180 deg, the cross product seems to be inaccurate,
                # but hopefully this doesn't matter that much, because at that angle, the difference between theta,
                # and (2π - theta), isn't that big.
                return theta
            else:
                # it's roughly 0, the two cross products are pointing in opposite directions
                return 2 * math.pi - theta
    
        crossprods: list[tf.fVector3] = [normalized_cross(reference_vector, uvec) for uvec in neighbor_unit_vectors]
        # item [0] is reference vector X itself and will be zero vector. All other items (unless exactly at 180 deg
        # from reference vector) will have magnitude of 1, and point either roughly away from, or roughly toward,
        # the center of the yolk. We select one for comparison, arbitrarily (as long as it's not at 0 or 180 deg
        # from the reference vector, i.e. as long as the cross is not the zero vec), and decide which side of the
        # circle each point is on, by deciding whether the cross product is pointing roughly the same direction
        # os this item, or the opposite direction.
        
        reference_cross: tf.fVector3
        for crossprod in crossprods:
            if crossprod.length() > 0:
                reference_cross = crossprod
                break
    
        # noinspection PyUnboundLocalVariable
        corrected_angles: list[float] = [corrected_angle(theta, reference_cross, crossprods[i])
                                         for i, theta in enumerate(original_angles)]
        return corrected_angles
    
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
            projection: tf.fVector3 = neighbor_vec.projected(reference_cross)
            dotprod: float = projection.dot(reference_cross)
            if dotprod >= 0:
                # > 0: the projection is pointing the same direction as the reference; particle on the same side;
                # == 0: the projection is the zero vector; either neighbor_vec is the reference_vector,
                #       or it's pointing exactly 180 deg from it.
                return theta
            else:
                # < 0: the projection is pointing the opposite direction from the reference;
                # particle on the opposite side,
                return 2 * math.pi - theta
    
        big_particle: tf.ParticleHandle = Big.items()[0]
        normal_vector: tf.fVector3 = p.position - big_particle.position
        reference_cross: tf.fVector3 = cross(reference_vector, normal_vector)
        corrected_angles: list[float] = [corrected_angle(theta, neighbor_unit_vectors[i], reference_cross)
                                         for i, theta in enumerate(original_angles)]
        return corrected_angles

    neighbors: tf.ParticleList = p.getBondedNeighbors()
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

    neighbor_coords: tuple[tf.fVector3] = neighbors.positions
    neighbor_vectors: list[tf.fVector3] = [coord - p.position for coord in neighbor_coords]
    neighbor_unit_vectors: list[tf.fVector3] = [vector.normalized() for vector in neighbor_vectors]
    reference_vector: tf.fVector3 = neighbor_unit_vectors[0]
    dotprods: list[float] = [uvec.dot(reference_vector) for uvec in neighbor_unit_vectors]
    angles_to_reference: list[float] = [math.acos(truncate(dotprod)) for dotprod in dotprods]

    # deprecated_corrected_angles: list[float] = deprecated_disambiguate(angles_to_reference,
    #                                                                    neighbor_unit_vectors,
    #                                                                    reference_vector)
    
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
    # Get all the small particles. There should be about 700, and 55, respectively. Note
    # these two lists are live. Instead of assigning to a variable, make a new list from each of them,
    # that's not live? Thought it might make a difference in memory management. Doesn't seem to help, though.
    littles = tf.ParticleList(Little.items())
    edge_lords = tf.ParticleList(LeadingEdge.items())
    print("littles, edge_lords contain:", len(littles), len(edge_lords))
    print(littles.thisown)
    
    littles_step = round(len(littles) / 7)  # pick about 8 interior particles
    edge_step = round(len(edge_lords) / 5)  # pick about 6 edge particles
    
    found_color = "white"
    neighbor_color = "lightgray"
    found_particles = []
    # Iterate over each list with a step, to pick a small subset of particles
    for i in range(0, len(littles), littles_step):
        found_particles.append(littles.item(i))
    
    for i in range(0, len(edge_lords), edge_step):
        found_particles.append(edge_lords.item(i))
    
    tested = False
    for p in found_particles:
        
        # Set color on particle
        p.style = tf.rendering.Style()
        p.style.setColor(found_color)
        
        # Find the neighbors of this particle
        neighbors = find_neighbors(p)
        
        for neighbor in neighbors:
            if neighbor not in found_particles:
                neighbor.style = tf.rendering.Style()
                neighbor.style.setColor(neighbor_color)
