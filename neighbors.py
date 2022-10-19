"""neighbors module (for now, this is based on operations within EVL)"""

import math

from epiboly_init import *

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
        
        # discard any remainding candidates that are in the shadow of an accepted neighbor
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

def _native_neighbors(p: tf.ParticleHandle, distance_factor: float) -> list[tf.ParticleHandle]:
    """Native method of Tissue Forge for finding neighbors of particle p
    
    p: particle_handle
    distance_factor: when multiplied by the particle radius, gives the distance to search.
    returns: neighbors of p in a plain python list, ordered by increasing distance from p
    """
    # Get all the particles within the threshold distance of p.
    neighbors = p.neighbors(distance_factor * p.radius, [].extend([Little, LeadingEdge]))
    neighbors = sorted(neighbors, key=lambda neighbor: neighbor.distance(p))
    return neighbors

# noinspection PyUnreachableCode
def find_neighbors(p: tf.ParticleHandle, distance_factor: float = 1.5) -> list[tf.ParticleHandle]:
    """A central place to keep the decision of which neighbor algorithm to use, consistently throughout the program.
    
    p: particle_handle
    distance_factor: when multiplied by the particle radius, gives the distance to search.
        Default value works as basic threshold for making bonds while minimizing crossings.
    returns: neighbors of p in a plain python list, ordered by increasing distance from p
    """
    neighbors: list[tf.ParticleHandle]
    
    # Pick one of these methods (both return neighbors ordered by increasing distance from particle):
    if False:
        neighbors = _unshadowed_neighbors(p, distance_factor)
    else:
        neighbors = _native_neighbors(p, distance_factor)
    return neighbors

def get_non_bonded_neighbors(phandle: tf.ParticleHandle) -> list[tf.ParticleHandle]:
    """Not quite the inverse of particleHandle.getBondedNeighbors()
    
    phandle: particleHandle
    returns: list of neighbors, using one of the neighbor algorithms in this module, but excluding any that
        the particle is already bonded to.
    """
    my_bonded_neighbor_ids: list[int]
    neighbors: list[tf.ParticleHandle]
    non_bonded_neighbors: list[tf.ParticleHandle]
    
    # Who am I already bonded to?
    my_bonded_neighbor_ids = [neighbor.id for neighbor in phandle.getBondedNeighbors()]
    
    # Who are all my neighbors? (bonded or not)
    neighbors = find_neighbors(phandle)
    non_bonded_neighbors = [neighbor for neighbor in neighbors
                            if neighbor.id not in my_bonded_neighbor_ids]
    return non_bonded_neighbors

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
