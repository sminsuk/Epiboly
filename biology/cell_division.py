"""cell_division.py - Cell division"""
import math
import matplotlib.pyplot as plt
import numpy as np

import tissue_forge as tf
import epiboly_globals as g

import config as cfg
import biology.bond_maintenance as bonds
import utils.epiboly_utils as epu
import utils.global_catalogs as gc
import utils.tf_utils as tfu

# Initialize just once
_generator: np.random.Generator = np.random.default_rng()
_cumulative_cell_divisions: int = 0

# For calibrating division rate to EVL area.
# Because surface area of a slice of a sphere = 2πrh, area increase will be proportional to height increase.
_expected_divisions_per_height_unit: float = 0
_evl_previous_z: float = 0
_cell_division_cessation_phi: float = 0

def initialize_division_rate_tracking_by_evl_area() -> None:
    """Calibrate division rate to the changes in EVL area
    
    Call this once, after simulation setup and equilibration, but before any additional timesteps
    
    Initially, used timestep calibration (figure out how many cell divisions per timestep, in order
    to have the desired number by the end of the simulation). Then switched to area calibration (figure out
    how many cell divisions for a given incremental increase in surface area of the EVL, in order to have
    the desired number by the time the leading edge reaches a given location).
    
    Timestep calibration:
    - wasn't universal; needed magic numbers for every different case;
    - was only approximate, because the total number of timesteps in the sim isn't known until the sim is finished;
    
    Area calibration:
    - should be universal; won't need to be tweaked every time I change a parameter of the sim;
    - should be perfectly tuned because the area increase is always the same;
    """
    global _expected_divisions_per_height_unit, _evl_previous_z, _cell_division_cessation_phi
    
    if not cfg.cell_division_enabled:
        return
    
    embryo_radius: float = g.Big.radius + g.Little.radius
    embryo_height: float = 2 * embryo_radius
    evl_initial_height: float = embryo_height * cfg.epiboly_initial_percentage / 100
    cell_division_cessation_height: float = embryo_height * cfg.cell_division_cessation_percentage / 100
    evl_total_height_increase: float = cell_division_cessation_height - evl_initial_height

    _cell_division_cessation_phi = epu.phi_for_epiboly(epiboly_percentage=cfg.cell_division_cessation_percentage)

# Todo
#  #### This whole section should now be deprecated - as long as cells get smaller on splitting, they should always fit!
#  #### BUT: Can I use this calculation to automate the determination of what cell radius should be for a given number
#  #### of cells at initialization? Should be a similar calculation.
    # Calculate whether the increased area (from epiboly_initial_percentage to cell_division_cessation_percentage)
    # is enough to accommodate the requested number of particles without crowding.
    # This deals with the fact that particles have a fixed size and if density is high, adding new
    # particles will result in repulsion that drives further area increase. We don't want that because:
    # (perhaps three ways of saying the same thing)
    # 1. we only want area increase to be driven by explicitly modeled forces
    # 2. we want division to reflect area increase, not the other way around: one-way causality
    # 3. it's a positive feedback: area increase -> new particles -> area increase
    # The control (balanced forces) illustrates the problem because it should have ~0 area increase, hence
    # 0 or near 0 cell division, but that's not the case when cfg.total_epiboly_divisions is too large.
    # Given the configured geometry (particle radii and epiboly_initial_percentage) at the time of this
    # commit, this calculation results in a maximum of 3,064 cell divisions over the course of full epiboly.
    #   Note to self: this also assumes the initial setup is well-packed but not over-packed. That was tuned
    # manually by adjusting particle size to work well with the desired number of particles, but if I ever need to
    # change that config, this now provides an algorithm to tune the particle radius / particle count automatically.
    
    # some geometry:
    area_to_height_ratio = 2 * np.pi * embryo_radius  # from area of spherical segment = 2 pi R h
    circumscribed_hexagon_ratio = 2 * np.sqrt(3) / np.pi  # area ratio of hexagon circumscribed around a circle

    # Would the area occupied by the requested particles in a hexagonal packing, fit in the EVL area increase?
    total_area_increase: float = evl_total_height_increase * area_to_height_ratio
    particle_area: float = np.pi * (g.Little.radius ** 2)  # area occupied by rendered particle itself
    circumscribed_hexagon_area: float = particle_area * circumscribed_hexagon_ratio
    elbow_room_factor: float = 1.0  # For now. We may need fudge factor > 1 to realistically avoid crowding.
    particle_footprint: float = elbow_room_factor * circumscribed_hexagon_area  # approx. because based on plane
    new_particle_capacity: int = math.floor(total_area_increase / particle_footprint)
    
    # Throttle to the number of particles that can fit
    print(f"Requested divisions: {cfg.total_epiboly_divisions}; capacity: {new_particle_capacity}")
    total_epiboly_divisions: int = min(cfg.total_epiboly_divisions, new_particle_capacity)
# #### ^ End of area that's totally deprecated now ^

    _expected_divisions_per_height_unit = total_epiboly_divisions / evl_total_height_increase
    _evl_previous_z = epu.leading_edge_mean_z()

def _adjust_positions(p1: tf.ParticleHandle, p2: tf.ParticleHandle) -> None:
    """Tissue Forge particle splitting is randomly oriented, but we need to constrain the axis to within the sheet"""
    r1: float
    r2: float
    theta1: float
    theta2: float
    phi1: float
    phi2: float
    
    # Particle rather than ParticleHandle in this case, since that's what sphericalPosition() requires
    yolk_particle: tf.Particle = g.Big.particle(0)
    
    r1, theta1, phi1 = p1.sphericalPosition(particle=yolk_particle)
    r2, theta2, phi2 = p2.sphericalPosition(particle=yolk_particle)
    while theta1 == theta2 and phi1 == phi2:
        # In the unlikely event that the division axis was exactly radial (relative to the yolk cell),
        # add a small random displacement to one of the particles until the two particles are no longer
        # aligned that way. Subsequent operation requires that there be some angle between them, to prevent
        # them from ending up in exactly the same location.
        magnitude: float = p1.radius / 10
        displacement: tf.fVector3 = tf.random_unit_vector() * magnitude
        p1.position += displacement
        r1, theta1, phi1 = p1.sphericalPosition(particle=yolk_particle)
        
    if r1 != r2:
        # Keep both particles within the layer. One is too close to the yolk center, one is too far away.
        corrected_r = g.Big.radius + g.Little.radius
        relative_cartesian1: tf.fVector3 = tfu.cartesian_from_spherical([corrected_r, theta1, phi1])
        relative_cartesian2: tf.fVector3 = tfu.cartesian_from_spherical([corrected_r, theta2, phi2])
        yolk_phandle: tf.ParticleHandle = yolk_particle.handle()
        p1.position = yolk_phandle.position + relative_cartesian1
        p2.position = yolk_phandle.position + relative_cartesian2
        
def _alt_split(parent: tf.ParticleHandle) -> tf.ParticleHandle:
    """This should work, but currently does not. Keep it around and unused until we get it worked out
    
    Instead of dividing the particle in any random direction, and then having to adjust the positions
    of the daughters, calculate a randomm direction within the tangent plane, and use the second overload
    of ParticleHandle.split() to provide that direction.
    
    Also, to get the daughters to be right next to each other, double the volume (and also the mass) of the
    particle before splitting, and the splitting operation will then halve them, giving daughters of the
    same dimension as the parent, placed adjacent to one another.
    """
    # Note: .split() reduces mass and volume both to half the parent's original value. So, double both of them,
    # so that we get daughter particles of the same mass and volume as the parent.
    # Double the particle volume (by multiplying the particle radius by cube root 2).
    # This also means the daughters will be placed just touching, so we don't have to adjust their locations afterward.
    parent.radius *= 2 ** (1 / 3)
    parent.mass *= 2
    
    # Note that split() takes a "direction" argument, but it's not recognized as a keyword argument
    daughter: tf.ParticleHandle = parent.split(epu.random_tangent(parent))
    assert daughter.radius == g.Little.radius, f"resulting radius = {daughter.radius}"
    assert daughter.mass == g.Little.mass, f"resulting mass = {daughter.mass}"

    return daughter

def _split(parent: tf.ParticleHandle) -> tf.ParticleHandle:
    """Original approach. Divide in random directions then adjust the daughters to how I want them."""
    daughter: tf.ParticleHandle = parent.split()
    parent.radius = g.Little.radius
    parent.mass = g.Little.mass
    daughter.radius = parent.radius
    daughter.mass = parent.mass
    # Note: .split() reduces mass and volume both to half the parent's original value (so radius to 1/cube_rt(2)
    # or about 0.8), and then places the parent and the daughter next to each other, and just touching.
    # But once you override that by changing that radius, of course you lose the "just touching". These
    # particles will now be overlapping, and moved in arbitrary directions (usually in/out from the yolk surface).
    
    # This will at least bring them back to the yolk surface. They're still likely overlapping but that will resolve.
    _adjust_positions(parent, daughter)
    
    return daughter

def _divide(parent: tf.ParticleHandle) -> tf.ParticleHandle:
    daughter: tf.ParticleHandle
    if cfg.use_alt_cell_splitting_method:
        daughter = _alt_split(parent)
    else:
        daughter = _split(parent)
    
    daughter.style = tf.rendering.Style()       # Not inherited from parent, so create it
    daughter.style.color = tfu.lighter_blue     # for now, change it
    gc.add_particle(daughter, radius=gc.get_cell_radius(parent))

    bond_count: int = len(daughter.bonded_neighbors)
    if bond_count > 0:
        print(tfu.bluecolor + f"New particle has {bond_count} bonds before any have been made!" + tfu.endcolor)

    bonds.make_all_bonds(daughter)  # ToDo: This needs a bug fix, never took into account cfg.max_edge_neighbor_count
    return daughter

def cell_division() -> None:
    """Cell division
    
    Since the rate of division will be extremely low, don't loop over all the particles. Instead, decide how
    many particles will divide during this timestep, and select them at random. This should be much more
    computationally efficient.
    
    With area-based rate calibration, number of divisions in any given timestep wil be derived from an actual
    on-the-fly measurement representing epiboly progress. Average division rate (divisions per unit time) will
    thus drift over time, as the parameter sent to the poisson function changes. This should result in a consistent
    total number of divisions over the course of the sim, and it should not depend on any other algorithmic option
    settings or parameters. -- Also, decoupling cell division rate from absolute time, and coupling it instead to
    rate of leading edge advancement, should have the beneficial effect of preventing division from happening when
    there's no external force to generate epiboly.
    
    Use Poisson to determine how many cells will actually divide this time.
    """
    global _cumulative_cell_divisions, _evl_previous_z
    if not cfg.cell_division_enabled:
        return
    
    if epu.leading_edge_mean_phi() > _cell_division_cessation_phi:
        return
        
    # Note that z coordinate *decreases* as epiboly progresses
    evl_current_z: float = epu.leading_edge_mean_z()
    delta_z: float = _evl_previous_z - evl_current_z
    if delta_z <= 0:
        return
    
    expected_divisions: float = _expected_divisions_per_height_unit * delta_z
    num_divisions: int = _generator.poisson(lam=expected_divisions)
    
    _evl_previous_z = evl_current_z
        
    assert num_divisions >= 0, f"Poisson result = {num_divisions}, This should NEVER happen!"
    if num_divisions == 0:
        return
    
    # Select the particles to split
    phandle: tf.ParticleHandle
    particles: list[tf.ParticleHandle] = [phandle for phandle in g.Little.items()]
    selected_particles: np.ndarray
    if cfg.cell_division_biased_by_tension:
        # a particle's probability of being selected should be proportional to the tension it is under
        strains: list[float] = [tfu.strain(phandle) for phandle in particles]
        relative_probabilities: np.ndarray = np.clip(strains, a_min=0.0, a_max=None)
        if cfg.tension_squared:
            relative_probabilities = np.square(relative_probabilities)
        p_normalized: np.ndarray = relative_probabilities / np.sum(relative_probabilities)
        selected_particles = _generator.choice(particles, size=num_divisions, replace=False, p=p_normalized)
    else:
        # all particles have a uniform probability of being selected
        selected_particles = _generator.choice(particles, size=num_divisions, replace=False)
    
    daughter: tf.ParticleHandle
    tf.Logger.setLevel(level=tf.Logger.INFORMATION)  # Not using DEBUG because that gets tons of internal TF Debug msgs
    for phandle in selected_particles:
        # ### DEBUG: For cells with small x, i.e. cells near the yz coordinate plane (relative to yolk center as origin)
        # ###           either screen them out (don't split), or paint them to identify them. Either way, log it.
        screen: bool = False    # Config: If True, screen them out, don't paint anything (paint flag ignored)
        paint: bool = False     # Config: If NOT screening, whether to paint only these daughters
        tolerance: float = 0.2
        pos: tf.fVector3 = epu.embryo_cartesian_coords(phandle)
        if abs(pos.x()) < tolerance:
            if screen:
                msg = f"Skipping split of particle at position = {pos}"
                tf.Logger.log(level=tf.Logger.INFORMATION, msg=msg)
                print(tfu.bluecolor, msg, tfu.endcolor)
                continue
            elif paint:
                msg = f"Painting daughter pair from parent at position = {pos}"
                tf.Logger.log(level=tf.Logger.INFORMATION, msg=msg)
                print(tfu.bluecolor, msg, tfu.endcolor)
        else:
            paint = False       # Don't change this one. Never paint daughters from > tolerance parent
        # ### END DEBUG

        _cumulative_cell_divisions += 1
        daughter = _divide(phandle)
        assert daughter.type_id == g.Little.id, f"Daughter is of type {daughter.type()}!"
        print(f"New cell division (cumulative: {_cumulative_cell_divisions},"
              f" total cells: {len(g.Little.items()) + len(g.LeadingEdge.items())}),"
              f" daughter id={daughter.id}, {daughter.type()}, {len(daughter.bonded_neighbors)} new bonds")
        
        # ### DEBUG: Color the cells so I can see where they end up
        if paint:
            phandle.style.color = tfu.light_gray
            daughter.style.color = tfu.green
        # ### END DEBUG
    tf.Logger.setLevel(level=tf.Logger.ERROR)

def get_state() -> dict:
    """generate state to be saved to disk"""
    return {"cumulative_cell_divisions": _cumulative_cell_divisions,
            "expected_divisions_per_height_unit": _expected_divisions_per_height_unit,
            "evl_previous_z": _evl_previous_z,
            "cell_division_cessation_phi": _cell_division_cessation_phi,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global _cumulative_cell_divisions, _expected_divisions_per_height_unit, _evl_previous_z
    global _cell_division_cessation_phi
    _cumulative_cell_divisions = d["cumulative_cell_divisions"]
    _expected_divisions_per_height_unit = d["expected_divisions_per_height_unit"]
    _evl_previous_z = d["evl_previous_z"]
    _cell_division_cessation_phi = d["cell_division_cessation_phi"]

def _timestep_tracking_for_tests() -> tuple[int, float]:
    expected_timesteps: int = 22000  # original length of the full sim before cell division was implemented
    total_epiboly_divisions: int = 7500  # original estimate before I started adjusting it
    expected_divisions_per_timestep = total_epiboly_divisions / expected_timesteps
    return expected_timesteps, expected_divisions_per_timestep

def _test1() -> None:
    """Testing whether this does what I want.

    Is the random.Generator.poisson() func actually a python generator, so I can just call it once with
    size=None (default), each time I want a value? Yes, that's the case!
    """
    print("lam = 2.5: ", end="")
    for _ in range(20):
        print(f"{_generator.poisson(2.5)}, ", end="")
    print()
    
def _test2() -> None:
    """With a tiny lambda does it generate the distribution I expect? It does!"""
    foo, expected_divisions_per_timestep = _timestep_tracking_for_tests()
    lam: float = expected_divisions_per_timestep
    results: np.ndarray = _generator.poisson(lam=lam, size=35)
    print(f"lam = {round(lam, 3)}, length = {len(results)}, type = {type(results[0])}, mean = {np.mean(results)}")
    print(results)

def _test3(size: int = None) -> None:
    """With a larger output array, does the mean come closer to the target?
    
    It does! I expect to call it once per timestep, so note the results with size = expected_timesteps
    (the original length of the full sim before cell division was implemented).
    """
    expected_timesteps, expected_divisions_per_timestep = _timestep_tracking_for_tests()
    if size is None:
        size = expected_timesteps
    
    lam: float = expected_divisions_per_timestep
    results: np.ndarray = _generator.poisson(lam=lam, size=size)
    print(f"lam = {round(lam, 3)}, length = {len(results)}, mean = {np.mean(results)}")
    
def _test4() -> None:
    """What does this distribution really look like?
    
    Just like it should! Log scale on second plot so that it's possible to see the bars for the very rare values.
    """
    expected_timesteps, expected_divisions_per_timestep = _timestep_tracking_for_tests()
    lam: float = expected_divisions_per_timestep
    results: np.ndarray = _generator.poisson(lam=lam, size=expected_timesteps)
    print(f"Histogram of {expected_timesteps} values; first linear, then log:")
    print(f"max value = {max(results)}, total count = {sum(results)}")
    plt.hist(results,
             bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             align="left",
             rwidth=0.9)
    plt.show()
    count, bins, ignored = plt.hist(results,
                                    bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    align="left",
                                    rwidth=0.9,
                                    log=True)
    print(f"count: {count}")
    plt.show()

if __name__ == '__main__':
    """Run some tests to check my own understanding"""
    
    _test1()
    _test1()
    _test1()
    print()
    _test2()
    _test2()
    _test2()
    _test3(200)
    _test3(200)
    _test3(200)
    _test3(2000)
    _test3(2000)
    _test3(2000)
    _test3()
    _test3()
    _test3()
    print()
    _test4()
