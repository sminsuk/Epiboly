"""cell_division.py - Cell division"""
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

# For calibrating division rate to time.
_expected_timesteps: int = 0
_expected_divisions_per_timestep: float = 0

def initialize_division_rate_tracking() -> None:
    """Call this once, after simulation setup and equilibration, but before any additional timesteps
    
    Initially, used timestep calibration. Then switched to area calibration. But that's not entirely
    worked out yet, so, need to keep timestep calibration as an option for now. (Also, the tests use it.)
    
    Timestep calibration:
    - isn't universal; needs magic numbers for every different case;
    - is only approximate, because the total number of timesteps in the sim isn't known until the sim is finished;
    
    Area calibration:
    - should be universal (if I can get it right); won't need to be tweaked every time I change a parameter of the sim;
    - should be perfectly tuned because the area increase is always the same;
    """
    if cfg.cell_division_enabled:
        if cfg.calibrate_division_rate_to_timesteps:
            _initialize_timestep_tracking()
        else:
            _initialize_evl_area_tracking()
    
def _initialize_timestep_tracking() -> None:
    """Calibrate division rate to the passage of time"""
    global _expected_timesteps, _expected_divisions_per_timestep
    if cfg.cell_division_enabled:
        # For space_filling_enabled, value based on only N=2, because haven't been running it lately
        # (see results from 2023 Mar. 29, 30)
        _expected_timesteps = 10500 if cfg.space_filling_enabled else 8900
    else:
        # (This is a relic from when only the Poisson part had been written, and the rest of cell division
        # was only stubbed out. I used this to test the behavior of that Poisson functionality, reporting
        # each "division" and the cumulative total. Keeping for now.)
        _expected_timesteps = 29000 if cfg.space_filling_enabled else 22000
    _expected_divisions_per_timestep = cfg.total_epiboly_divisions / _expected_timesteps

def _initialize_evl_area_tracking() -> None:
    """Calibrate division rate to the changes in EVL area"""
    global _expected_divisions_per_height_unit, _evl_previous_z
    
    evl_final_height: float = 2 * (g.Big.radius + g.Little.radius)
    evl_initial_height: float = evl_final_height * cfg.epiboly_initial_percentage / 100
    evl_total_height_increase: float = evl_final_height - evl_initial_height
    _expected_divisions_per_height_unit = cfg.total_epiboly_divisions / evl_total_height_increase

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

def _divide(parent: tf.ParticleHandle) -> tf.ParticleHandle:
    daughter: tf.ParticleHandle = parent.split()
    parent.radius = g.Little.radius
    parent.mass = g.Little.mass
    # Note: .split() reduces mass and volume both to half the parent's original value (so radius to 1/cube_rt(2)
    # or about 0.8), and then places the parent and the daughter next to each other, and just touching.
    # But once you override that by changing that radius, of course you lose the "just touching". These
    # particles will now be overlapping at first, but they'll push each other apart soon enough.
    
    daughter.radius = parent.radius
    daughter.mass = parent.mass
    daughter.style = tf.rendering.Style()       # ToDo: test, is this needed, or inherits from parent?
    daughter.style.color = tfu.lighter_blue     # for now, change it
    gc.add_particle(daughter)
    
    _adjust_positions(parent, daughter)

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
    
    Justification for division rate:
    1. Linlin's single-embryo data set has 2252 nuclei at the first time point (4.02 hpf) and
        6580 nuclei at the final time point (8.77 hpf), for an increase of 4328.
    2. Assume that represents 4328 cell divisions during that time span.
    3. The position of the leading edge ("elevation" in radians above the equator) moves during that time from about
         +0.2 to about -0.7. Translating her "el" to my "phi" (in radians below the animal pole, phi = pi/2 - el),
         it moves from 1.37 to 2.27.
    4. My simulation starts at phi = 1.43 (don't take her absolute starting point literally, so I won't change this)
        and ends at phi = pi * 0.95 = 2.98.
    5. Thus the arc traversed by my sim, over the arc traversed by her data, is (2.98 - 1.43)/(2.27 - 1.37) = 1.72.
    6. If you assume (a gross oversimplification, but good enough) that divisions per unit time will be constant
        over the course of the simulation, then extrapolate from her 4328 to 4328 * 1.72 = 7,454 divisions.
        (Set in config.)
    7. A typical sim runs to completion in around 26000 timesteps (set in config).
    8. Expected divisions per timestep thus comes to around 0.29, on average.
        8A. However, perhaps better to make a more tailored assumption about the number of timesteps.
            When I ran this (with only a stub, no actual cell division, but monitoring how many divisions would
            be triggered), in a nearly 30K-timestep run, 8682 divisions occurred. This seems like a lot.
            Length of sim depends on whether I have the "space-filling" algorithm enabled. And whether I ultimately
            keep that, depends at least in port on the outcome of this cell-division implementation.
            In 4 runs with space filling DISabled, total timesteps = 21K, 22K, 21K, 28K.
            In 4 runs with space filling ENabled (diffusion coefficient = 40), total timesteps = 28K, 27K, 29K, 32K.
            So for now, do a calculation based on whether that is enabled or not.
        8B. But later, made this depend on leading edge displacement, instead of on time. So now it will no
            longer be a predetermined value per timestep, but will instead be derived from an actual on-the-fly
            measurement representing epiboly progress. Average division rate will thus drift over time, as the
            parameter sent to the poisson function changes. This should result in a more consistent total number
            of divisions over the course of the sim, and it should no longer depend on whether the space filling
            algorithm is enabled, nor on any other parameter. -- Also, decoupling cell division rate from absolute
            time, and coupling it instead to rate of leading edge advancement, should have the beneficial effect of
            preventing division from happening when there's no external force to generate epiboly.
        8C. Currently, I've made it an option which approach to use, while I get the subtleties worked out for
            area-based calibration.
    9. Use Poisson to determine how many cells will actually divide this time.
    """
    global _cumulative_cell_divisions, _evl_previous_z
    if not cfg.cell_division_enabled:
        return
    
    expected_divisions: float
    if cfg.calibrate_division_rate_to_timesteps:
        expected_divisions = _expected_divisions_per_timestep
    else:
        # Note that z coordinate *decreases* as epiboly progresses
        evl_current_z: float = epu.leading_edge_mean_z()
        if evl_current_z >= _evl_previous_z:
            return
        
        expected_divisions = _expected_divisions_per_height_unit * (_evl_previous_z - evl_current_z)
        _evl_previous_z = evl_current_z
    
    num_divisions: int = _generator.poisson(lam=expected_divisions)
    if num_divisions <= 0:
        return
        
    # Select the particles to split
    phandle: tf.ParticleHandle
    
    p_list: list[tf.ParticleHandle] = [phandle for phandle in g.Little.items()]
    selected_particles: np.ndarray = _generator.choice(p_list, size=num_divisions, replace=False)
    
    daughter: tf.ParticleHandle
    for phandle in selected_particles:
        _cumulative_cell_divisions += 1
        daughter = _divide(phandle)
        assert daughter.type_id == g.Little.id, f"Daughter is of type {daughter.type()}!"
        print(f"New cell division (cumulative: {_cumulative_cell_divisions},"
              f" total cells: {len(g.Little.items()) + len(g.LeadingEdge.items())}),"
              f" daughter id={daughter.id}, {daughter.type()}, {len(daughter.bonded_neighbors)} new bonds")

def get_state() -> dict:
    """generate state to be saved to disk"""
    return {"cumulative_cell_divisions": _cumulative_cell_divisions,
            "expected_divisions_per_height_unit": _expected_divisions_per_height_unit,
            "evl_previous_z": _evl_previous_z,
            "expected_timesteps": _expected_timesteps,
            "expected_divisions_per_timestep": _expected_divisions_per_timestep}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global _cumulative_cell_divisions, _expected_divisions_per_height_unit, _evl_previous_z
    global _expected_timesteps, _expected_divisions_per_timestep
    _cumulative_cell_divisions = d["cumulative_cell_divisions"]
    _expected_divisions_per_height_unit = d["expected_divisions_per_height_unit"]
    _evl_previous_z = d["evl_previous_z"]
    _expected_timesteps = d["expected_timesteps"]
    _expected_divisions_per_timestep = d["expected_divisions_per_timestep"]

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
    lam: float = _expected_divisions_per_timestep
    results: np.ndarray = _generator.poisson(lam=lam, size=35)
    print(f"lam = {round(lam, 3)}, length = {len(results)}, type = {type(results[0])}, mean = {np.mean(results)}")
    print(results)

def _test3(size: int) -> None:
    """With a larger output array, does the mean come closer to the target?
    
    It does! I expect to call it once per timestep, so note the results with size=_expected_timesteps, which
    represents the full sim.
    """
    lam: float = _expected_divisions_per_timestep
    results: np.ndarray = _generator.poisson(lam=lam, size=size)
    print(f"lam = {round(lam, 3)}, length = {len(results)}, mean = {np.mean(results)}")
    
def _test4() -> None:
    """What does this distribution really look like?
    
    Just like it should! Log scale on second plot so that it's possible to see the bars for the very rare values.
    """
    lam: float = _expected_divisions_per_timestep
    results: np.ndarray = _generator.poisson(lam=lam, size=_expected_timesteps)
    print(f"Histogram of {_expected_timesteps} values; first linear, then log:")
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
    _initialize_timestep_tracking()
    
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
    _test3(_expected_timesteps)
    _test3(_expected_timesteps)
    _test3(_expected_timesteps)
    print()
    _test4()
