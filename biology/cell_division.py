"""cell_division.py - Cell division"""
import matplotlib.pyplot as plt
import numpy as np

import tissue_forge as tf
import epiboly_globals as g

import config as cfg
import biology.bond_maintenance as bonds
import utils.global_catalogs as gc
import utils.tf_utils as tfu

# Initialize just once
_generator: np.random.Generator = np.random.default_rng()
_expected_timesteps: int = 29000 if cfg.space_filling_enabled else 23000
_expected_divisions_per_timestep: float = cfg.total_epiboly_divisions / _expected_timesteps
_cumulative_cell_divisions: int = 0

def divide(parent: tf.ParticleHandle) -> tf.ParticleHandle:
    daughter: tf.ParticleHandle = parent.split()
    parent.radius = g.Little.radius
    parent.mass = g.Little.mass
    
    daughter.radius = parent.radius
    daughter.mass = parent.mass
    daughter.style = tf.rendering.Style()
    daughter.style.color = tfu.lighter_blue     # for now, change it
    gc.add_particle(daughter)

    bond_count: int = len(daughter.bonds)
    if bond_count > 0:
        print(tfu.bluecolor + f"New particle has {bond_count} bonds before any have been made!" + tfu.endcolor)

    bonds.make_all_bonds(daughter)
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
            (Idea for maybe later: make this depend on leading edge displacement per timestep, instead of on time.
            Then it would no longer be an "expected" value, but would instead be derived from an actual on-the-fly
            measurement representing epiboly progress. Average division rate would then drift over time, as
            the parameter sent to the poisson function changes. This should result in a more consistent total
            number of divisions over the course of the sim. -- Also, decoupling cell division rate from absolute
            time, and coupling it instead to rate of leading edge advancement, would have the beneficial effect of
            preventing division from happening when there's no external force to generate epiboly.)
    9. Use Poisson to determine how many cells will actually divide this time.
    """
    global _cumulative_cell_divisions
    num_divisions: int = _generator.poisson(lam=_expected_divisions_per_timestep)
    if num_divisions <= 0:
        return
        
    # Select the particles to split
    phandle: tf.ParticleHandle
    
    p_list: list[tf.ParticleHandle] = [phandle for phandle in g.Little.items()]
    selected_particles: np.ndarray = _generator.choice(p_list, size=num_divisions, replace=False)
    
    daughter: tf.ParticleHandle
    for phandle in selected_particles:
        _cumulative_cell_divisions += 1
        daughter = divide(phandle)
        print(f"New cell division (cumulative: {_cumulative_cell_divisions}), {len(daughter.bonds)} new bonds")

def get_state() -> dict:
    """generate state to be saved to disk"""
    return {"cumulative": _cumulative_cell_divisions}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved."""
    global _cumulative_cell_divisions
    _cumulative_cell_divisions = d["cumulative"]

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
