"""Cell division"""
import matplotlib.pyplot as plt
import numpy as np

import tissue_forge as tf
import config as cfg

# Initialize just once
_generator: np.random.Generator = np.random.default_rng()
_expected_divisions_per_timestep: float = cfg.total_epiboly_divisions / cfg.expected_timesteps
_cumulative_cell_divisions: int = 0

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
    9. Use Poisson to determine how many cells will actually divide this time.
    """
    global _cumulative_cell_divisions
    num_divisions: float = _generator.poisson(lam=_expected_divisions_per_timestep)
    
    # stub:
    if num_divisions > 0:
        _cumulative_cell_divisions += num_divisions
        print(f"{num_divisions} cell(s) dividing. Cumulative: {_cumulative_cell_divisions}")

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
    
    It does! I expect to call it once per timestep, so note the results with size=26000, which is
    a typical number of timesteps in the full sim.
    """
    lam: float = _expected_divisions_per_timestep
    results: np.ndarray = _generator.poisson(lam=lam, size=size)
    print(f"lam = {round(lam, 3)}, length = {len(results)}, mean = {np.mean(results)}")
    
def _test4() -> None:
    """What does this distribution really look like?
    
    Just like it should! Log scale on second plot so that it's possible to see the bars for the very rare values.
    """
    lam: float = _expected_divisions_per_timestep
    results: np.ndarray = _generator.poisson(lam=lam, size=26000)
    print("Histogram of 26000 values; first linear, then log:")
    print(f"max value = {max(results)}")
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
    _test3(26000)
    _test3(26000)
    _test3(26000)
    print()
    _test4()