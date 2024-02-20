"""cell_division.py - Cell division"""
# import math
import matplotlib.pyplot as plt
import numpy as np

import tissue_forge as tf
import epiboly_globals as g

import config as cfg
import biology.bond_maintenance as bonds
import neighbors as nbrs
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
    
    embryo_height: float = 2 * epu.embryo_radius()
    evl_initial_height: float = embryo_height * cfg.epiboly_initial_percentage / 100
    cell_division_cessation_height: float = embryo_height * cfg.cell_division_cessation_percentage / 100
    evl_total_height_increase: float = cell_division_cessation_height - evl_initial_height

    _cell_division_cessation_phi = epu.phi_for_epiboly(epiboly_percentage=cfg.cell_division_cessation_percentage)

    print(f"Requested {cfg.total_epiboly_divisions} divisions,"
          f" by {cfg.cell_division_cessation_percentage}% epiboly, then stop")
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
        magnitude: float = p1.radius / 5
        displacement: tf.fVector3 = tf.random_unit_vector() * magnitude
        p1.position += displacement
        r1, theta1, phi1 = p1.sphericalPosition(particle=yolk_particle)
        
    if r1 != r2:
        # Keep both particles within the layer. One is too close to the yolk center, one is too far away.
        corrected_r = epu.embryo_radius()
        relative_cartesian1: tf.fVector3 = tfu.cartesian_from_spherical([corrected_r, theta1, phi1])
        relative_cartesian2: tf.fVector3 = tfu.cartesian_from_spherical([corrected_r, theta2, phi2])
        yolk_phandle: tf.ParticleHandle = yolk_particle.handle()
        p1.position = yolk_phandle.position + relative_cartesian1
        p2.position = yolk_phandle.position + relative_cartesian2
        
def _split(parent: tf.ParticleHandle) -> tf.ParticleHandle:
    """Divide the cell into two daughters, each with half the original apical surface area
    
    Depending on config variable, use the original approach, or the new-and-improved approach which
    needs a temporary workaround because of a Tissue Forge bug. Keep them both around until next TF version finalized:
    
    Original approach:
    
        Divide in random directions then adjust the daughters to how I want them.
    
        Resulting PARTICLES will have the same radius and mass as before, but the resulting CELLS
        will have smaller radii sufficient to result in an apical surface area half the original.
        
        This is intended for particles that are significantly smaller than their cells; cells are squamous and
        divide within the layer, so halving the volume also means halving the surface area; thickness doesn't change.
        
        Also, to place the daughter particles in the correct relative positions (not touching the way TF wants to do,
        but placed far enough apart for the CELLS to be touching), increase the volume of the particles before
        splitting, such that when TF halves that volume, we get the desired size and location based on cell radius
        (see detailed comment within).
        
        Also, treat the mass of the particles as a linear dimension, because it represents drag, which in a
        squamous epithelium is proportional to lateral surface area (excluding apical and basal), which in turn
        is proportional to circumference because width does not change. So increase it by a factor of sqrt(2)
        before splitting; TF will then halve that, for a net adjustment of sqrt(2)/2, just like for any other
        linear dimension like radius.
        
    New approach:
    
        Instead of dividing the particle in any random direction, and then having to adjust the positions
        of the daughters, calculate a random direction within the tangent plane (unless the particle is in
        the margin, then take the horizontal direction within the tangent plane), and use the second overload
        of ParticleHandle.split() to provide that direction. (For now, using a workaround for the bug in that
        overload in TF v. 0.2.0, to be fixed in the next release.)
    """
    # We trick Tissue Forge into generating the correct new size. It wants to make the descendants half the
    # volume of the original parent, but the particles are always spheres. So it reduces radius by a factor
    # of cube_rt(2). What we want is for the squamous surface area to be halved, so a radius reduction of sqrt(2).
    # And TF doesn't know about our cell radius, only our particle radius. We increase the size of our particle
    # to let it masquerade as our cell, by adopting the radius of the cell. But when TF halves the volume of the
    # spherical particle, its equatorial cross-sectional area (the cell squamous apical surface area) won't go down by
    # half, but only by 2**(2/3). So we need to adjust to a smaller particle to result in the right apical surface
    # area reduction. The math says this: if we set the particle radius prior to splitting, to the cell radius over
    # 2**(1/6), its new radius after splitting by TF will be the CELL radius we want (the old cell radius over
    # sqrt(2)), so we capture that and apply it to the CELL radii, then return the PARTICLE radii to what the
    # original parent was:
    #
    # let r = current CELL radius
    # so pi r**2 is the current cell apical surface area; we want to end up with half that, or pi r**2 / 2
    # set particle radius = r/(2**(1/6))
    # thus particle volume has been set to (4/3)pi ( r/(2**(1/6)) )**3 = (4/3)pi r**3 / sqrt(2)
    # after splitting by TF, particle volume is halved, thus is now: (2/3)pi r**3 / sqrt(2)
    # thus particle radius is now ( r**3 / 2sqrt(2) )**(1/3) = r / ( 2sqrt(2) )**(1/3) = r / sqrt(2)
    # thus particle apical surface area is now pi ( r / sqrt(2) )**2 = pi r**2 / 2
    # which is what we wanted. Apical surface area has been halved.
    #
    # We could also just calculate that new radius ourselves, but we do it this way so that
    # TF will place the two descendant particles an appropriate distance apart. (It places them next to each other,
    # and just touching.) Since TF will be calculating the new radius for us, we can just copy that result.
    parent.radius = gc.get_cell_radius(parent) / (2 ** (1 / 6))
    
    # Treat "mass" (because in Overdamped dynamics it actually represents drag) the same as a linear dimension
    # like radius (CELL radius, not PARTICLE radius). TF wants to treat it like volume, i.e. halving it when the
    # volume of the particle is halved. Thus after splitting, it should go down by a factor of sqrt(2), but we
    # bump it UP by a factor of sqrt(2) before splitting, then TF will halve that, producing the net change we want.
    parent.mass *= np.sqrt(2)
    
    daughter: tf.ParticleHandle
    if cfg.use_alt_cell_splitting_method:
        # Note that split() takes a "direction" argument, but it's not recognized as a keyword argument
        if parent.type() == g.LeadingEdge:
            daughter = parent.split(epu.horizontal_tangent(parent))
        else:
            # ToDo: split(direction) requires a unit vector in TF v. 0.2.0, but that requirement will be
            #  removed in the next release. Can then ditch .normalized().
            #  And of course once that is settled, I can remove support for the original non-alt method,
            #  and remove the debug code in cell_division(), below.
            daughter = parent.split(epu.random_tangent(parent).normalized())
    else:
        daughter = parent.split()
        
    # Because the PARTICLE radius that TF calculated, is what we want the new CELL radius to be, the descendant
    # particles have been placed one CELL diameter apart; and we can copy that resulting PARTICLE radius as the
    # new CELL radius:
    new_cell_radius: float = parent.radius
    
    # Then the particles themselves can be set back to the original:
    parent.radius = g.Little.radius
    daughter.radius = g.Little.radius
    gc.set_cell_radius(parent, radius=new_cell_radius)
    gc.add_particle(daughter, radius=new_cell_radius)
    
    if not cfg.use_alt_cell_splitting_method:
        # The descendants will now be oriented at an arbitrary angle (usually not in the sheet, but in/out from the
        # yolk surface). We have to adjust them, bringing them back to the yolk surface. This will usually make them
        # overlap (in terms of CELL radii), but that will resolve as they push each other apart.
        _adjust_positions(parent, daughter)
    
    return daughter

def _divide(parent: tf.ParticleHandle) -> tf.ParticleHandle:
    daughter: tf.ParticleHandle = _split(parent)
    daughter.style = tf.rendering.Style()       # Not inherited from parent, so create it
    epu.update_color(daughter)
    epu.update_color(parent)

    # Need to change r0 on all the bonds on the parent to reflect its new cell radius
    bhandle: tf.BondHandle
    for bhandle in tfu.bonds(parent):
        bonds.update_bond(bhandle)

    bond_count: int = len(daughter.bonded_neighbors)
    assert bond_count == 0, f"New particle has {bond_count} bonds before any have been made!"
    
    if daughter.type() == g.LeadingEdge:
        # Find the parent's 2 edge neighbors, and select the one closer to daughter.
        neighbor: tf.ParticleHandle = min(nbrs.bonded_neighbors_of_types(parent, [g.LeadingEdge]),
                                          key=daughter.distance)

        # Break the bond between that neighbor and parent, and bond them each to daughter,
        # before finally bonding daughter to internal particles
        edge_bond: tf.BondHandle = tfu.bond_between(neighbor, parent)
        assert edge_bond, "parent and its bonded neighbor share no bond? Of course they do, this should never happen!"
        gc.destroy_bond(edge_bond)
        bonds.make_bond(daughter, neighbor)
        bonds.make_bond(daughter, parent)
        bonds.remodel_angles(neighbor, parent, p_becoming=daughter, add=True)

    bonds.make_all_bonds(daughter)  # ToDo: This needs a bug fix, never took into account cfg.max_edge_neighbor_count
    return daughter

_one_time_cell_depletion_msg_sent: bool = False
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
    global _cumulative_cell_divisions, _evl_previous_z, _one_time_cell_depletion_msg_sent
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
    particles.extend(g.LeadingEdge.items())
    selected_particles: np.ndarray
    if cfg.cell_division_largest_first:
        undivided_particles: list[tf.ParticleHandle] = [phandle for phandle in particles
                                                        if epu.is_undivided(phandle)]
        
        # If running out and there aren't enough, just divide the ones that remain
        num_divisions = min(num_divisions, len(undivided_particles))
        
        if undivided_particles:
            # As long as there are any cells left that haven't divided even once, strictly use those up
            # before any other particle divides a second time. That should ALWAYS be true, assuming
            # cfg.total_epiboly_divisions is less than the total original cell population. The pool of undivided
            # cells should never get used up.
            #
            # In the event more divisions than that are requested, then once those run out, the correct solution
            # is to then deplete the next lower cell size, and then the next, and so on. But I haven't decided
            # the best way to write that code, and I don't expect it to ever happen, since zebrafish EVL cell
            # population does not divide enough to double during the course of epiboly. Hence, just stop
            # dividing and print a warning message in case too many cell divisions ever requested. If that ever
            # needs to become a thing, I'll have to implement code to handle that.
            
            selected_particles = _generator.choice(undivided_particles, size=num_divisions, replace=False)
        else:
            selected_particles = np.ndarray([])
            if not _one_time_cell_depletion_msg_sent:
                msg: str = "Ran out of undivided particles (only 1 division per particle currently supported)"
                tf.Logger.log(level=tf.Logger.WARNING, msg=msg)
                print(tfu.bluecolor, msg, tfu.endcolor)
                _one_time_cell_depletion_msg_sent = True
    elif cfg.cell_division_biased_by_tension:
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
    if num_divisions > 0:
        # (silliness to screen out empty list because it's an np.ndarray, and empty np.ndarrays are not iterable!)
        for phandle in selected_particles:
            # ### DEBUG:
            # ### For cells with small x, i.e. cells near the yz coordinate plane (relative to yolk center as origin)
            # ### either screen them out (don't split), or paint them to identify them. Either way, log it.
            # ### ToDo: We now have workaround for the bug; but changes coming in next release. Once that's
            # ###  stabilized, can ditch all this debug code (including those Logger statements bracketing this).
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
            if phandle.type() == g.LeadingEdge:
                epu.cumulative_edge_divisions += 1
            daughter = _divide(phandle)
            assert daughter.type() == phandle.type(), f"Daughter is type {daughter.type()}, parent is {phandle.type()}!"
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
