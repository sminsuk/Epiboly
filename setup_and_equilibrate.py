"""Create all the particles, and let them equilibrate. Generates the T = 0 configuration"""

import math
import time

import tissue_forge as tf
from epiboly_init import Little, Big, LeadingEdge   # Tissue Forge initialization and global ParticleType creation
import config as cfg

from biology import bond_maintenance as bonds
from control_flow import dynamics as dyn, \
    exec_tests as xt
from utils import tf_utils as tfu,\
    epiboly_utils as epu,\
    global_catalogs as gc,\
    plotting as plot
from utils import video_export as vx

def initialize_interior(leading_edge_phi):
    """Setup cap of interior particles (formerly known as setup_random_points()).

    Output of benchmarking code (when requesting 6000 particles by either method):
    random_points() takes ~17 sec, vs. random_nd_spherical() which takes ~0.016 sec! (1000x faster!)
        (The latter is not quite as uniform, though, and has more holes. Fixed by requesting
        more particles, and honestly even the random_points() method probably needed more as well,
        because I did notice a less-than-uniform placement even with that method.)
    filtering and scaling takes ~0.06 sec
    instantiating 1-at-a-time takes only ~0.009 sec, vs. with factory() which takes ~0.007 sec.

    So it's not the instantiating that takes time, it's the generation of all the coordinates that
    takes the time. So, implemented a function using numpy, which is pre-compiled C/C++, instead of
    built-in random_points().

    (Although come to think of it, isn't every TissueForge function by definition a precompiled C++
    function? Why would random_points() be slow?)
    """
    print("Calculating particle positions.")
    start = time.perf_counter()
    
    # Generate position vectors.
    # number of points requested are for the full sphere. Final total will be less after filtering.
    
    # With the improvement of clipping off the "escaper" interior cells below the leading edge,
    # it's no longer necessary to leave a gap between the two sets of cells. So the old "edge_margin"
    # variable is eliminated, and the process is now much simpler and easier to work with. (Maybe so
    # much so, that I'll stick with this instead of the newer algorithm? We'll see.)
    # Tweak: density (number of points requested) for these two approaches (tf.random_points()
    # and su.random_nd_spherical()). Goal is to get an initial layout (after equilibration) that touches
    # the bonded ring without deforming it. random_nd_spherical() is a bit less uniform so has more holes to
    # start with, hence more bunching elsewhere, hence tends to expand more unevenly during equilibration.
    # noinspection PyUnreachableCode
    if True:
        # new method:
        # (gets list of plain python list[3])
        vectors = tfu.random_nd_spherical(npoints=cfg.num_spherical_positions, dim=3)
    else:
        # or alternatively, old method using tf built-in (and transform to match the output type of
        # the new method, so I can test either way):
        # noinspection PyTypeChecker
        vectors = tf.random_points(tf.PointsType.Sphere.value, cfg.num_spherical_positions)
        vectors = [vector.as_list() for vector in vectors]
    
    random_points_time = time.perf_counter()
    
    # Filter to include only the ones inside the existing ring of LeadingEdge particles
    filtered_vectors = [vector for vector in vectors
                        if tfu.spherical_from_cartesian(vector)[2] < leading_edge_phi]
    num_particles = len(filtered_vectors)
    print(f"Creating {num_particles} particles.")
    filtered_time = time.perf_counter()
    
    # Transform unit sphere to sphere with radius = big particle radius + small particle radius
    # (i.e. particles just touching) and concentric on the big particle.
    # And even though plain python lists[3] is what we ultimately need, easiest to do the math
    # by converting them to fVector3 here.
    big_particle: tf.ParticleHandle = Big.items()[0]
    scale: float = big_particle.radius + LeadingEdge.radius

    def final_position(vector) -> tf.fVector3:
        return big_particle.position + tf.fVector3(vector) * scale
    
    # Better workaround of argument problem. Rebuilding the list of positions got past the
    # TypeError in factory(), but the error message for that TypeError showed that it wasn't
    # really the list of positions, but the positions themselves (type fVector3), that were
    # the problem: C++ (apparently on Mac only) not able to accept an fVector3 as parameter
    # to the fVector3 constructor (i.e. copy-constructor). And this is probably why, calling
    # still resulted in an error reported by tf itself rather than by python, complaining
    # "error: Number of particles to create could not be determined." Therefore, in our final
    # transformation, turn the resulting fVector3 objects back into normal python lists (hence normal
    # C++ vectors) using fVector3.as_list(), solved the problem.
    final_positions = [final_position(vector).as_list() for vector in filtered_vectors]
    scaled_time = time.perf_counter()
    
    # Original approach, slower: instantiate particles 1-at-a-time.
    # Benchmarked at around 0.009 seconds!
    #     for position in final_positions:
    #         LeadingEdge(position)
    
    # New approach, faster: instantiate particles using ParticleType.factory().
    #
    # Note that when I run it in Jupyter, the text output from this function, including the text output
    # of random_points(), all appears instantaneously *after* it finishes running, but when run from a
    # plain python script, I noticed that that output came out very slowly. So this was a red herring!
    #
    # Benchmarked at around 0.007 seconds! Barely faster than 1-at-a-time. So the problem was not
    # really the particle creation, but the calculation of random_points()!
    Little.factory(positions=final_positions)
    
    # Give these each a Style object so I can access them later
    # Also add each particle to the global catalog
    phandle: tf.ParticleHandle
    for phandle in Little.items():
        phandle.style = tf.rendering.Style()
        phandle.style.color = Little.style.color
        particle_values: gc.ParticleData = {"handle": phandle}
        gc.particles_by_id[phandle.id] = particle_values
    
    finished = time.perf_counter()
    # print("generating unit sphere coordinates takes:", random_points_time - start, "seconds")
    # print("filtering takes:", filtered_time - random_points_time, "seconds")
    # print("scaling (and converting to list) takes:", scaled_time - filtered_time, "seconds")
    # print("filtering/scaling/converting all together take:", scaled_time - random_points_time, "seconds")
    # print("instantiating takes:", finished - scaled_time, "seconds")

def initialize_full_sphere_evl_cells() -> None:
    """Setup what will be the interior particles, but covering the entire sphere.
    
    We'll get rid of the ones we don't want, later, after equilibration, to give us just a cap.
    """
    print("Calculating particle positions.")
    start = time.perf_counter()
    
    # Generate position vectors.
    # number of points requested are for the full sphere. Final total will be less after filtering (later).
    
    # noinspection PyUnreachableCode
    if True:
        # new method:
        # (gets list of plain python list[3])
        vectors = tfu.random_nd_spherical(npoints=cfg.num_spherical_positions, dim=3)
    else:
        # or alternatively, old method using tf built-in (and transform to match the output type of
        # the new method, so I can test either way):
        # noinspection PyTypeChecker
        vectors = tf.random_points(tf.PointsType.Sphere.value, cfg.num_spherical_positions)
        vectors = [vector.as_list() for vector in vectors]
    
    random_points_time = time.perf_counter()
    
    num_particles = len(vectors)
    print(f"Creating {num_particles} particles.")
    filtered_time = time.perf_counter()
    
    # Transform unit sphere to sphere with radius = big particle radius + small particle radius
    # (i.e. particles just touching) and concentric on the big particle.
    # And even though plain python lists[3] is what we ultimately need, easiest to do the math
    # by converting them to fVector3 here.
    big_particle: tf.ParticleHandle = Big.items()[0]
    scale: float = big_particle.radius + LeadingEdge.radius
    
    def final_position(vector) -> tf.fVector3:
        return big_particle.position + tf.fVector3(vector) * scale
    
    # Better workaround of argument problem. Rebuilding the list of positions got past the
    # TypeError in factory(), but the error message for that TypeError showed that it wasn't
    # really the list of positions, but the positions themselves (type fVector3), that were
    # the problem: C++ (apparently on Mac only) not able to accept an fVector3 as parameter
    # to the fVector3 constructor (i.e. copy-constructor). And this is probably why, calling
    # still resulted in an error reported by tf itself rather than by python, complaining
    # "error: Number of particles to create could not be determined." Therefore, in our final
    # transformation, turn the resulting fVector3 objects back into normal python lists (hence normal
    # C++ vectors) using fVector3.as_list(), solved the problem.
    final_positions = [final_position(vector).as_list() for vector in vectors]
    scaled_time = time.perf_counter()
    
    # Original approach, slower: instantiate particles 1-at-a-time.
    # Benchmarked at around 0.009 seconds!
    #     for position in final_positions:
    #         LeadingEdge(position)
    
    # New approach, faster: instantiate particles using ParticleType.factory().
    #
    # Note that when I run it in Jupyter, the text output from this function, including the text output
    # of random_points(), all appears instantaneously *after* it finishes running, but when run from a
    # plain python script, I noticed that that output came out very slowly. So this was a red herring!
    #
    # Benchmarked at around 0.007 seconds! Barely faster than 1-at-a-time. So the problem was not
    # really the particle creation, but the calculation of random_points()!
    Little.factory(positions=final_positions)
    
    # Give these each a Style object so I can access them later
    # Also add each particle to the global catalog
    phandle: tf.ParticleHandle
    for phandle in Little.items():
        phandle.style = tf.rendering.Style()
        phandle.style.color = Little.style.color
        particle_values: gc.ParticleData = {"handle": phandle}
        gc.particles_by_id[phandle.id] = particle_values
    
    finished = time.perf_counter()
    # print("generating unit sphere coordinates takes:", random_points_time - start, "seconds")
    # print("filtering takes:", filtered_time - random_points_time, "seconds")
    # print("scaling (and converting to list) takes:", scaled_time - filtered_time, "seconds")
    # print("filtering/scaling/converting all together take:", scaled_time - random_points_time, "seconds")
    # print("instantiating takes:", finished - scaled_time, "seconds")

def filter_evl_to_animal_cap(leading_edge_z: float) -> None:
    """Filter to include only the ones above where the ring will be."""
    phandle: tf.ParticleHandle
    vegetal_particles: list[tf.ParticleHandle] = [phandle for phandle in Little.items()
                                                  if phandle.position.z() < leading_edge_z]
    for phandle in vegetal_particles:
        del gc.particles_by_id[phandle.id]
        phandle.destroy()
        
    print(f"{len(Little.items())} particles remaining")
        
def add_interior_bonds():
    print("Bonding interior particles.")
    particle: tf.ParticleHandle
    for particle in Little.items():
        bonds.make_all_bonds(particle)
        assert len(particle.bonds) >= cfg.min_neighbor_count,\
            "Failed initialization: particles can't find enough nearby neighbors to bond to."
    
    print(f"Created {len(tf.BondHandle.items())} bonds.")

def initialize_bonded_edge():
    def phi_for_epiboly(epiboly_percentage=40):
        """Convert % epiboly into phi for spherical coordinates (in radians)

        epiboly_percentage: % of *vertical* distance from animal to vegetal pole (not % of arc).
        From staging description at zfin.org:

        'The extent to which the blastoderm has spread over across the yolk cell provides an extremely useful staging
        index from this stage until epiboly ends. We define percent-epiboly to mean the fraction of the yolk cell that
        the blastoderm covers; percent-coverage would be a more precise term for what we mean to say, but
        percent-epiboly immediately focuses on the process and is in common usage. Hence, at 30%-epiboly the blastoderm
        margin is at 30% of the entire distance between the animal and vegetal poles, as one estimates along the
        animal-vegetal axis.'
        """
        radius_percentage = 2 * epiboly_percentage
        adjacent = 100 - radius_percentage
        cosine_phi = adjacent / 100
        phi_rads = math.acos(cosine_phi)
        # print("intermediate results: radius_percentage, adjacent, cosine_phi, degrees =",
        #       radius_percentage, adjacent, cosine_phi, math.degrees(phi_rads))
        return phi_rads
    
    def create_ring():
        print("Generating leading edge particles.")
        
        # Where the edge should go
        leading_edge_phi = phi_for_epiboly(epiboly_percentage=cfg.epiboly_initial_percentage)
        #         print("leading edge: phi =", math.degrees(leading_edge_phi))
        
        # some basic needed quantities
        big_particle: tf.ParticleHandle = Big.items()[0]
        scale: float = big_particle.radius + LeadingEdge.radius
        r_latitude = math.sin(leading_edge_phi) * scale
        
        # (gets list of plain python list[2] - unit vectors)
        # (note, unit circle, i.e., size of the unit sphere *equator*, not the size of that latitude circle)
        unit_circle_vectors = tfu.random_nd_spherical(npoints=cfg.num_leading_edge_points, dim=2)
        
        # make 3D and scaled
        z_latitude = math.cos(leading_edge_phi) * scale
        latitude_center_position = big_particle.position + tf.fVector3([0, 0, z_latitude])
        
        def final_position(unit_vector):
            return (latitude_center_position
                    + tf.fVector3([*unit_vector, 0]) * r_latitude)
        
        final_positions = [final_position(vector).as_list() for vector in unit_circle_vectors]
        
        LeadingEdge.factory(positions=final_positions)
        
        # Give these each a Style object so I can access them later
        # Also add each particle to the global catalog
        phandle: tf.ParticleHandle
        for phandle in LeadingEdge.items():
            phandle.style = tf.rendering.Style()
            phandle.style.color = LeadingEdge.style.color
            particle_values: gc.ParticleData = {"handle": phandle}
            gc.particles_by_id[phandle.id] = particle_values
        
        return leading_edge_phi
    
    def create_bonds():
        print("Bonding ring particles.")
        
        # Use for each of the bonds we'll create here
        r0 = LeadingEdge.radius * 2
        small_small_attraction_bonded = tf.Potential.harmonic(r0=r0,
                                                              k=cfg.harmonic_edge_spring_constant,
                                                              min=r0,
                                                              max=cfg.max_potential_cutoff
                                                              )
        # plot:
        # small_small_attraction_bonded.plot(potential=True, force=True, ymin=-0.001, ymax=0.01, min=0.28, max=0.34)

        # Sort all the new particles on theta, into a new list (copy, not live)
        sorted_particles = sorted(LeadingEdge.items(), key=epu.embryo_theta)
        
        # Now they can just be bonded in the order in which they are in the list
        previous_particle = sorted_particles[-1]  # last one
        for particle in sorted_particles:
            # print("binding particles with thetas:",
            #       math.degrees(theta(previous_particle)),
            #       math.degrees(theta(particle)))
            gc.make_bond(small_small_attraction_bonded, previous_particle, particle, r0)
            previous_particle = particle
            
    leading_edge_phi = create_ring()
    create_bonds()
    return leading_edge_phi

def initialize_leading_edge_bending_resistance() -> None:
    """Add Angles to the leading edge, to keep it straight

    The normal Bonds were added to the leading edge before allowing all the particles to equilibrate.
    Adding the Angles would be convenient to do in the same loop, but can't do that, because they need to be
    done only AFTER the interior particles have equilibrated and been given their own bonds. So do a similar
    loop here and only call it after equilibration and all bond initialization is finished.
    """
    print("Adding Angles to ring particles.")
    
    if not cfg.angle_bonds_enabled:
        return
    
    edge_angle_potential: tf.Potential = tf.Potential.harmonic_angle(k=cfg.harmonic_angle_spring_constant,
                                                                     theta0=bonds.harmonic_angle_equilibrium_value(),
                                                                     tol=cfg.harmonic_angle_tolerance)
    
    # Sort all the leading edge particles on spherical coordinate theta, into a new list (copy, not live).
    # This is just like when we made the bonds. Now that we have the bonds, we COULD follow the links from
    # particle to particle, but it's easier to just sort the list of particles by theta again.
    sorted_particles = sorted(LeadingEdge.items(), key=epu.embryo_theta)
    
    # Now they can just be processed in the order in which they are in the list
    previous_particle = sorted_particles[-1]  # last one
    before_previous_particle = sorted_particles[-2]  # 2nd-to-last
    for particle in sorted_particles:
        tf.Angle.create(edge_angle_potential, before_previous_particle, previous_particle, particle)
        before_previous_particle = previous_particle
        previous_particle = particle

def initialize_particles() -> None:
    big_particle: tf.ParticleHandle = Big([5, 5, 5])
    big_particle.frozen = True
    
    # Small particles in two steps. First create simple ring of bonded LeadingEdge, then fill in the interior particles.
    leading_edge_phi = initialize_bonded_edge()
    initialize_interior(leading_edge_phi)

def replace_all_small_small_potentials(new_potential):
    """Wipes out old potential, replaces with new, for all small-small interactions"""
    tf.bind.types(new_potential, LeadingEdge, LeadingEdge)
    tf.bind.types(new_potential, LeadingEdge, Little)
    tf.bind.types(new_potential, Little, Little)

def freeze_leading_edge_z(frozen: bool = True) -> None:
    phandle: tf.ParticleHandle
    for phandle in LeadingEdge.items():
        phandle.frozen_x = False
        phandle.frozen_y = False
        phandle.frozen_z = frozen
        
def freeze_leading_edge_completely() -> None:
    phandle: tf.ParticleHandle
    for phandle in LeadingEdge.items():
        phandle.frozen = True
        
def unfreeze_leading_edge() -> None:
    phandle: tf.ParticleHandle
    for phandle in LeadingEdge.items():
        phandle.frozen = False
        
def screenshot_true_zero() -> None:
    """If exporting screenshots including equilibration, start with one screenshot before any timestepping
    
    Only works in windowless mode, because screenshots in windowed mode don't work until after the simulator opens.
    
    Single screenshot illustrates that the one labeled "Timestep 0" is really timestep 1.
    (The two images will be slightly different.)
    """
    if cfg.show_equilibration and vx.screenshot_export_enabled() and not cfg.windowed_mode:
        vx.save_screenshot("Timestep true zero")
        
def initialize_movie_export() -> None:
    """If exporting screenshots for video including equilibration, create a task list for that.
    
    This will override the task list for the running Universe.time readout at the bottom of the console, but
    the latter won't be needed because we'll be showing the image filenames, which also include Universe.time.
    """
    if cfg.show_equilibration and vx.screenshot_export_enabled():
        vx.set_screenshot_export_interval(25)
        dyn.execute_repeatedly(tasks=[{"invoke": vx.save_screenshot_repeatedly},
                                      {"invoke": plot.show_graph}
                                      ])

def show_equilibrating_message() -> None:
    if cfg.windowed_mode and not cfg.show_equilibration:
        print("Equilibrating; simulator will appear shortly...")
    else:
        print("Equilibrating...")
        
def show_is_equilibrated_message() -> None:
    print(f"Leading edge is {'' if xt.leading_edge_is_equilibrated() else 'not '}equilibrated")
        
def equilibrate(duration: float) -> None:
    if cfg.show_equilibration and cfg.windowed_mode:
        # User must quit the simulator after each equilibration step (each of the multiple launches of the
        # simulator window) in order to proceed. (It will be relaunched automatically.)
        # This is for use during development only.
        tf.show()
    else:
        # Remember that with "until" arg, this is not steps, it's units of Universe.time.
        # (Number of steps = duration / Universe.dt)
        # (And furthermore, it's a duration; it will not run "until" that time, but for that AMOUNT of time!)
        tf.step(until=duration)
    
def equilibrate_to_leading_edge() -> None:
    freeze_leading_edge_z(True)
    equilibrate(300)
    freeze_leading_edge_z(False)

def move_ring_z(destination: float) -> None:
    p: tf.ParticleHandle
    for p in LeadingEdge.items():
        x, y, z = p.position
        p.position = tf.fVector3([x, y, destination])

def setup_global_potentials() -> None:
    # Potentials, bound at the level of types:
    #
    # Large-small: LJ, originally max = equilibrium distance = sum of radii (for only repulsion), but then
    # expanded max to include attraction for the purposes of bringing particles down to the surface.
    # Should eventually switch this to Morse (or maybe harmonic) for ease of use (and consistency),
    # if I ever need to change it again.
    #
    # Small-small (both types, to themselves and to each other):
    # harmonic with repulsion only (max = equilibrium distance = sum of radii, so potential
    # applied only inside the equilibrium distance).
    
    # Big-small equilibrium distance = 3.08
    # (ToDo: Probably should switch this to harmonic for consistency.)
    # Note, with LJ, to adjust well-depth with minimal change to equilibrium distance, keep A/B constant.
    big_small_pot = tf.Potential.lennard_jones_12_6(min=0.275, max=5, A=7.3e6, B=17088)
    tf.bind.types(big_small_pot, Big, LeadingEdge)
    
    # Also bind to Little (interior) particles.
    tf.bind.types(big_small_pot, Big, Little)
    
    r0 = LeadingEdge.radius * 2
    
    # All small particles repel each other all the time, inside r0
    small_small_repulsion = tf.Potential.harmonic(r0=r0,
                                                  k=cfg.harmonic_repulsion_spring_constant,
                                                  max=r0
                                                  )
    
    replace_all_small_small_potentials(new_potential=small_small_repulsion)

    # plot:
    # big_small_pot.plot(potential=True, force=True, ymin=-1e2, ymax=1e2)
    # small_small_repulsion.plot(potential=True, force=True, ymin=-0.1, ymax=0.01)
    
def initialize_embryo() -> None:
    setup_global_potentials()
    show_equilibrating_message()
    initialize_particles()
    screenshot_true_zero()
    initialize_movie_export()
    equilibrate_to_leading_edge()
    
    # Various refactorings aside, one functional change to fix this older version of the algorithm:
    # Repeat the filter, to get escapers. (Note to self: should I now shrink/eliminate the gap?)
    leading_edge_z: float = LeadingEdge.items()[0].position.z()
    filter_evl_to_animal_cap(leading_edge_z)
    if cfg.show_equilibration:
        vx.save_screenshot("Escapers removed")

    show_is_equilibrated_message()
    add_interior_bonds()
    initialize_leading_edge_bending_resistance()

    # # ################# Test ##################
    # # Free-runnning equilibration without interior bonds.
    # # Instead of add_interior_bonds() and bending_resistance() (comment out the calls above),
    # # DESTROY the ring bonds
    # bhandle: tf.BondHandle
    # for bhandle in tf.BondHandle.items():
    #     bhandle.destroy()
    # # ############## End of test ##############

def new_initialize_embryo() -> None:
    """Based on the experiments done in alt_initialize_embryo(), but with the development scaffolding removed
    
    Note that the equilibration times at each step are highly trial-and-error, and each one sets the context for
    the subsequent steps, so these are all infinitely tweakable. These, along with number of particles (see
    num_spherical_positions) and size of particles, determine the final outcome of equilibration.
    
    Intended to run from main, but can also run from the development code down below: set cfg.show_equilibration=False
    """
    setup_global_potentials()
    show_equilibrating_message()
    
    big_particle: tf.ParticleHandle = Big([5, 5, 5])
    big_particle.frozen = True
    initialize_bonded_edge()
    freeze_leading_edge_z(True)
    
    screenshot_true_zero()
    initialize_movie_export()
    
    equilibrate(40)
    initialize_leading_edge_bending_resistance()
    freeze_leading_edge_completely()
    leading_edge_z: float = LeadingEdge.items()[0].position.z()
    move_ring_z(destination=9.8)
    initialize_full_sphere_evl_cells()
    equilibrate(100)
    filter_evl_to_animal_cap(leading_edge_z)
    move_ring_z(destination=leading_edge_z)
    freeze_leading_edge_z(True)
    
    # # This was temporary camera manipulation to get a good shot of the instability bug. Still need it?
    # vx.save_screenshot("Capture the gap before moving the camera")
    # tf.system.camera_view_top()   # one or the other, top() or reset()
    # tf.system.camera_reset()
    # tf.system.camera_zoom_to(-13)
    
    equilibrate(150)
    # Repeat the filtering, to trim "escaped" interior particles that end up below the leading edge:
    filter_evl_to_animal_cap(leading_edge_z)
    
    add_interior_bonds()
    equilibrate(10)  # Happens quickly, once bonds are added
    
    # # ################# Test ##################
    # # Free-runnning equilibration without interior bonds.
    # # Instead of add_interior_bonds() (comment out the calls above),
    # # DESTROY the ring bonds and Angles.
    # angle: tf.AngleHandle
    # bhandle: tf.BondHandle
    # for angle in tf.AngleHandle.items():
    #     angle.destroy()
    # for bhandle in tf.BondHandle.items():
    #     bhandle.destroy()
    # # ############## End of test ##############
    
    unfreeze_leading_edge()
    equilibrate(10)
    show_is_equilibrated_message()

def show() -> None:
    """Call during development and testing, immediately after calling equilibrate()
    
    If show_equilibration True, then we watched it happen;
    if False, then bring up simulator so we can examine the results.
    """
    if not cfg.show_equilibration:
        tf.show()
    
def alt_initialize_embryo() -> None:
    setup_global_potentials()
    show_equilibrating_message()
    
    big_particle: tf.ParticleHandle = Big([5, 5, 5])
    big_particle.frozen = True
    initialize_bonded_edge()
    freeze_leading_edge_z(True)
    print("Equilibrating ring (frozen in z) (40)")
    equilibrate(40)
    initialize_leading_edge_bending_resistance()
    freeze_leading_edge_completely()
    show()  # let me examine the results
    print("Equilibrated ring, now frozen completely and moving them out of the way")
    leading_edge_z: float = LeadingEdge.items()[0].position.z()
    move_ring_z(destination=9.8)
    tf.show()
    print("Moved the ring out of the way, now creating the full sphere of EVL cells")
    initialize_full_sphere_evl_cells()
    tf.show()
    print("Created full sphere of EVL cells, now letting them equilibrate (100)")
    equilibrate(100)
    show()
    print("Equilibrated full sphere of EVL cells, now filtering the excess")
    filter_evl_to_animal_cap(leading_edge_z)
    tf.show()
    print("Filtered down to animal cap, now putting the ring back")
    move_ring_z(destination=leading_edge_z)
    freeze_leading_edge_z(True)
    tf.show()
    print("Ring is back in place and unfrozen (in x and y only), now equilibrating a bit more (150)")
    equilibrate(150)
    show()
    print("Equilibrated with z frozen, now re-filtering to remove escapers")
    filter_evl_to_animal_cap(leading_edge_z)
    tf.show()
    
    print("Removed escapers, now adding interior bonds")
    add_interior_bonds()
    tf.show()
    print("Added interior bonds, now equilibrating a bit more (10)")
    equilibrate(10)  # Happens quickly, once bonds are added
    show()
    print("Equilibrated with bonds and still frozen in z, now unfreezing and letting the edge relax (10)")

    # # ################# Test ##################
    # # Free-runnning equilibration without interior bonds.
    # # Instead of add_interior_bonds() (comment out the calls above),
    # # DESTROY the ring bonds and Angles.
    # print("Removed escapers, now ELIMINATING bonds")
    # angle: tf.AngleHandle
    # bhandle: tf.BondHandle
    # for angle in tf.AngleHandle.items():
    #     angle.destroy()
    # for bhandle in tf.BondHandle.items():
    #     bhandle.destroy()
    # tf.show()
    # print("Eliminated bonds, now unfreezing and letting the edge relax (10)")
    # # ############## End of test ##############
    
    unfreeze_leading_edge()
    equilibrate(10)
    show()
    show_is_equilibrated_message()
    print("Edge relaxed, now letting 'er rip (" + tfu.bluecolor + "Goal: " + tfu.endcolor
          + "Should not expand more, if well-equilibrated and cell count and radius are correct!)")
    
    # Still ToDo: run the whole script with this version and see how it goes.
    # Then, work on getting the cell numbers and sizes correct.

if __name__ == "__main__":
    # While developing this module, just execute this in isolation.
    # Designed to run in windowed mode, and flipping between show_equilibration True/False for testing.
    def show_utime() -> None:
        print(f"\rUniverse.time = {round(tf.Universe.time, 2)}", end="")
    
    epu.reset_camera()
    dyn.initialize_master_event()
    dyn.execute_repeatedly(tasks=[{"invoke": show_utime}])
    
    # ***** Choose one: *****
    # initialize_embryo()         # to run the old one and make sure I haven't broken the sim during W.I.P.
    # alt_initialize_embryo()     # to run the new one with the ability to pause and examine each step
    new_initialize_embryo()     # to run the new one without pauses, as it will play when run in the sim from main
    
    # And then just let it run and see how much further equilibration happens
    # (Small duration, equal to the amount of time for equilibration-proper, so that it
    # will have a stopping point when free-running in windowless mode; in windowed mode,
    # need to quit manually, by quitting simulator.)
    equilibrate(300)
    
    plot.save_graph()
    vx.make_movie()
    
    # Only after making the movie, so that these stills won't be included
    vx.final_result_screenshots()
    
    