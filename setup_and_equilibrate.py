"""setup_and_equilibrate.py

Create all the particles, and let them equilibrate. Generates the T = 0 configuration
"""

import math
import time

import tissue_forge as tf
import epiboly_globals as g
import epiboly_init
import config as cfg

from biology import bond_maintenance as bonds
from control_flow import events
from utils import tf_utils as tfu,\
    epiboly_utils as epu,\
    global_catalogs as gc,\
    plotting as plot
from utils import video_export as vx

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
    big_particle: tf.ParticleHandle = g.Big.items()[0]
    scale: float = big_particle.radius + g.LeadingEdge.radius
    
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
    #         g.LeadingEdge(position)
    
    # New approach, faster: instantiate particles using ParticleType.factory().
    #
    # Note that when I run it in Jupyter, the text output from this function, including the text output
    # of random_points(), all appears instantaneously *after* it finishes running, but when run from a
    # plain python script, I noticed that that output came out very slowly. So this was a red herring!
    #
    # Benchmarked at around 0.007 seconds! Barely faster than 1-at-a-time. So the problem was not
    # really the particle creation, but the calculation of random_points()!
    g.Little.factory(positions=final_positions)
    
    # Give these each a Style object so I can access them later
    # Also add each particle to the global catalog
    phandle: tf.ParticleHandle
    for phandle in g.Little.items():
        phandle.style = tf.rendering.Style()
        phandle.style.color = g.Little.style.color
        gc.add_particle(phandle)
    
    finished = time.perf_counter()
    # print("generating unit sphere coordinates takes:", random_points_time - start, "seconds")
    # print("filtering takes:", filtered_time - random_points_time, "seconds")
    # print("scaling (and converting to list) takes:", scaled_time - filtered_time, "seconds")
    # print("filtering/scaling/converting all together take:", scaled_time - random_points_time, "seconds")
    # print("instantiating takes:", finished - scaled_time, "seconds")

def filter_evl_to_animal_cap(leading_edge_z: float) -> None:
    """Filter to include only the ones above where the ring will be."""
    phandle: tf.ParticleHandle
    vegetal_particles: list[tf.ParticleHandle] = [phandle for phandle in g.Little.items()
                                                  if phandle.position.z() < leading_edge_z]
    for phandle in vegetal_particles:
        gc.destroy_particle(phandle)
        
    print(f"{len(g.Little.items())} particles remaining")
        
def add_interior_bonds():
    print("Bonding interior particles.")
    particle: tf.ParticleHandle
    for particle in g.Little.items():
        bonds.make_all_bonds(particle)
    
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
        big_particle: tf.ParticleHandle = g.Big.items()[0]
        scale: float = big_particle.radius + g.LeadingEdge.radius
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
        
        g.LeadingEdge.factory(positions=final_positions)
        
        # Give these each a Style object so I can access them later
        # Also add each particle to the global catalog
        phandle: tf.ParticleHandle
        for phandle in g.LeadingEdge.items():
            phandle.style = tf.rendering.Style()
            phandle.style.color = g.LeadingEdge.style.color
            gc.add_particle(phandle)
    
    def create_bonds():
        print("Bonding ring particles.")
        
        # Use for each of the bonds we'll create here
        r0 = g.LeadingEdge.radius * 2
        small_small_attraction_bonded = tf.Potential.harmonic(r0=r0,
                                                              k=cfg.harmonic_edge_spring_constant,
                                                              min=r0,
                                                              max=cfg.max_potential_cutoff
                                                              )
        # plot:
        # small_small_attraction_bonded.plot(potential=True, force=True, ymin=-0.001, ymax=0.01, min=0.28, max=0.34)

        # Sort all the new particles on theta, into a new list (copy, not live)
        sorted_particles = sorted(g.LeadingEdge.items(), key=epu.embryo_theta)
        
        # Now they can just be bonded in the order in which they are in the list
        previous_particle = sorted_particles[-1]  # last one
        for particle in sorted_particles:
            # print("binding particles with thetas:",
            #       math.degrees(theta(previous_particle)),
            #       math.degrees(theta(particle)))
            gc.create_bond(small_small_attraction_bonded, previous_particle, particle)
            previous_particle = particle
            
    create_ring()
    create_bonds()

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
    sorted_particles = sorted(g.LeadingEdge.items(), key=epu.embryo_theta)
    
    # Now they can just be processed in the order in which they are in the list
    previous_particle = sorted_particles[-1]  # last one
    before_previous_particle = sorted_particles[-2]  # 2nd-to-last
    for particle in sorted_particles:
        gc.create_angle(edge_angle_potential, before_previous_particle, previous_particle, particle)
        before_previous_particle = previous_particle
        previous_particle = particle
        
    # bonds.test_ring_is_fucked_up()

def replace_all_small_small_potentials(new_potential):
    """Wipes out old potential, replaces with new, for all small-small interactions"""
    tf.bind.types(new_potential, g.LeadingEdge, g.LeadingEdge)
    tf.bind.types(new_potential, g.LeadingEdge, g.Little)
    tf.bind.types(new_potential, g.Little, g.Little)

def freeze_leading_edge_z() -> None:
    phandle: tf.ParticleHandle
    for phandle in g.LeadingEdge.items():
        phandle.frozen_x = False
        phandle.frozen_y = False
        phandle.frozen_z = True
        
def freeze_leading_edge_completely() -> None:
    phandle: tf.ParticleHandle
    for phandle in g.LeadingEdge.items():
        phandle.frozen = True
        
def unfreeze_leading_edge() -> None:
    phandle: tf.ParticleHandle
    for phandle in g.LeadingEdge.items():
        phandle.frozen = False
        
def screenshot_true_zero() -> None:
    """If exporting screenshots including equilibration, start with one screenshot before any timestepping
    
    Only works in windowless mode, because screenshots in windowed mode don't work until after the simulator opens.
    
    Single screenshot illustrates that the one labeled "Timestep 0" is really timestep 1.
    (The two images will be slightly different.)
    """
    if cfg.show_equilibration and vx.screenshot_export_enabled() and not cfg.windowed_mode:
        vx.save_screenshots("Timestep true zero")
        
def initialize_export_tasks() -> None:
    """If exporting plots and or screenshots for video, including equilibration, create a task list for that.
    
    This will override the task list for the running Universe.time readout at the bottom of the console, but
    the latter won't be needed because we'll be showing the image filenames, which also include Universe.time.
    """
    if cfg.show_equilibration:
        task_list: list[events.Task] = [{"invoke": plot.show_graphs}]
        if vx.screenshot_export_enabled():
            vx.set_screenshot_export_interval(25)
            task_list.append({"invoke": vx.save_screenshot_repeatedly})
            
        events.execute_repeatedly(tasks=task_list)

def show_equilibrating_message() -> None:
    if cfg.windowed_mode and not cfg.show_equilibration:
        print("Equilibrating; simulator will appear shortly...")
    else:
        print("Equilibrating...")
        
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
    
def move_ring_z(destination: float) -> None:
    p: tf.ParticleHandle
    for p in g.LeadingEdge.items():
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
    tf.bind.types(big_small_pot, g.Big, g.LeadingEdge)
    
    # Also bind to Little (interior) particles.
    tf.bind.types(big_small_pot, g.Big, g.Little)
    
    r0 = g.LeadingEdge.radius * 2
    
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
    """Based on the experiments done in alt_initialize_embryo(), but with the development scaffolding removed
    
    Compared to my previous method, a new approach, start all tension-generating forces at once:
    
    Unfreeze the ring before adding any interior bonds, and let it equilibrate for 10; also, don't add the Angle bonds
    until after this equilibration is over.
    
    Then add the bonds and angles, and start the bond update rules, as well as the external force, all at the same time,
    with no intervening equilibration. (I.e., add the bonds, then segue directly into the sim proper.)
    
    Note that the equilibration times at each step are highly trial-and-error, and each one sets the context for
    the subsequent steps, so these are all infinitely tweakable. These, along with number of particles (see
    num_spherical_positions) and size of particles, determine the final outcome of equilibration.
    
    Intended to run from main, but can also run from the development code down below.
    To get a movie of just this alone, set cfg.show_equilibration=True and cfg.windowed_mode=False.
    
    Note that at the end of this process, the sheet is unstable, because once bonds are added, they need
    to be balanced by yolk cortical tension, which is not applied until the sim proper begins. To see the
    stable state where yolk cortical tension has been added but no additional force to drive epiboly,
    set cfg.external_force to 0, and run the full sim from main.
    """
    setup_global_potentials()
    show_equilibrating_message()
    
    big_particle: tf.ParticleHandle = g.Big([5, 5, 5])
    big_particle.frozen = True
    initialize_bonded_edge()
    freeze_leading_edge_z()
    
    screenshot_true_zero()
    initialize_export_tasks()
    
    equilibrate(100)
    freeze_leading_edge_completely()
    leading_edge_z: float = g.LeadingEdge.items()[0].position.z()
    move_ring_z(destination=9.8)
    initialize_full_sphere_evl_cells()
    equilibrate(100)
    filter_evl_to_animal_cap(leading_edge_z)
    move_ring_z(destination=leading_edge_z)
    freeze_leading_edge_z()
    
    # # This was temporary camera manipulation to get a good shot of the instability bug. Still need it?
    # vx.save_screenshots("Capture the gap before moving the camera")
    # tf.system.camera_view_top()   # one or the other, top() or reset()
    # tf.system.camera_reset()
    # tf.system.camera_zoom_to(-13)
    
    equilibrate(100)
    # Repeat the filtering, to trim "escaped" interior particles that end up below the leading edge:
    filter_evl_to_animal_cap(leading_edge_z)
    
    unfreeze_leading_edge()
    equilibrate(10)

    add_interior_bonds()
    initialize_leading_edge_bending_resistance()
    # equilibrate(10)  # Now none at all for this
    
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
    
    big_particle: tf.ParticleHandle = g.Big([5, 5, 5])
    big_particle.frozen = True
    initialize_bonded_edge()
    freeze_leading_edge_z()
    screenshot_true_zero()          # Need these? I wasn't sure when I did the big clean-up
    initialize_export_tasks()       # of this module, after not having used it in awhile.
    print("Equilibrating ring (frozen in z) (100)")
    equilibrate(100)
    freeze_leading_edge_completely()
    show()  # let me examine the results
    print("Equilibrated ring, now frozen completely and moving them out of the way")
    leading_edge_z: float = g.LeadingEdge.items()[0].position.z()
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
    freeze_leading_edge_z()
    tf.show()
    print("Ring is back in place and unfrozen (in x and y only), now equilibrating a bit more (100)")
    equilibrate(100)
    show()
    print("Equilibrated with z frozen, now re-filtering to remove escapers")
    filter_evl_to_animal_cap(leading_edge_z)
    tf.show()
    
    print("Removed escapers, now unfreezing and letting the edge relax (10)")
    unfreeze_leading_edge()
    equilibrate(10)
    show()
    
    print("Now adding interior bonds and edge angles")
    add_interior_bonds()
    initialize_leading_edge_bending_resistance()
    tf.show()   # (If you run this simulator, it will start to shrink, because balancing yolk tension not added yet.)

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
    # tf.show()
    # # ############## End of test ##############

if __name__ == "__main__":
    # While developing this module, just execute this in isolation.
    # Designed to run in windowed mode, and flipping between show_equilibration True/False for testing.
    def show_utime() -> None:
        print(f"\rUniverse.time = {round(tf.Universe.time, 2)}", end="")
    
    epiboly_init.init()
    epu.reset_camera()  # Maybe should be vx.init_camera_data() now? Deal with, if I ever need to do this again.
    events.initialize_master_event()
    events.execute_repeatedly(tasks=[{"invoke": show_utime}])
    
    # ***** Choose one: *****
    alt_initialize_embryo()     # to run initialization with the ability to pause and examine each step
    # initialize_embryo()         # run initialization without pauses, as it will play when run in the sim from main
    
    if cfg.show_equilibration:
        vx.make_movie()
    
    # Only after making the movie, so that these stills won't be included
    vx.final_result_screenshots()
    
    