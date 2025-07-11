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

# We'll calculate these from the desired number of cells in the EVL.
_initial_cell_radius: float = 0
_num_spherical_positions: int = 0
_num_leading_edge_points: int = 0

# Value depends on setup method being used. Module main needs this for certain timing
equilibration_time: int = 0

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
        vectors = tfu.random_nd_spherical(npoints=_num_spherical_positions, dim=3)
    else:
        # or alternatively, old method using tf built-in (and transform to match the output type of
        # the new method, so I can test either way):
        # noinspection PyTypeChecker
        vectors = tf.random_points(tf.PointsType.Sphere.value, _num_spherical_positions)
        vectors = [vector.as_list() for vector in vectors]
    
    random_points_time = time.perf_counter()
    
    num_particles = len(vectors)
    print(f"Creating {num_particles} particles.")
    filtered_time = time.perf_counter()
    
    # Transform unit sphere to sphere with radius = yolk particle radius + EVL particle radius
    # (i.e. particles just touching) and concentric on the yolk particle.
    # And even though plain python lists[3] is what we ultimately need, easiest to do the math
    # by converting them to fVector3 here.
    yolk_particle: tf.ParticleHandle = g.Yolk.items()[0]
    scale: float = yolk_particle.radius + g.LeadingEdge.radius
    
    def final_position(vector) -> tf.fVector3:
        return yolk_particle.position + tf.fVector3(vector) * scale
    
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
    g.Evl.factory(positions=final_positions)
    
    # Give these each a Style object so I can access them later
    # Also assign each particle its own cell radius, and add each one to the global catalog
    phandle: tf.ParticleHandle
    for phandle in g.Evl.items():
        phandle.style = tf.rendering.Style()
        gc.add_particle(phandle, radius=_initial_cell_radius)
    
    finished = time.perf_counter()
    # print("generating unit sphere coordinates takes:", random_points_time - start, "seconds")
    # print("filtering takes:", filtered_time - random_points_time, "seconds")
    # print("scaling (and converting to list) takes:", scaled_time - filtered_time, "seconds")
    # print("filtering/scaling/converting all together take:", scaled_time - random_points_time, "seconds")
    # print("instantiating takes:", finished - scaled_time, "seconds")

def filter_evl_to_animal_cap_phi(leading_edge_phi: float) -> None:
    """Filter to include only the ones above where the leading edge will be."""
    phandle: tf.ParticleHandle
    vegetal_particles: list[tf.ParticleHandle] = [phandle for phandle in g.Evl.items()
                                                  if epu.embryo_phi(phandle) > leading_edge_phi]
    for phandle in vegetal_particles:
        gc.destroy_particle(phandle)
    
    print(f"{len(g.Evl.items())} particles remaining")

def filter_evl_to_animal_cap(leading_edge_z: float) -> None:
    """Filter to include only the ones above where the ring will be.
    
    Now with the move to cell radius >> particle radius, need to account for that, and
    cut off the cells that are too close to the edge, which would otherwise push the edge
    down before the simulation even starts.
    """
    phandle: tf.ParticleHandle
    vegetal_particles: list[tf.ParticleHandle] = [phandle for phandle in g.Evl.items()
                                                  if phandle.position.z() < leading_edge_z + 2.5 * _initial_cell_radius]
    for phandle in vegetal_particles:
        gc.destroy_particle(phandle)
        
    print(f"{len(g.Evl.items())} particles remaining")
        
def add_interior_bonds():
    print("Bonding interior particles.")
    particle: tf.ParticleHandle
    for particle in g.Evl.items():
        bonds.make_all_bonds(particle, min_neighbor_count=5)
    
    print(f"Created {len(tf.BondHandle.items())} bonds.")

def remove_edge_bonds() -> None:
    # Break any bonds between two leading edge particles. (They have non-edge spring constants, so
    # they need to be re-made. Plus the connection topology is arbitrary and we need to control that.)
    bhandle: tf.BondHandle
    for bhandle in tf.BondHandle.items():
        if bonds.is_edge_bond(bhandle):
            gc.destroy_bond(bhandle)

def create_edge_bonds() -> None:
    """Make bonds between adjacent leading edge particles.
    
    Note: any existing bonds between any pair of edge particles should be removed before this is called.
    """
    print("Bonding leading edge particles.")
    
    # Sort all the leading edge particles on theta, into a new list (copy, not live)
    sorted_particles: list[tf.ParticleHandle] = sorted(g.LeadingEdge.items(), key=epu.embryo_theta)
    
    # Now they can just be bonded in the order in which they are in the list
    particle: tf.ParticleHandle
    previous_particle: tf.ParticleHandle = sorted_particles[-1]  # last one
    for particle in sorted_particles:
        # Make bonds with appropriate spring constant
        bonds.make_bond(particle, previous_particle)
        previous_particle = particle

def initialize_bonded_edge():
    def create_ring():
        print("Generating leading edge particles.")
        
        # Where the edge should go
        leading_edge_phi = epu.phi_for_epiboly(epiboly_percentage=cfg.epiboly_initial_percentage)
        
        # some basic needed quantities
        yolk_particle: tf.ParticleHandle = g.Yolk.items()[0]
        scale: float = yolk_particle.radius + g.LeadingEdge.radius
        r_latitude = math.sin(leading_edge_phi) * scale
        
        # (gets list of plain python list[2] - unit vectors)
        # (note, unit circle, i.e., size of the unit sphere *equator*, not the size of that latitude circle)
        unit_circle_vectors = tfu.random_nd_spherical(npoints=_num_leading_edge_points, dim=2)
        
        # make 3D and scaled
        z_latitude = math.cos(leading_edge_phi) * scale
        latitude_center_position = yolk_particle.position + tf.fVector3([0, 0, z_latitude])
        
        def final_position(unit_vector):
            return (latitude_center_position
                    + tf.fVector3([*unit_vector, 0]) * r_latitude)
        
        final_positions = [final_position(vector).as_list() for vector in unit_circle_vectors]
        
        g.LeadingEdge.factory(positions=final_positions)
        
        # Give these each a Style object so I can access them later
        # Also assign each particle its own cell radius, and add each one to the global catalog
        phandle: tf.ParticleHandle
        for phandle in g.LeadingEdge.items():
            phandle.style = tf.rendering.Style()
            gc.add_particle(phandle, radius=_initial_cell_radius)
            gc.set_cell_original_tier(phandle, tier=0)
    
    create_ring()
    create_edge_bonds()

def initialize_leading_edge_bending_resistance() -> None:
    """Add Angles to the leading edge, to keep it straight

    The normal Bonds were added to the leading edge before allowing all the particles to equilibrate.
    Adding the Angles would be convenient to do in the same loop, but can't do that, because they need to be
    done only AFTER the interior particles have equilibrated and been given their own bonds. So do a similar
    loop here and only call it after equilibration and all bond initialization is finished.
    """
    if not cfg.angle_bonds_enabled:
        print("Running without Angle bonds")
        return
    
    print("Adding Angle bonds to ring particles.")

    edge_angle_potential: tf.Potential = tf.Potential.harmonic_angle(k=cfg.harmonic_angle_spring_constant,
                                                                     theta0=bonds.evl_edge_vertex_equilibrium_angle(),
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
        
    # bonds.test_ring_is_fubar()

def replace_all_evl_evl_potentials(new_potential):
    """Wipes out old potential, replaces with new, for all EVL-EVL interactions"""
    tf.bind.types(new_potential, g.LeadingEdge, g.LeadingEdge)
    tf.bind.types(new_potential, g.LeadingEdge, g.Evl)
    tf.bind.types(new_potential, g.Evl, g.Evl)

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
        # ToDo: Bug here. See commit message for this comment (2023-06-06) for details and how to fix.
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
    # Yolk-EVL: originally max = equilibrium distance = sum of radii (for only repulsion), but then
    # expanded max to include attraction for the purposes of bringing particles down to the surface.
    yolk_evl_pot = tf.Potential.harmonic(k=cfg.harmonic_yolk_evl_spring_constant,
                                         r0=g.Yolk.radius + g.Evl.radius,
                                         min=0.275,
                                         max=5)
    tf.bind.types(yolk_evl_pot, g.Yolk, g.LeadingEdge)
    
    # Also bind to Evl (interior) particles.
    tf.bind.types(yolk_evl_pot, g.Yolk, g.Evl)
    
    # EVL-EVL (both types, to themselves and to each other):
    # harmonic with repulsion only (max = equilibrium distance = sum of radii, so potential
    # applied only inside the equilibrium distance). (Attraction will be applied on a cell-by-cell
    # basis, i.e. via explicit bonds.)
    # In contrast to yolk-EVL interactions, base this on cell radius, not particle radius
    r0 = _initial_cell_radius * 2
    
    # All EVL particles repel each other all the time, inside r0
    evl_evl_repulsion = tf.Potential.harmonic(r0=r0,
                                              k=cfg.harmonic_repulsion_spring_constant,
                                              max=r0)
    
    replace_all_evl_evl_potentials(new_potential=evl_evl_repulsion)

    # plot:
    # yolk_evl_pot.plot(potential=True, force=True, ymin=-1e2, ymax=1e2)
    # evl_evl_repulsion.plot(potential=True, force=True, ymin=-0.1, ymax=0.01)
    
def adjust_global_evl_potentials() -> None:
    """Adjust r0 for type-based repulsion between EVL particles. (Leaves yolk-EVL potentials alone.)
    
    The global EVL-EVL potential is repulsion only and is used to provide volume exclusion. This is important
    for equilibration during setup, and also throughout the simulation. During setup, all cells are the
    same size, so it was clean to have r0 be twice that radius. Once the main sim begins, if cell division
    enabled, it's a bit more awkward because of different sized cells. More elaborate solution would be to
    have the different cell sizes use different ParticleTypes so that global type-based potentials could
    differ depending on cell size. But for now, keep things simple. Divided and undivided cells use the same
    ParticleType, hence must all have the same potential with the same r0. So base r0 on the smaller cell size.
    """
    if cfg.cell_division_enabled:
        # Reduce repulsion to amount appropriate for divided cells.
        r0: float = _initial_cell_radius * math.sqrt(2)
        evl_evl_repulsion: tf.Potential = tf.Potential.harmonic(r0=r0,
                                                                k=cfg.harmonic_repulsion_spring_constant,
                                                                max=r0)

        replace_all_evl_evl_potentials(new_potential=evl_evl_repulsion)

def find_boundary() -> None:
    """Boundary cells are those above the line that are bonded to any below the line"""
    # Call this once, just to display the value of leading edge phi where we want the edge cells to actually be
    epu.phi_for_epiboly(epiboly_percentage=cfg.epiboly_initial_percentage)
    
    # Call it again with a value one cell radius below that, to get the value of phi to use
    # for separating those cells (the "inner boundary" of the subgraph) from the ones bonded to
    # (the "outer boundary" of the subgraph).
    radius_as_percentage: float = 50 * _initial_cell_radius / epu.embryo_radius()
    cutoff_line: float = cfg.epiboly_initial_percentage + radius_as_percentage
    leading_edge_phi = epu.phi_for_epiboly(epiboly_percentage=cutoff_line)
    
    # Find the boundary particles and make them leading edge
    p: tf.ParticleHandle
    neighbor: tf.ParticleHandle
    for p in g.Evl.items():
        if epu.embryo_phi(p) < leading_edge_phi:
            if any([epu.embryo_phi(neighbor) >= leading_edge_phi for neighbor in p.bonded_neighbors]):
                p.become(g.LeadingEdge)
                gc.set_cell_original_tier(p, tier=0)
                
    # Delete all the particles below the leading edge
    filter_evl_to_animal_cap_phi(leading_edge_phi)
    
    # Now make bonds between adjacent leading edge particles (replacing any existing edge-edge bonds)
    remove_edge_bonds()
    create_edge_bonds()
    
def setup_initial_cell_number_and_size() -> None:
    """Based on the requested initial number of EVL cells, calculate their size

    Also store how many cells it takes to cover the whole sphere (and in the case of the older initialization
    methodology, also how many we want in the marginal ring); we'll use these intermediate results to know
    how many to instantiate.
    """
    global _initial_cell_radius, _num_spherical_positions, _num_leading_edge_points
    
    _num_spherical_positions = math.floor(cfg.epiboly_initial_num_evl_cells * 100 / cfg.epiboly_initial_percentage)
    embryo_surface_area: float = 4 * math.pi * epu.embryo_radius() ** 2
    
    # Footprint area of each cell should be treated as a hexagon, not a circle
    initial_cell_hexagonal_footprint_area: float = embryo_surface_area / _num_spherical_positions
    
    # Cell radius is therefore the radius of the inscribed circle (NOT the radius of a presumed circular footprint).
    # i.e., distance from hexagon center to center of hexagon edge. This results in a radius about 5% smaller.
    # Makes a big difference. Otherwise, cells are too large and are crowded.
    _initial_cell_radius = math.sqrt(initial_cell_hexagonal_footprint_area / (2 * math.sqrt(3)))

    # Hand this result off to the epu module, where it will live for the remainder of the simulation
    epu.initial_cell_radius = _initial_cell_radius
    print(f"Calculated initial cell radius: {_initial_cell_radius}")
    print(f"Calculated total particles on sphere before filtering: {_num_spherical_positions}")

    # For new init methodology (graph based), the above is all we need. We'll create all cells at once
    # from _num_spherical_positions, and then discover which ones are the margin. But with the
    # older init methodology ("with config" because it used to use explicit config values for both
    # size and number of cells, but now we config only total number of cells and derive cell size
    # from that), we do it differently:
    if not cfg.initialization_algo_graph_based:
        # Probably going to get rid of this in the near future anyway. ToDo
        
        # We'll set up a ring of EVL margin cells separately, so we need to know how many. It can't
        # be a config because we can only calculate it now that we know the cell radius:
        relative_z: float = epu.relative_z_from_epiboly_percentage(cfg.epiboly_initial_percentage)
        _num_leading_edge_points = math.floor(epu.circumference_of_circle_at_relative_z(relative_z) /
                                              (2 * _initial_cell_radius))
        print(f"Calculated total particles in ring: {_num_leading_edge_points}")

        # And _num_spherical_positions is also now incorrect for this method, but there isn't a suitable way to
        # correct it here. Instead, we correct for it later with very strict filtering in filter_evl_to_animal_cap().
        
def initialize_paint_pattern() -> None:
    p: tf.ParticleHandle
    all_particles: list[tf.ParticleHandle] = []
    all_particles.extend(g.Evl.items())
    all_particles.extend(g.LeadingEdge.items())

    # If we are using one of the lineage-tracing PaintPatterns, then initialize the pattern; i.e.,
    # determine which cells to add lineage tracer to
    match cfg.paint_pattern:
        case cfg.PaintPattern.ORIGINAL_TIER:
            # Tier 0 particles were labeled during LeadingEdge creation; other cells have tier = None
            # Now set tier on those other cells
            finished: bool = False
            previous_tier: int = 0
            while not finished:
                untiered_particles: list[tf.ParticleHandle] = [p for p in all_particles
                                                               if gc.get_cell_original_tier(p) is None]
                next_tiered_particles: list[tf.ParticleHandle] = \
                    [p for p in untiered_particles
                     if any([gc.get_cell_original_tier(neighbor) == previous_tier
                             for neighbor in p.bonded_neighbors])]
                finished = len(next_tiered_particles) == 0
                next_tier: int = previous_tier + 1
                for p in next_tiered_particles:
                    gc.set_cell_original_tier(p, tier=next_tier)
                previous_tier = next_tier
                
            # Add lineage tracer to all particles that are in the tier we're interested in
            for p in all_particles:
                if gc.get_cell_original_tier(p) == cfg.paint_tier:
                    gc.set_cell_lineage_tracer(p)
        
        case cfg.PaintPattern.VERTICAL_STRIPE:
            for p in all_particles:
                if 0 < epu.embryo_theta(p) < math.pi / 6:
                    gc.set_cell_lineage_tracer(p)
            
        case cfg.PaintPattern.PATCH:
            front_theta: float = -math.pi / 2
            edge_phi: float = epu.leading_edge_max_phi()
            patch_width: float = math.radians(cfg.patch_width)
            patch_height: float = math.radians(cfg.patch_height)
            min_phi_dist: float = math.radians(cfg.patch_margin_gap)
            max_phi_dist: float = min_phi_dist + patch_height
            max_theta_dist: float = patch_width / 2
            for p in all_particles:
                theta, phi = epu.embryo_coords(p)
                phi_distance: float = edge_phi - phi
                theta_distance: float = abs(front_theta - theta)
                if theta_distance < max_theta_dist and min_phi_dist <= phi_distance < max_phi_dist:
                    gc.set_cell_lineage_tracer(p)

    epu.update_all_particle_colors()

def initialize_embryo() -> None:
    """Initialize all particles and bonds
    
    Intended to run from main, but can also run the method of choice from the development code down below.
    To get a movie of just this alone, set cfg.show_equilibration=True and cfg.windowed_mode=False, then
    tweak sim_finished() (in main.py) to end the simulation after setup.equilibration_time + 1.
    (1 extra timestep needed in order to capture changes in particles and bonds after the final equilibration)
    
    Note that the equilibration times at each step are highly trial-and-error, and each one sets the context for
    the subsequent steps, so these are all infinitely tweakable. These, along with number of particles (see
    num_spherical_positions) and size of particles, determine the final outcome of equilibration.
    
    Note that at the end of this process, the sheet is unstable, because once there is a bonded network with an
    edge, it's under tension, which needs to be balanced by yolk cortical tension, which is not applied until the
    sim proper begins. To see the stable state where yolk cortical tension has been added but no additional force
    to drive epiboly, set cfg.run_balanced_force_control to True, and run the full sim from main.
    """
    setup_initial_cell_number_and_size()
    if cfg.initialization_algo_graph_based:
        initialize_embryo_with_graph_boundary()
    else:
        initialize_embryo_with_config()

def initialize_embryo_with_graph_boundary() -> None:
    """Discover leading edge of particles using graph theory definition of 'boundary'
    
    i.e.: https://en.wikipedia.org/wiki/Boundary_(graph_theory)
    
    Compared to my previous method, the number of leading edge cells is no longer arbitrary, no longer
    specified with a config variable, and no longer created separately as a ring before adding the internal
    particles above that. Instead, set up a layer of particles, and then discover which ones should
    be designated as the edge. And, instead of creating a sphere of particles and deleting the ones below the
    desired edge before adding bonds, now we create a sphere of particles and add the bonds right away, which
    creates a graph. This makes it easy to define a desired subset of that graph above a certain polar angle,
    and find the boundary of that subset, which we designate as edge particles. Only then do we delete all the
    excess particles (and bonds) below that.
    
    As a bonus, this is actually a simpler method of setting up, with fewer steps.
    """
    global equilibration_time
    durations: list[int] = [400, 100]
    equilibration_time = sum(durations)
    
    setup_global_potentials()
    show_equilibrating_message()
    
    yolk_particle: tf.ParticleHandle = g.Yolk([5, 5, 5])
    yolk_particle.frozen = True
    yolk_particle.style = tf.rendering.Style()
    yolk_particle.style.color = g.Yolk.style.color
    # Uncomment to make the yolk invisible:
    # yolk_particle.style.visible = False
    
    initialize_full_sphere_evl_cells()
    screenshot_true_zero()
    initialize_export_tasks()
    
    # This equilibration, quite long, removes most of the particle overlap as shown by the tension graphs at T=0
    equilibrate(durations[0])
    add_interior_bonds()
    
    # Safe to equilibrate even though there's bonds under tension without external balancing force, because the
    # bonded network covers the WHOLE surface. It's stable until after part of that network is removed.
    equilibrate(durations[1])
    
    find_boundary()
    adjust_global_evl_potentials()
    initialize_leading_edge_bending_resistance()
    initialize_paint_pattern()
    
def initialize_embryo_with_config() -> None:
    """Older init method of arbitrarily deciding how many edge particles to have, and creating a ring of them
    
    Based on the experiments done in alt_initialize_embryo_with_config(), but with the development scaffolding removed
    
    Compared to my even older method (deleted), a better approach, start all tension-generating forces at once:
    Unfreeze the ring before adding any interior bonds, and let it equilibrate. Then add the bonds and angles,
    and start the bond update rules, as well as the external force, all at the same time, with no intervening
    equilibration. (I.e., add the bonds, then segue directly into the sim proper.)
    """
    global equilibration_time
    durations: list[int] = [100, 100, 100, 400]
    equilibration_time = sum(durations)
    
    setup_global_potentials()
    show_equilibrating_message()
    
    yolk_particle: tf.ParticleHandle = g.Yolk([5, 5, 5])
    yolk_particle.frozen = True
    yolk_particle.style = tf.rendering.Style()
    yolk_particle.style.color = g.Yolk.style.color
    initialize_bonded_edge()
    freeze_leading_edge_z()
    
    screenshot_true_zero()
    initialize_export_tasks()
    
    equilibrate(durations[0])
    freeze_leading_edge_completely()
    leading_edge_z: float = g.LeadingEdge.items()[0].position.z()
    move_ring_z(destination=9.8)
    initialize_full_sphere_evl_cells()
    equilibrate(durations[1])
    filter_evl_to_animal_cap(leading_edge_z)
    move_ring_z(destination=leading_edge_z)
    freeze_leading_edge_z()
    
    # # This was temporary camera manipulation to get a good shot of the instability bug. Still need it?
    # vx.save_screenshots("Capture the gap before moving the camera")
    # tf.system.camera_view_top()   # one or the other, top() or reset()
    # tf.system.camera_reset()
    # tf.system.camera_zoom_to(-13)
    
    equilibrate(durations[2])
    # Repeat the filtering, to trim "escaped" interior particles that end up below the leading edge:
    filter_evl_to_animal_cap(leading_edge_z)
    
    unfreeze_leading_edge()
    
    # This last bit, quite long, removes most of the particle overlap as shown by the tension graphs at T=0
    equilibrate(durations[3])

    add_interior_bonds()
    adjust_global_evl_potentials()
    initialize_leading_edge_bending_resistance()
    initialize_paint_pattern()
    
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
    
def alt_initialize_embryo_with_config() -> None:
    global equilibration_time
    durations: list[int] = [100, 100, 100, 400]
    equilibration_time = sum(durations)
    
    setup_global_potentials()
    show_equilibrating_message()
    
    yolk_particle: tf.ParticleHandle = g.Yolk([5, 5, 5])
    yolk_particle.frozen = True
    yolk_particle.style = tf.rendering.Style()
    yolk_particle.style.color = g.Yolk.style.color
    initialize_bonded_edge()
    freeze_leading_edge_z()
    screenshot_true_zero()          # Need these? I wasn't sure when I did the big clean-up
    initialize_export_tasks()       # of this module, after not having used it in awhile.
    print("Equilibrating ring (frozen in z) (100)")
    equilibrate(durations[0])
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
    equilibrate(durations[1])
    show()
    print("Equilibrated full sphere of EVL cells, now filtering the excess")
    filter_evl_to_animal_cap(leading_edge_z)
    tf.show()
    print("Filtered down to animal cap, now putting the ring back")
    move_ring_z(destination=leading_edge_z)
    freeze_leading_edge_z()
    tf.show()
    print("Ring is back in place and unfrozen (in x and y only), now equilibrating a bit more (100)")
    equilibrate(durations[2])
    show()
    print("Equilibrated with z frozen, now re-filtering to remove escapers")
    filter_evl_to_animal_cap(leading_edge_z)
    tf.show()
    
    print("Removed escapers, now unfreezing and letting the edge relax (400)")
    unfreeze_leading_edge()
    equilibrate(durations[3])
    show()
    
    print("Now adding interior bonds and edge angles")
    add_interior_bonds()
    adjust_global_evl_potentials()
    initialize_leading_edge_bending_resistance()
    initialize_paint_pattern()
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
    
    setup_initial_cell_number_and_size()
    
    # ***** Choose one: *****
    alt_initialize_embryo_with_config()     # to run initialization with the ability to pause and examine each step
    # initialize_embryo_with_config()       # run init without pauses, as it will play when run in the sim from main
    # initialize_embryo_with_graph_boundary()     # Never created an "alt" for this one, but could if needed
    
    if cfg.show_equilibration:
        vx.make_movie()
    
    # Only after making the movie, so that these stills won't be included
    vx.final_result_screenshots()
    
    