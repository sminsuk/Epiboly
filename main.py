"""Epiboly simulation

In progress: bond breaking & remaking

Third experiment with bond maintenance: now, with stress relaxation.

Break bonds with a probability proportional to tension. Probability a linear function of energy (between r0 and
a multiple of r0).

Make new bonds whenever particles approach close enough to qualify as "nearby", same as the initial setup.
"""

import math
import time

import tissue_forge as tf
from epiboly_init import Little, Big, LeadingEdge   # Tissue Forge initialization and global ParticleType creation
import config as cfg

from biology import bond_maintenance as bonds,\
    microtubules as mt
from control_flow import dynamics as dyn, \
    exec_tests as xt
import neighbors as nbrs
from utils import tf_utils as tfu,\
    global_catalogs as gc

from control_flow.interactive import is_interactive, toggle_visibility
if is_interactive():
    # Importing this at the global level causes PyCharm to keep deleting my epiboly_init import,
    # (ToDo: see if this is still true, now that I got rid of "import *".)
    # I guess because it thinks it's redundant; something about the "import *".
    # Doing it here prevents that. And this is only needed in interactive sessions.
    # (ToDo: also, turns out is_interactive() functionality is now provided in tf, I can probably lose mine.)
    from control_flow.interactive import *
    print("Interactive module imported!")

def freeze_leading_edge(frozen: bool = True):
    for particle in LeadingEdge.items():
        particle.frozen_z = frozen

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
    
    # Tweak parameters: density (number of points requested), edge_margin (gap left when placing
    # interior cells so they aren't placed too close to ring, and so they have room to expand into),
    # for these two approaches (tf.random_points()
    # and su.random_nd_spherical()). Goal is to get an initial layout (after equilibration) that touches
    # the bonded ring without deforming it. random_nd_spherical() is a bit less uniform so has more holes to
    # start with, hence more bunching elsewhere, hence tends to expand more unevenly during equilibration.
    # Note that increasing the density but also the size of the gap, can balance out to give the same
    # number of particles, but with different initial distribution and equilibration dynamics.
    # For awhile had different numbers of point and edge_margins for the two, but ultimately came around
    # to the same values for each (2050 and 0.15), so I guess those must be the "right" values.
    # (Note: moved the definition of these constants to the config module, and tweaked it further.)
    # Could conceivably even have different spring constants on the potentials in the two cases; I have
    # tweaked that to be best for equilibration; when ready to "start the biology running", can easily
    # change them at that time.
    # noinspection PyUnreachableCode
    if True:
        # new method:
        # (gets list of plain python list[3])
        vectors = tfu.random_nd_spherical(npoints=cfg.num_spherical_positions, dim=3)
        edge_margin = cfg.edge_margin_interior_points
    else:
        # or alternatively, old method using tf built-in (and transform to match the output type of
        # the new method, so I can test either way): 
        # noinspection PyTypeChecker
        vectors = tf.random_points(tf.PointsType.Sphere.value, cfg.num_spherical_positions)
        vectors = [vector.as_list() for vector in vectors]
        edge_margin = cfg.edge_margin_interior_points
    
    random_points_time = time.perf_counter()
    
    # Filter to include only the ones inside the existing ring of LeadingEdge particles
    # I could probably calculate this more correctly, but for now just subtract a little bit
    # from leading_edge_phi (eyeball it) so that the two particle types don't overrun each other.
    # This is in radians so 5 degrees ~ 0.09. (See edge_margin values above.)
    filtered_vectors = [vector for vector in vectors
                        if tfu.spherical_from_cartesian(vector)[2] < leading_edge_phi - edge_margin]
    num_particles = len(filtered_vectors)
    print(f"Creating {num_particles} particles.")
    filtered_time = time.perf_counter()
    
    # Transform unit sphere to sphere with radius = big particle radius + small particle radius
    # (i.e. particles just touching) and concentric on the big particle.
    # And even though plain python lists[3] is what we ultimately need, easiest to do the math
    # by converting them to fVector3 here.
    scale: float = Big.radius + LeadingEdge.radius
    
    # (note, if testing with the old random_points() above, don't use su.vec because the
    # vectors here are already in that form, and it will crash.)
    # final_position = lambda vector: big_particle.position + su.vec(vector) * scale
    def final_position(vector):
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

def add_interior_bonds():
    print("Bonding interior particles.")
    for particle in Little.items():
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
        scale = Big.radius + LeadingEdge.radius
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
        
        def theta(particle):
            # theta relative to the animal/vegetal axis
            r, theta, phi = particle.sphericalPosition(origin=big_particle.position)
            return theta

        # Use for each of the bonds we'll create here
        r0 = LeadingEdge.radius * 2
        small_small_attraction_bonded = tf.Potential.harmonic(r0=r0,
                                                              k=cfg.harmonic_edge_spring_constant,
                                                              min=r0,
                                                              max=6
                                                              )

        # Sort all the new particles on theta, into a new list (copy, not live)
        sorted_particles = sorted(LeadingEdge.items(), key=theta)
        
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

    def spherical_coordinates_theta(particle: tf.ParticleHandle) -> float:
        # theta relative to the animal/vegetal axis
        r, theta, phi = particle.sphericalPosition(origin=big_particle.position)
        return theta

    if not cfg.angle_bonds_enabled:
        return
    
    edge_angle_potential: tf.Potential = tf.Potential.harmonic_angle(k=cfg.harmonic_angle_spring_constant,
                                                                     theta0=cfg.harmonic_angle_equilibrium_value(),
                                                                     tol=cfg.harmonic_angle_tolerance)

    # Sort all the leading edge particles on spherical coordinate theta, into a new list (copy, not live).
    # This is just like when we made the bonds. Now that we have the bonds, we COULD follow the links from
    # particle to particle, but it's easier to just sort the list of particles by theta again.
    sorted_particles = sorted(LeadingEdge.items(), key=spherical_coordinates_theta)

    # Now they can just be processed in the order in which they are in the list
    previous_particle = sorted_particles[-1]  # last one
    before_previous_particle = sorted_particles[-2]  # 2nd-to-last
    for particle in sorted_particles:
        tf.Angle.create(edge_angle_potential, before_previous_particle, previous_particle, particle)
        before_previous_particle = previous_particle
        previous_particle = particle

def initialize_particles() -> None:
    """In two steps. First create simple ring of bonded LeadingEdge, then fill in the interior particles."""
    leading_edge_phi = initialize_bonded_edge()
    initialize_interior(leading_edge_phi)

def replace_all_small_small_potentials(new_potential):
    """Wipes out old potential, replaces with new, for all small-small interactions"""
    tf.bind.types(new_potential, LeadingEdge, LeadingEdge)
    tf.bind.types(new_potential, LeadingEdge, Little)
    tf.bind.types(new_potential, Little, Little)

# ######################### main ##########################

print(f"tissue-forge version = {tf.version.version}")
print(f"CUDA installed: {'Yes' if tf.has_cuda else 'No'}")

# Potentials, bound at the level of types:
#
# Large-small: LJ, originally max = equilibrium distance = sum of radii (for only repulsion), but then
# expanded max to include attraction for the purposes of bringing particles down to the surface during
# setup. Setup at first with this attraction so all particles go to the right place. Then, once the
# biology starts, can remove that from interior particles, because they should not remain attached.
# Should eventually switch this to Morse (or maybe harmonic) for ease of use (and consistency),
# if I ever need to change it again.
# 
# Small-small (both types, to themselves and to each other): 
# During setup: harmonic with repulsion only (max = equilibrium distance = sum of radii, so potential
# applied only inside the equilibrium distance). 
# Then: this will be replaced by a new potential added via bonds, handling both attraction and repulsion.

# Big-small equilibrium distance = 3.15
# Note, with LJ, to adjust well-depth with minimal change to equilibrium distance, keep A/B constant.
big_small_pot = tf.Potential.lennard_jones_12_6(min=0.275, max=5, A=9.612e6, B=19608)
tf.bind.types(big_small_pot, Big, LeadingEdge)

# Also bind to Little (interior) particles, just during equilibration. Once the biology starts, will remove it.
tf.bind.types(big_small_pot, Big, Little)

# for the eventual removal
big_small_repulsion_only = tf.Potential.lennard_jones_12_6(min=0.275, max=3.15, A=9.612e6, B=19608)

r0 = LeadingEdge.radius * 2

# All small particles repel each other all the time, inside r0
small_small_repulsion = tf.Potential.harmonic(r0=r0,
                                              k=5.0,
                                              max=r0
                                              )

replace_all_small_small_potentials(new_potential=small_small_repulsion)

big_particle = Big([5, 5, 5])

initialize_particles()

# ######################### interactive ##########################

def plots():
    big_small_pot.plot(potential=True, force=True, ymin=-1e2, ymax=1e2)
    # small_small_LJ.plot(potential=True, force=True, ymin=-1e1, ymax=1e1)
    small_small_repulsion.plot(potential=True, force=True, ymin=-0.1, ymax=0.01)
    # Can't do this one anymore because I made the declaration local:
    # small_small_attraction_bonded.plot(potential=True, force=True, ymin=-0.001, ymax=0.01, min=0.28, max=0.34)
    # This fails ("Maximum allowed size exceeded"):
    # combo = small_small_repulsion + small_small_attraction_bonded
    # combo.plot(potential=True, force=True, ymin=-1e1, ymax=1e1)

def reset_camera():
    tf.system.camera_view_front()
    tf.system.camera_zoom_to(-12)

def equilibrate_to_leading_edge(steps: int = 1):
    freeze_leading_edge(True)
    tf.step(steps)
    freeze_leading_edge(False)
    print(f"Leading edge is {'' if xt.leading_edge_is_equilibrated() else 'not '}equilibrated")
    
# Future note: I'd like to be able to enable lagging here, programmatically, but it's missing from the API.
# TJ will add it in a future release.
reset_camera()
print("Invisibly equilibrating; simulator will appear shortly...")
equilibrate_to_leading_edge(300)
add_interior_bonds()
initialize_leading_edge_bending_resistance()

# toggle_visibility()
# toggle_visibility()
dyn.execute_repeatedly(tasks=[
        # {"invoke": mt.update_tangent_forces,
        #  "args": {"magnitude": 5}
        #  },
        {"invoke": mt.apply_even_tangent_forces,
         "args": {"magnitude": 5}
         },
        {"invoke": bonds.maintain_bonds},
        ])

tf.show()
dyn.execute_repeatedly(tasks=[
        # {"invoke": mt.update_tangent_forces,
        #  "args": {"magnitude": 5}
        #  },
        {"invoke": mt.apply_even_tangent_forces,
         "args": {"magnitude": 5}
         },
        {"invoke": bonds.maintain_bonds,
         "args": {
                  #     ###For making/breaking algorithm:
                  # "k_adhesion": 1.0,
                  # "k_neighbor_count": 1.0,
                  # "k_angle": 1.0,
                  # "k_edge_neighbor_count": 1.0,
                  # "k_edge_angle": 1.0,
                  #     ###For relaxation:
                  # "relaxation_saturation_factor": 2,
                  # "viscosity": 0.001
                  }
         },
        ])
# mt.remove_tangent_forces()

# tf.step()
# while not xt.is_equilibrated(epsilon=0.01):
#     # print("stepping!")
#     tf.step()
#     tf.Simulator.redraw()  # will this work once irun() is working? Do irun() before the loop?

# tf.step(50)
#
# toggle_radius()
# toggle_radius()
tf.show()
