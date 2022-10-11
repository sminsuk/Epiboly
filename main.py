"""Epiboly simulation

In progress: bond breaking & remaking (in PyCharm!)

First experiment with bond maintenance once they've been created. I think there are 2 ways to break bonds,
and you probably need both:
B1. automatically when they get stretched too far;
B2. stochastically among those that are *not* stretched too far.
And maybe 3 ways to make new bonds as you go:
M1. automatically whenever unbonded particles get within range (which would entail doing a neighbors
    check on every particle, every time step, basically redoing all the bonds from scratch each time).
    (Possibly is only likely to happen when other bonds break.)
M2. explicitly coordinated with bond breaking, so that either
        • whenever a bond breaks, you look for new neighbors to replace it with; but (in either case, but
            especially in the case of B2), there might not be another new neighbor available, which means
            you could end up with a more loosely connected network (does that matter?)
        • bonds only are broken and made in pairs, so that a bond won't break *unless* there's
            a new neighbor available to bond with in its place - including for bonds that have gotten too long.
            
As a first experiment, I think the best approach is B2 + M2. So don't actually consider distance at all
(although perhaps consider it as a factor in how likely a bond is to "attempt" a break). Once you have a
candidate bond for breaking, check both particles for neighbors that are close enough to bond to.
Calculate the energy change from the lost bond and the newly made bonds, and only do it when the energy
change is favorable. (Later, as a step 2, can add a stochastic Metropolis criterion just like in Potts.)

Changes:
• Implemented first step in algorithm: find the *furthest* bonded particle (the bond most likely to break), and
    the bondHandle connecting them. (Need to fix though: should select at random.)
• Found and remedied missing functionality in tf. Can't get a particle from a particle id (don't even bother, it
    would be expensive). Also no *direct* way, given a particle and one of its bondedNeighbors(), to access the
    bond connecting them. Wrote a utility function to do this and added to sharon_utils.
• Switched to processing *every* particle on each time step, not just one. It actually seems worryingly slow.
• Some refactoring, in particular neighbor-finding stuff.
• Import into PyCharm, do a fair bit of cleanup in response to all its errors and warnings, and start using
    type hinting.
    ToDo: I seem to have broken something! In iPython **only**, once bond maintenance starts,
     it never returns. Need to troubleshoot this! -- actually, I just wasn't waiting long enough. Too slow!
• Note, NEXT TODO:
    ToDo: implement the next step in breaking and making bonds: find *un*-bonded neighbors able to make
     new bonds. Candidate bond creations. Calculate the energy. ===== in progress =====
• Some more infrastructure thoughts: better to pass a list of dictionaries just like for exec_queue. Should make for
    a much cleaner dynamics module. No need for individual function setters and unsetters, because passing the list
    *is* the mechanism of turning on and off, all at once. Setter should replace old list with new list! This would also
    provide a natural way to set the order of operations from main calling code, not having to edit the module code.
    And finally, in both control modules I could pass "invoke_args" and "wait_args" dictionaries inside each task
    dictionary, obviating the need for lambdas and such. (Done for exec_queue; next, try it in dynamics!)
• But actually I don't even need exec_queue! I should apply this plan to the dynamics module, but ToDo: retire exec_queue
• Cleaned up the way I equilibrate, hold the leading edge with frozen z while doing so, and tweak the number
    of particles created until that works better. (Got it good enough for EMBRIO presentation; still needs more
    tweaking, though. Note James thoughts in notes from 2022-09-21 Group Meeting about equilibration methods!)
• More ToDo: Whip up some actin ring contraction, soon!

Note to self -- to do next:
• The gaps play havoc once a force is applied, especially since Expt. 13, now that the neighbor distance
    criterion is much smaller. I think I need to work on the numbers of particles and the initial layout.
    I need a way to fill in all the gaps, yet not distort the ring. Maybe, let the ring equilibrate first,
    then try freezing those particles; then add the interior particles and let them equilibrate. And for as
    long as it takes. With nowhere else to go, maybe they will fill in the gaps. And I can play around and
    get exactly the right number of both types.
• But maybe what comes first is, exploring the param space on the bond potentials, and figuring out how
    to allow tissue remodeling (bond making and breaking). Then maybe easier and more productive
    to fix the above in that context.

Carry-forward reminder of a couple other things still ToDo:
• get keypress wait condition working? Doesn't crash kernel anymore but still won't detect. (Of course, as a
    workaround for my simple use case, you always can just pause)
• continue to improve encapsulation of globals by writing classes/modules. And better code organization.

"""

import math
import time

from epiboly_init import *  # Tissue Forge initialization and global ParticleType class/instance creation

from biology import bond_maintenance as bonds,\
    microtubules as mt
from control_flow import dynamics as dyn, \
    exec_queue as xq, \
    exec_tests as xt
import neighbors as nbrs
import sharon_utils as su

from control_flow.interactive import is_interactive
if is_interactive():
    # Importing this at the global level causes PyCharm to keep deleting my epiboly_init import,
    # I guess because it thinks it's redundant; something about the "import *".
    # Doing it here prevents that. And this is only needed in interactive sessions.
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
    # (Note: moved the definition of these constants to the global epiboly_init module, and tweaked it further.)
    # Could conceivably even have different spring constants on the potentials in the two cases; I have
    # tweaked that to be best for equilibration; when ready to "start the biology running", can easily
    # change them at that time.
    # noinspection PyUnreachableCode
    if True:
        # new method:
        # (gets list of plain python list[3])
        vectors = su.random_nd_spherical(npoints=num_spherical_positions, dim=3)
        edge_margin = edge_margin_interior_points
    else:
        # or alternatively, old method using tf built-in (and transform to match the output type of
        # the new method, so I can test either way): 
        # noinspection PyTypeChecker
        vectors = tf.random_points(tf.PointsType.Sphere.value, num_spherical_positions)
        vectors = [vector.as_list() for vector in vectors]
        edge_margin = edge_margin_interior_points
    
    random_points_time = time.perf_counter()
    
    # Filter to include only the ones inside the existing ring of LeadingEdge particles
    # I could probably calculate this more correctly, but for now just subtract a little bit
    # from leading_edge_phi (eyeball it) so that the two particle types don't overrun each other.
    # This is in radians so 5 degrees ~ 0.09. (See edge_margin values above.)
    filtered_vectors = [vector for vector in vectors
                        if su.spherical_from_cartesian(vector)[2] < leading_edge_phi - edge_margin]
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
        return big_particle.position + su.vec(vector) * scale
    
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
    for particle in Little.items():
        particle.style = tf.rendering.Style()
        particle.style.setColor("cornflowerblue")
        particle.style.visible = True
    
    finished = time.perf_counter()
    print("generating unit sphere coordinates takes:", random_points_time - start, "seconds")
    print("filtering takes:", filtered_time - random_points_time, "seconds")
    print("scaling (and converting to list) takes:", scaled_time - filtered_time, "seconds")
    print("filtering/scaling/converting all together take:", scaled_time - random_points_time, "seconds")
    print("instantiating takes:", finished - scaled_time, "seconds")

def add_interior_bonds():
    print("Bonding interior particles.")
    for particle in Little.items():
        # Bond to all neighbors not already bonded to        
        neighbors = nbrs.get_non_bonded_neighbors(particle)
        for neighbor in neighbors:
            tf.Bond.create(small_small_attraction_bonded, neighbor, particle)
    
    print(f"Created {len(tf.BondHandle.items())} bonds.")

def initialize_bonded_edge():
    def create_ring():
        print("Generating leading edge particles.")
        
        # Where the edge should go
        leading_edge_phi = su.phi_for_epiboly(epiboly_percentage=epiboly_initial_percentage)
        #         print("leading edge: phi =", math.degrees(leading_edge_phi))
        
        # some basic needed quantities
        scale = Big.radius + LeadingEdge.radius
        r_latitude = math.sin(leading_edge_phi) * scale
        
        # (gets list of plain python list[2] - unit vectors)
        # (note, unit circle, i.e., size of the unit sphere *equator*, not the size of that latitude circle)
        unit_circle_vectors = su.random_nd_spherical(npoints=num_leading_edge_points, dim=2)
        
        # make 3D and scaled
        z_latitude = math.cos(leading_edge_phi) * scale
        latitude_center_position = big_particle.position + su.vec([0, 0, z_latitude])
        
        def final_position(unit_vector):
            return (latitude_center_position
                    + su.vec([*unit_vector, 0]) * r_latitude)
        
        final_positions = [final_position(vector).as_list() for vector in unit_circle_vectors]
        
        LeadingEdge.factory(positions=final_positions)
        
        # Give these each a Style object so I can access them later
        for particle in LeadingEdge.items():
            particle.style = tf.rendering.Style()
            particle.style.setColor("gold")
            particle.style.visible = True
        
        return leading_edge_phi
    
    def create_bonds():
        print("Bonding ring particles.")
        
        def theta(particle):
            # theta relative to the animal/vegetal axis
            r, theta, phi = particle.sphericalPosition(origin=big_particle.position)
            return theta
        
        # Sort all the new particles on theta, into a new list (copy, not live)
        sorted_particles = sorted(LeadingEdge.items(), key=theta)
        
        # Now they can just be bonded in the order in which they are in the list
        previous_particle = sorted_particles[-1]  # last one
        for particle in sorted_particles:
            #             print("binding particles with thetas:",
            #                   math.degrees(theta(previous_particle)),
            #                   math.degrees(theta(particle)))
            tf.Bond.create(small_small_attraction_bonded, previous_particle, particle)
            previous_particle = particle
    
    leading_edge_phi = create_ring()
    create_bonds()
    return leading_edge_phi

def initialize_particles() -> None:
    """In two steps. First create simple ring of bonded LeadingEdge, then fill in the interior particles."""
    leading_edge_phi = initialize_bonded_edge()
    initialize_interior(leading_edge_phi)

def replace_all_small_small_potentials(new_potential):
    """Wipes out old potential, replaces with new, for all small-small interactions"""
    tf.bind.types(new_potential, LeadingEdge, LeadingEdge)
    tf.bind.types(new_potential, LeadingEdge, Little)
    tf.bind.types(new_potential, Little, Little)

########################## main ##########################

# Potentials, bound at the level of types:
#
# Large-small: LJ, originally max = equilibrium distance = sum of radii (for only repulsion), but then
# expanded max to include attraction for the purposes of bringing particles down to the surface during
# setup. Setup at first with this attraction so all particles go to the right place. Then once the
# biology starts, can remove that from interior particles, because they should not remain attached.
# Should eventually switch this to Morse (or maybe harmonic) for ease of use (and consistency),
# if I ever need to change it again.
# 
# Small-small (both types, to themselves and to each other): 
# During setup: harmonic with repulsion only (max = equilibrium distance = sum of radii, so potential
# applied only inside the equilibrium distance). 
# Then: a second, attractive, potential will be added via bonds.

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

# Attractive potential to be used in a subsequent step. For now, same as repulsion except for the range.
# (Probably won't be doing anything like this on the type level anymore)
small_small_attraction_by_type = tf.Potential.harmonic(r0=r0,
                                                       k=0.1,
                                                       min=r0,
                                                       max=r0 + 1
                                                       )

replace_all_small_small_potentials(new_potential=small_small_repulsion)

big_particle = Big([5, 5, 5])

# Attractive potential to be used only in bonded interactions. Like before, only huge max!
small_small_attraction_bonded = tf.Potential.harmonic(r0=r0,
                                                      k=7.0,
                                                      min=r0,
                                                      max=6
                                                      )

######## short named functions to use in execute_sequentially()
def random_initialization():
    # proxy for initialize_particles(), because we want to call that *before* invoking the queue,
    # not inside it. Gave this a name that serves as a good output message, better than 
    # "calling <lambda>" which is what you get if you pass "None".
    pass

########

# Want initialize_particles() to run *before* user clicks Run on the simulator, so run it
# outside of the sequential queue:
initialize_particles()

xq.execute_sequentially([
        # {"invoke": random_initialization,
        #  "wait": xt.is_equilibrated,
        #  "wait_args": {"epsilon": 0.05}
        #  # "verbose": False,
        #  },
        # # {"invoke": nbrs.paint_neighbors},
        # # Can't use this one because on_keypress() doesn't seem to work!
        # # {"invoke": xt.setup_keypress_detection,
        # #      "wait": xt.keypress_event,
        # #      "verbose": False},
        # {"invoke": add_interior_bonds},
        # # Removing big-little attraction: I don't want this after all. (But great for testing "invoke_args"!)
        # # {"invoke": tf.bind.types,
        # #  "invoke_args": {"p": big_small_repulsion_only, "a": Big, "b": Little},
        # #  },
        # # {"invoke": toggle_visibility},
        # # {"invoke": toggle_visibility},
        ])

########################## interactive ##########################

def plots():
    big_small_pot.plot(potential=True, force=True, ymin=-1e2, ymax=1e2)
    # small_small_LJ.plot(potential=True, force=True, ymin=-1e1, ymax=1e1)
    small_small_repulsion.plot(potential=True, force=True, ymin=-0.1, ymax=0.01)
    small_small_attraction_bonded.plot(potential=True, force=True, ymin=-0.001, ymax=0.01, min=0.28, max=0.34)
    # This fails ("Maximum allowed size exceeded"):
    # combo = small_small_repulsion + small_small_attraction_bonded
    # combo.plot(potential=True, force=True, ymin=-1e1, ymax=1e1)

def reset_camera():
    # tf.system.camera_view_front()
    tf.system.camera_zoom_to(-12)

def equilibrate_to_leading_edge(steps: int = 1):
    freeze_leading_edge(True)
    tf.step(steps)
    freeze_leading_edge(False)
    print(f"Leading edge is {'' if xt.leading_edge_is_equilibrated() else 'not '}equilibrated")
    
reset_camera()
print("Invisibly equilibrating; simulator will appear shortly...")
equilibrate_to_leading_edge(300)
add_interior_bonds()

dyn.execute_repeatedly(tasks=[
        {"invoke": mt.update_tangent_forces,
         "args": {"magnitude": 5}
         },
        # {"invoke": bonds.maintain_bonds,
        #  "args": {"ptypes": [Little,
        #                      LeadingEdge,
        #                      ]
        #           }
        #  },
        ])

print("press space to starting stretching, then close window to stop...")
tf.show()

mt.remove_tangent_forces()
dyn.execute_repeatedly()    # turn off everything
print("Now press space to release")


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
