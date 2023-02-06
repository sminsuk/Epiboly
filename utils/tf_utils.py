"""Tissue Forge utilities

These are general purpose utility functions for Tissue Forge, not specific to any particular simulation.
"""
import math
import numpy as np
import sys
import traceback
from typing import Optional

from utils import global_catalogs as gc
import tissue_forge as tf

def cartesian_from_spherical(sphere_vec):
    """Given a vector in spherical coords (with angles in radians), return the cartesian equivalent.

    sphere_vec: 3-item sequence (fVector3, ordinary List, etc.), vector in spherical coords [r, theta, phi] in radians
    returns: fVector3, vector in cartesian coords [x, y, z]
    """
    r, theta, phi = sphere_vec
    
    dxy = r * math.sin(phi)  # length of projection of vector onto the xy plane
    # print("dxy =", dxy)
    dx = dxy * math.cos(theta)
    dy = dxy * math.sin(theta)
    dz = r * math.cos(phi)
    return tf.fVector3([dx, dy, dz])

def cartesian_from_spherical_degs(sphere_vec):
    """Given a vector in spherical coords (with angles in degrees), return the cartesian equivalent.

    sphere_vec: 3-item sequence (fVector3, ordinary List, etc.), vector in spherical coords [r, theta, phi] in degrees
    returns: fVector3, vector in cartesian coords [x, y, z]
    """
    r, theta_degs, phi_degs = sphere_vec
    return cartesian_from_spherical([r, math.radians(theta_degs), math.radians(phi_degs)])

def spherical_from_cartesian(cartesian_vec):
    """Given a vector in cartesian coords, return the spherical equivalent (with angles in radians).

    cartesian_vec: 3-item sequence (fVector3, tuple, list, etc.), vector in cartesian coords [x, y, z]
    returns: fVector3, vector in spherical coords [r, theta, phi] in radians
    """
    dx, dy, dz = cartesian_vec
    r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    dxy = math.sqrt(dx ** 2 + dy ** 2)  # length of projection of vector onto the xy plane
    theta = 0 if dxy == 0 else math.copysign(math.acos(dx / dxy), dy)
    phi = 0 if r == 0 else math.acos(dz / r)
    return tf.fVector3([r, theta, phi])

def spherical_degs_from_cartesian(cartesian_vec):
    """Given a vector in cartesian coords, return the spherical equivalent (with angles in degrees).

    cartesian_vec: 3-item sequence (fVector3, tuple, list, etc.), vector in cartesian coords [x, y, z]
    returns: fVector3, vector in spherical coords [r, theta, phi] in degrees
    """
    r, theta, phi = spherical_from_cartesian(cartesian_vec)
    return tf.fVector3([r, math.degrees(theta), math.degrees(phi)])

def random_nd_spherical(npoints: int, dim: int) -> list[list[float]]:
    """Generate uniform distributed points on n-dimensional sphere

    npoints: number of points to generate
    dim: number of dimensions (2 for circle, 3 for sphere, etc.)
    returns: a list[npoints] of plain python lists[dim]
    """
    # Generate uniformly distributed unit vectors, as an array[dim][npoints] of scalar components.
    # Source of this algorithm: https://stackoverflow.com/a/33977530 - includes link to
    # Wolfram MathWorld article on the reasoning. They used np.random.randn(), but the linked
    # doc page for that says, "New code should use the standard_normal method of a default_rng()
    # instance instead".
    #
    # (Also note they used ndim for the argument name, but that's confusing
    # because it's different from a standard numpy ndim. These functions return a 2-dimensional
    # array no matter what dim is; dim is the *size* of the array along one axis. For a sphere,
    # dim = 3 so we have a size of (3, npoints), and an array of 3D coordinates is a 2-dimensional
    # array. For a circle, dim = 2 so we have a size of (2, npoints), and an array of 2D
    # coordinates is also a 2-dimensional array, just with only 2 rows for x and y instead of
    # 3 for x, y, and z. Or: "dim" is the number of dimensions of the geometrical shape we are
    # generating, "ndim" is the number of dimensions of the array that holds a list of coordinates.)
    vec = np.random.default_rng().standard_normal(size=(dim, npoints))
    vec /= np.linalg.norm(vec, axis=0)
    
    # generate an iterator of dim-tuples
    position_it = zip(*vec)
    
    # a list[npoints] of dim-tuples
    positions = list(position_it)
    
    # return a list[npoints] of plain python lists[dim]
    return [list(position) for position in positions]

# for printing in color
redcolor = "\033[91m"
bluecolor = "\033[94m"
endcolor = "\033[0m"

# rgb colors for Style objects:
cornflower_blue = tf.fVector3([0.12743769586086273, 0.3005438446998596, 0.8468732833862305])
gold = tf.fVector3([1.0, 0.6795425415039062, 0.0])
white = tf.fVector3([1.0, 1.0, 1.0])
gray = tf.fVector3([0.5, 0.5, 0.5])

def exception_handler():
    """General exception handler to be used inside TF events.

    Within TF events, python won't catch exceptions on its own!
    Call this in the "except:" clause, inside your event's invoke_method:
    
    def my_func(event):
        try:
            [do something]
        except Exception:
            # display the error and stack trace, which python fails to do
            tf_utils.exception_handler()
            
            # python also fails to exit the program if an error occurs during a TF event;
            # but at least you can cancel the event instead of calling a broken event repeatedly
            event.remove()
            
            # That stops custom events, but it won't stop the script (and even calling sys.exit() won't work
            # from here), and it won't stop TF timestepping. To do that, set a global signal so that the main script
            # (outside this invoke method) can exit for you. (Only works with tf.step(), not tf.show(); check
            # the signal in a loop between calls to tf.step(), and take appropriate action if detected.)
            global some_flag
            some_flag = True
            
            # TF documentation says invoke_method should return 1 on error. Unclear whether this has any effect.
            return 1
        return 0
        
    tf.event.on_time(period=[some value], invoke_method=my_func)
    """
    (exc_type, exc_value, exc_traceback) = sys.exc_info()
    print(redcolor, exc_type, exc_value, endcolor)
    traceback.print_tb(exc_traceback)

# Hmm. May not need these two? At least in my own original use case, now that I know how to get a particleHandle
# from a particle id, I may not have needed to write these.
def bonds_between(p1, p2):
    """returns a list of all bonds connecting two given particles.

    p1, p2: particleHandle
    returns: list of BondHandle objects. May be empty or of any length, since multiple bonds are possible
        between any particle pair.
    """
    p1p2bonds = [bond for bond in p1.bonds
                 if p2.id in bond.parts]
    return p1p2bonds

def bond_between(p1, p2, verbose=True):
    """returns one bond connecting two given particles.

    p1, p2: particleHandle
    verbose: boolean. Set to False to suppress warnings. If True, warns when more than one bond found.
    returns: If there are no bonds between the two particles, returns None.
        Otherwise, returns BondHandle of the first bond found.
    """
    p1p2bonds = bonds_between(p1, p2)
    if verbose and len(p1p2bonds) > 1:
        print(bluecolor + "Warning, more than 1 bond found. To retrieve them all, use bonds_between()" + endcolor)
    return None if not p1p2bonds else p1p2bonds[0]

def particle_from_id(id: int, type: tf.ParticleType = None) -> Optional[tf.ParticleHandle]:
    """Temporary work around, delete after issue is fixed in future Tissue Forge release

    returns: None if particle not found

    DEPRECATED: Hopefully do not need.
    
    Correct way is to call the ParticleHandle constructor with the id:
        phandle = tf.ParticleHandle(id)
    But currently, tf barfs, telling me that's not an existing function overload:
    
    Exception in maintain_bonds():
    <class 'TypeError'> Wrong number or type of arguments for overloaded function 'new_ParticleHandle'.
    Possible C/C++ prototypes are:
    TissueForge::ParticleHandle::ParticleHandle()
    TissueForge::ParticleHandle::ParticleHandle(int const &,int const &)
    
    Probably a missing constructor? TJ says use the second one shown. Second int arg is the id of the ParticleType.
    Of course, can't get the ParticleType directly without knowing the particle! So I loop through all the particles
    looking for it, and at that point I'll have my ParticleHandle so won't actually need to get the ParticleType or
    use the constructor. If this turns out to be too slow, I'll set up my own mapping at the time of particle
    creation.
    """
    p: Optional[tf.ParticleHandle]
    
    if type is not None:
        # value was provided, so do this the easy way:
        p = tf.ParticleHandle(id, type.id)
        assert p.type() is type, "Found particle, does not match expected type"
    else:
        # type not provided, so have to just search for the particle
        p = None
        for phandle in tf.Universe.particles:
            if phandle.id == id:
                p = phandle
                break
        
        if p is None:
            # not found
            return None
    
    assert p.id == id, "Got a particle, but the id is not as expected"
    return p

def bond_parts(b: tf.BondHandle) -> tuple[Optional[tf.ParticleHandle], Optional[tf.ParticleHandle]]:
    """Given a bondHandle, get particleHandles for the two bonded particles

    This is like bondHandle.parts, except .parts only returns particle ids, not particleHandles.
    Note, TJ considering changing that.

    b: bondHandle of an *active* bond
    returns: tuple of two particleHandles
    
    future: checking .active is not supposed be needed; those are supposed to be filtered out before you see them.
    Possibly the flag may not even be accessible in future versions.
    """
    assert b.active, "Can't get particles from an inactive bond!"
    # print(f"BondHandle.id = {b.id}")
    id1, id2 = b.parts
    
    # This didn't work because of bug in tissue forge. Use this eventually, after it's fixed:
    # p1 = tf.ParticleHandle(id1)
    # assert p1.id == id1, f"tf.ParticleHandle({id1}) gave a particle with id {p1.id}"
    # p2 = tf.ParticleHandle(id2)
    # print(f"found particles {p1.id} and {p2.id}")
    
    # This worked, but was ridiculously (and predictably) slow
    # p1 = particle_from_id(id1)
    # p2 = particle_from_id(id2)
    
    # Faster and better
    gcdict = gc.particles_by_id
    assert id1 in gcdict, f"Particle {id1} missing from global catalog!"
    assert id2 in gcdict, f"Particle {id2} missing from global catalog!"
    p1 = gcdict[id1]["handle"]
    p2 = gcdict[id2]["handle"]

    return p1, p2

def other_particle(p: tf.ParticleHandle, b: tf.BondHandle) -> tf.ParticleHandle:
    """Given a particle and one of its bonds, get the particle bonded to"""
    if tf.version.version != "0.0.1":
        print(f"p = {p}, type(p) = {type(p)}")
    assert p.id in b.parts, f"Bond {b.id} does not belong to particle {p.id}"
    id1, id2 = b.parts
    gcdict = gc.particles_by_id
    if tf.version.version != "0.0.1":
        print(f"id1 = {id1}, p = {p}, p.id = {p.id}")
    # Todo: In 0.0.2, this will crash because .parts now returns ParticleHandles, not particle ids!
    if id1 == p.id:
        return gcdict[id2]["handle"]
    else:
        return gcdict[id1]["handle"]
    
def bond_distance(b: tf.BondHandle) -> float:
    """"Get the distance between the two particles of a bond, i.e. the "length" of the bond

    b: bondHandle of an *active* bond
    returns: float
    
    future: checking .active is not supposed be needed; those are supposed to be filtered out before you see them.
    Possibly the flag may not even be accessible in future versions.
    """
    p1, p2 = bond_parts(b)
    return p1.distance(p2)

def cross(v1: tf.fVector3, v2: tf.fVector3) -> tf.fVector3:
    """ToDo: Once version 0.0.2 is working, it has built-in cross!"""
    return tf.fVector3([v1.y() * v2.z() - v1.z() * v2.y(),
                        v1.z() * v2.x() - v1.x() * v2.z(),
                        v1.x() * v2.y() - v1.y() * v2.x()])

def truncate(dotprod: float) -> float:
    """Restrict dot product to the range [-1, 1]

    Dot products can be anything; but in certain cases (dot product of two unit vectors), result should never
    be outside this range, and the angle should be retrievable by taking acos(dot_product).
    If dot product is outside that range, acos() will throw an exception.

    This issue arises in particular when taking the dot product of a unit vector with itself, or when the two
    unit vectors are exactly 180 deg apart. These should come out to exactly +/- 1.0. But in these cases,
    tf.fVector3.dot() produces an imprecise result that can be too large, and this will crash acos().
    """
    if dotprod > 1.0:
        return 1.0
    elif dotprod < -1.0:
        return -1.0
    else:
        return dotprod

def angle_from_unit_vectors(unit_vector1: tf.fVector3, unit_vector2: tf.fVector3) -> float:
    """ToDo: Once version 0.0.2 is working, it has built-in angle from vectors!"""
    cosine: float = truncate(unit_vector1.dot(unit_vector2))
    radians: float = math.acos(cosine)

    # alternative, using function from this module, based on table look-up. Faster? Apparently not!
    # radians: float = arccos(cosine)
    
    return radians

def angle_from_particles(p1: tf.ParticleHandle, p_vertex: tf.ParticleHandle, p2: tf.ParticleHandle) -> float:
    vector1: tf.fVector3 = p1.position - p_vertex.position
    vector2: tf.fVector3 = p2.position - p_vertex.position
    uvec1: tf.fVector3 = vector1.normalized()
    uvec2: tf.fVector3 = vector2.normalized()
    return angle_from_unit_vectors(uvec1, uvec2)

def _setup_cos_table() -> None:
    global _cos_table_by_millis
    
    # Test: in debugger, just examine the non-linear relationship between angle and cosine
    cos_to_angle_map_by_degrees: dict[float: tuple[int, float]] = {}
    degrees: int
    for degrees in range(181):
        radians: float = math.radians(degrees)
        cosine: float = math.cos(radians)
        cos_to_angle_map_by_degrees[cosine] = (degrees, radians)
    
    # Now set up the values to use to make sure we get at least 1-degree resolution almost everywhere
    cos_to_angle_map_by_millis: dict[int: tuple[float, float]] = {}
    millis: int
    for millis in range(1000, -1001, -1):
        cosine: float = millis / 1000
        radians: float = math.acos(cosine)
        degrees: float = round(math.degrees(radians), 2)
        cos_to_angle_map_by_millis[millis] = (degrees, radians)
    
    _cos_table_by_millis = cos_to_angle_map_by_millis

def arccos(cosine: float) -> float:
    """Given a value in the range [-1, 1] (a cosine), return the approximate angle, in range [0, π]"""
    millis: float = round(cosine * 1000)
    degrees, radians = _cos_table_by_millis[millis]
    return radians

def _test_arccos() -> None:
    for millis in range(1000, -1001, -100):
        cosine: float = millis / 1000
        print(f"arccos({cosine}) = {arccos(cosine)}, or {math.degrees(arccos(cosine))} degrees")
    
# Run once at load time
_cos_table_by_millis: dict[int: tuple[float, float]] = {}
_setup_cos_table()
# _test_arccos()

def _test_harmonic_angle() -> None:
    """For testing of issues with angle potential. Fails for certain values of theta0 parameter.
    
    Fascinating result:
    With accepting the default tolerance, theta0=42-138 degrees (note: centered around 90°) always works. 0° and 180°
    also always works. But depending on the value of k, the ranges in between those, don't work.
    For k < 2 (I tried 1.9), everything works. For k=2.0, it fails for theta0=41° and for 139°-140°. As k increases,
    the upper bound of the higher failure range goes up, and the lower bound of the lower failure range goes down,
    i.e. the failure ranges get larger at their extreme ends until only 0° and 180° remain. For k=5.0 (the value
    I've been using), it fails for theta0=17°-41° and 139°-163°. (The failure ranges
    max out at 0 and 180 asymptotically; I tried up to k = 5000 and couldn't get rid of 0° and 180°, but they might
    disappear eventually.)
    
    My theta0 is defined in bond_maintenance.harmonic_angle_equilibrium_value(); it equals
    math.pi - (two_pi / len(LeadingEdge.items())). I.e. ranging from just below 180°, down to, say,
    145° when there are 10 particles left in the ring. So I need those higher values of theta0 to work.
    
    Can I improve this behavior by tweaking other params? I'd like to keep k=5 for now if I can.
    Making min greater than its default of zero changes the specific ranges, but not in a helpful way.
    Tolerance defaults to 0.005 * (max - min). If I don't change min & max, then they default to 0 and π,
    so tol= 0.005 * π. As I raise tol, the failure zones shrink. With 0.008 * π, everything works for my
    desired parameters (k = 5, theta0 = anything)
    
    Conclusion: I can reduce k to 1.9 instead of 5, and avoid the bug. I tried it, the sim works pretty
    well, but not as pretty as I'd like. The leading edge is wavier. I would like to stick with k = 5.
    
    Increase tolerance to 0.008 * π, seems *maybe* less wavy, but also more jittery at the edge.
    
    But these effects also seem to depend on how many holes are present, which varies. (Maybe varies *because*
    of these parameters, but hard to say for sure.) So, maybe want to put off a final decision until after the
    holes are fixed, then see what I prefer. Might want to end up splitting the difference: reduce k at least a
    little, if it improves the jitter, but not so much that it gets more wavy. Select k first, then reevaluate
    to see how much the tolerance needs to be raised.
    """
    for degs in range(180, -1, -1):
        k = 5
        pot: tf.Potential = tf.Potential.harmonic_angle(k=k, theta0=math.radians(degs), tol=0.008 * math.pi)
        print(f"{degs}: {pot}")
        
    # With k=5, tol=default, fails with theta0= 150°, but succeeds at 165°. At tol=0.008, both theta0 succeed.
    test_pot_005_165: tf.Potential = tf.Potential.harmonic_angle(k=5, theta0=math.radians(165), tol=0.005 * math.pi)
    test_pot_008_165: tf.Potential = tf.Potential.harmonic_angle(k=5, theta0=math.radians(165), tol=0.008 * math.pi)
    test_pot_008_150: tf.Potential = tf.Potential.harmonic_angle(k=5, theta0=math.radians(150), tol=0.008 * math.pi)
    test_pot_005_165.plot(force=False, potential=True, max=math.pi)
    test_pot_008_165.plot(force=False, potential=True, max=math.pi)
    test_pot_008_150.plot(force=False, potential=True, max=math.pi)
    
    test_pot_lower_k_150: tf.Potential = tf.Potential.harmonic_angle(k=1.9, theta0=math.radians(150))
    test_pot_tiny_k_bigger_min: tf.Potential = tf.Potential.harmonic_angle(k=0.5, theta0=math.radians(150),
                                                                           min=math.pi/4)
    test_pot_lower_k_150.plot(force=False, potential=True, max=math.pi)
    test_pot_tiny_k_bigger_min.plot(force=False, potential=True, max=math.pi)

# _test_harmonic_angle()
