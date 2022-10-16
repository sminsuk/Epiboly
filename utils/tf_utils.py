"""Tissue Forge utilities

These are general purpose utility functions for Tissue Forge, not specific to any particular simulation.
"""
import math
import numpy as np
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

def random_nd_spherical(npoints, dim):
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

def exception_handler(exception, funcname):
    """General exception handler to be used inside tf events.

    Python won't catch them on its own! Call this in the "except:" clause

    exception: the exception object caught by try/except
    funcname: name of the function called (func.__name__), or description of code executed, in the "try:" clause.
    """
    stars = "**************************"
    print(redcolor + stars)
    print(f"Exception in {funcname}():")
    print(type(exception), exception)
    print(stars + endcolor)

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

def bond_distance(b: tf.BondHandle) -> float:
    """"Get the distance between the two particles of a bond, i.e. the "length" of the bond

    b: bondHandle of an *active* bond
    returns: float
    """
    p1, p2 = bond_parts(b)
    return p1.distance(p2)
