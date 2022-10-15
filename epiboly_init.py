"""Some basic types I want to have globally available in all modules

Everything that *should* be in the global namespace, should be here, and that should not be much.
Everything else should be local to a function, or in a module.
"""
from typing import TypedDict

import tissue_forge as tf

class LittleType(tf.ParticleTypeSpec):
    mass = 15
    radius = 0.15
    dynamics = tf.Overdamped

class BigType(tf.ParticleTypeSpec):
    mass = 1000
    radius = 3
    dynamics = tf.Overdamped

# Same as LittleType, but they will have different potentials and maybe other properties.
# As a subclass of Little, still gets its own color, and binding the superclass to a
# potential does NOT result in this getting bound.
class LeadingEdgeType(LittleType):
    pass

########
# In order to have my ParticleType instances also be globally available, I have to initialize 
# Tissue Forge before instantiating them, so that has to be done here as well.


dim = [10., 10., 10.]
# Cutoff = largest potential.max in the sim, so that all necessary potentials will be evaluated:
tf.init(dim=dim)  # , cutoff = 2)

Little: tf.ParticleType = LittleType.get()
Big: tf.ParticleType = BigType.get()
LeadingEdge: tf.ParticleType = LeadingEdgeType.get()

# Global particle and bond data. For storing stuff that is irretrievable once created
# Maybe put this in its own module, ore even a class?
# Usage: for now, leaving it up to the caller, since these things only happen in a couple of places:
#   Whenever creating a particle / bond, add it to particles_by_id / bonds_by_id, respectively
#   Whenever deleting a particle / bond, delete it from those dicts
#   BondData allows to retrieve the r0 from the potential attached to a given bond
#   ParticleData allows to retrieve ParticleHandle from an id. (This may not work. May have to
#       store actual Particles instead.)
class BondData(TypedDict):
    r0: float
    
class ParticleData(TypedDict):
    handle: tf.ParticleHandle
    
bonds_by_id: dict[int, BondData] = {}
particles_by_id: dict[int, ParticleData] = {}

# Just while determining empirically, the right number of interior particles to use.
# If there are too many, with frozen LeadingEdge, they'll pop past. If too few,
# they'll never fill the space no matter how long they equilibrate. Once the right
# number is determined, they should fill the space without popping through. Then
# Can comment this out and never use it again. (Or, perhaps will always use it during
# equilibration step, and release it afterward.)
# With this approach, instead of guessing how many particles to use and letting the
# leading edge get pushed downward by their expansion to an appropriate place, we'll
# set the location we *want* the leading edge, and get the right number of particles
# in there.
# LeadingEdge.frozen_z = True
epiboly_initial_percentage = 43     # real value for the misnomer "30% epiboly")

# See comments in function initialize_interior() in main for explanation
num_leading_edge_points = 60
num_spherical_positions = 2000  # 2050
edge_margin_interior_points = 0.15
