"""Some basic types I want to have globally available in all modules

Everything that *should* be in the global namespace, should be here, and that should not be much.
Everything else should be local to a function, or in a module.
"""
import tissue_forge as tf
from utils import tf_utils as tfu

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

windowless: bool = True
_window_size: list[int] = [800, 600]    # [800, 600] is default; [1200, 900] is nice and big for presentations
_dim = [10., 10., 10.]
# Cutoff = largest potential.max in the sim, so that all necessary potentials will be evaluated:
tf.init(dim=_dim, windowless=windowless, window_size=_window_size)  # , cutoff = 2)

Little: tf.ParticleType = LittleType.get()
Big: tf.ParticleType = BigType.get()
LeadingEdge: tf.ParticleType = LeadingEdgeType.get()

Little.style.color = tfu.cornflower_blue
LeadingEdge.style.color = tfu.gold
