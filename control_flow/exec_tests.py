import tissue_forge as tf
from epiboly_init import LeadingEdge
import neighbors as nbrs
from utils import tf_utils as tfu

key_was_pressed = False


def keypress_event():
    """Use this as the 'wait' method, to wait for keypress"""
    return key_was_pressed


def setup_keypress_detection():
    """Use this as the 'invoke' method, to wait for keypress"""

    def respond_to_keypress(event):
        global key_was_pressed
        key_was_pressed = True
        print(tfu.bluecolor + "Key detected" + tfu.endcolor)
        event.remove()

    global key_was_pressed
    key_was_pressed = False
    print(tfu.redcolor + "*** Press a key to proceed ***" + tfu.endcolor)
    tf.event.on_keypress(invoke_method=respond_to_keypress)


def is_equilibrated(epsilon):
    """Simplistic method. True when energy below threshold.
    
    To do: More sophisticated method as per TJ: gather a trailing collection of these values,
    and used standard deviation with respect to the mean, instead of just this instantaneous
    value. In order to get rid of noise.
    """
    energy = tf.Universe.kinetic_energy
    result = energy < epsilon
    if result:
        print("Equilibrated.")
    return result

def leading_edge_is_equilibrated() -> bool:
    """Leading edge is equilibrated if every particle has at least 2 non-leading-edge neighbors"""
    
    for edge_particle in LeadingEdge.items():
        # This finds both leading-edge neighbors, which there will always be 2 of, and non-leading-edge
        # neighbors, which there can be any number of. So need at least 4 to count as equilibrated.
        # (But note, it is possible for this to fail, i.e. never reach this state. Need a timeout etc.)
        if len(nbrs.find_neighbors(edge_particle)) < 4:
            return False
        
    return True
