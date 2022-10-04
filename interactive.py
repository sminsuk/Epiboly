"""Functions available to be called while script is running, from jupyter cells, ipython cmd line, etc."""

from epiboly_init import *

# Since right now I'm running from Jupyter and not from ipython, set these to False.
# They prevent these functions from executing when you Kernel > Restart & Run All in Jupyter, without having
# to have the calls commented out, which is a pain.
# Once I transition to running from ipython, can set them to True so that the functions will work the first time.
# And for now, when running from regular python, to run one of these at the start of the script, execute it twice in main.
vis_toggle_allowed = False
rad_toggle_allowed = False
downward_force_allowed = False

def toggle_visibility():
    """Depends on the fact that all Little particles were assigned a Style instance when they were created."""
    global vis_toggle_allowed
    if vis_toggle_allowed:
        for p in Little.items():
            p.style.visible = not p.style.visible
    else:
        print("First invoke, outside Jupyter call twice")
        vis_toggle_allowed = True
        
# Because I constantly make this mistake in typing, and then wonder why it doesn't work!
toggle_invisibility = toggle_visibility
        
def toggle_radius():
    """Careful! Best to run this only while paused, and then toggle it back again before
    unpausing, or might affect the simulation"""
    global rad_toggle_allowed
    if rad_toggle_allowed:
        threshold = Little.radius / 2
        tiny_radius = Little.radius / 5
        for p in Little.items():
            if p.radius > threshold:
                p.radius = tiny_radius
            else:
                p.radius = Little.radius
    else:
        print("First invoke, outside Jupyter call twice")
        rad_toggle_allowed = True
        
# Now that have the dynamics module, don't need this. CustomForce isn't very useful;
# even with a more complicated function, it cannot vary by particle. Marked for deletion!
def add_downward_force():
    global downward_force_allowed
    if downward_force_allowed:
        # The following force creation and assignment causes the kernel to die.
        # Apparently python destroys it when it goes out of scope, unware that it is bound
        # to a ParticleType at the C++ level.
        #
        # Setting thisown = 0 fixes it.
        #
        # Alternatively, create the variable at the global level, so that the python object is persistent,
        # and that fixes it as well. (In case there's ever any issue with thisown.)
        downward_force = tf.CustomForce([0, 0, -5], tf.Universe.dt)
        downward_force.thisown = 0 # Don't destroy after going out of scope
        tf.bind.force(downward_force, LeadingEdge)
        print("Added downward force.")
    else:
        print("First invoke, outside Jupyter call twice")
        downward_force_allowed = True
        
def count_bonds():
    """Tally number of bonds per particle. Should never be > 6 if using that criterion
    
    This one is harmless to run since it doesn't affect the simulation or even the rendering,
    so no need to ever guard against that like the others.
    """

    def add_to_histogram(p):
        neighbors = p.getBondedNeighbors()
        key = len(neighbors)
        if key not in histogram:
            histogram[key] = 0
        histogram[key] = histogram[key] + 1    

    histogram = {}
    for p in Little.items():
        add_to_histogram(p)
    for p in LeadingEdge.items():
        add_to_histogram(p)
    print(histogram)
        
# workaround for ipython console issues!
# Remember to use "irun" rather than "tf.irun"!
# Ugh, this doesn't work either!
# irun = tf._SimulatorPy.irun

