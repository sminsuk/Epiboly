"""Functions available to be called while script is running, from jupyter cells, ipython cmd line, etc."""
from IPython import get_ipython

from epiboly_init import Little, LeadingEdge

# Sometimes I still run from Jupyter and not from ipython, so set these to False.
# They prevent these functions from executing when you Kernel > Restart & Run All in Jupyter, without having
# to have the calls commented out, which is a pain.
# Once I transition to running from ipython, can set them to True so that the functions will work the first time.
# If I ever want to run these from regular python, will have to import them, then run them twice.
# (Or tweak how is_interactive() works.)
vis_toggle_allowed = False
rad_toggle_allowed = False

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
    resuming, or might affect the simulation"""
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
    
def is_interactive() -> bool:
    """See: https://stackoverflow.com/a/39662359 and https://stackoverflow.com/a/54967911 (same question,
    two relevant answers).
    """
    global vis_toggle_allowed, rad_toggle_allowed
    shell = get_ipython()
    if shell is None:
        return False
    
    # See StackOverflow
    is_jupyter_shell = (shell.__class__.__name__ == "ZMQInteractiveShell")
    
    # set these flags true in non-Jupyter shells (so I can use those functions the first time)
    # but false in Jupyter (so can run all cells without them doing anything that first time)
    print(f"Is {'' if is_jupyter_shell else 'not '}Jupyter")
    vis_toggle_allowed = not is_jupyter_shell
    rad_toggle_allowed = not is_jupyter_shell
    
    return True
    
# workaround for ipython console issues!
# Remember to use "irun" rather than "tf.irun"!
# Ugh, this doesn't work either!
# irun = tf._SimulatorPy.irun
# Nope, that either. TJ was able to reproduce. Waiting for fix.

