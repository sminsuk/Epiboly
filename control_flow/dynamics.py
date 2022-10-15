"""Set up dynamical changes to the system.

Creates events that are run every simulation step. By default, they do nothing,
until execute_repeatedly() is called, creating a task list.

The on_time event is for actions that involve iterating over multiple particles at once.
The on_particle events are for actions that involve a single particle.
on_particle events only take a *single* ParticleType. So if you want to run an event on
more than one type, you have to turn on that event for each one.
"""
from collections.abc import Callable
from typing import TypedDict

from epiboly_init import *
import sharon_utils as su

class Task(TypedDict, total=False):
    invoke: Callable[..., None]     # required
    args: dict                      # not required, if invoke function takes no args

def execute_repeatedly(tasks: list[Task] = None) -> None:
    """Execute these tasks, in order, once per timestep.
    
    Each time this is called, the old task list is replaced by the new one.
    To clear the task list so nothing is happening, omit the arg or pass [].
    """
    global _tasks
    
    if tasks is None:
        tasks = []
    
    # Validate each task in every possible way that this could go wrong
    # (that I can think of, and am able to test).
    # Filter out any baddies and keep the rest.
    _tasks = [task for task in tasks
              if (task
                  and "invoke" in task
                  and task["invoke"]
                  )
              ]
    
    for task in _tasks:
        # and although args not required, make sure there's an entry for it
        if "args" not in task:
            task["args"] = {}
            
        invoke: Callable[..., None] = task["invoke"]
        args: dict = task["args"]
        print(f"setting up invoke: {invoke.__name__} with args: {args}")

######## Private ########
_tasks: list[Task] = []

def _master_event(evt):
    """This is intended to be run every simulation step. For now, ignoring evt and letting it run forever."""
    global _tasks
    
    for task in _tasks:
        invoke: Callable[..., None] = task["invoke"]
        args: dict = task["args"]
        try:
            invoke(**args)
        except Exception as e:
            su.exception_handler(e, invoke.__name__)

# def _master_particle_event(evt):
#     """This runs every simulation step. Ignore evt and let it run forever, but pass the particle to the invoked actions.
#
#     Use for actions that act on a single particle.
#     evt contains the particle information.
#     For now, hard-coding that this acts on the two types, Little and LeadingEdge. I'll probably generalize
#     this, and set up a way to launch events for any generic set of ParticleTypes.
#     Also for now, using same master event for all types. If their behaviors get very divergent, might end up
#     being cleaner to have one master_particle_event for each type. Alternatively, different actions for each type.
#     """
#     global _bond_maintenance
#     if _bond_maintenance[evt.targetType.name]:
#         try:
#             _maintain_bonds(evt.targetParticle)
#         except Exception as e:
#             su.exception_handler(e, _master_particle_event.__name__)


# This will happen as soon as it's imported (after the tf.init)
tf.event.on_time(period=tf.Universe.dt, invoke_method=_master_event)

# These will happen as soon as they're imported (after the tf.init)
# tf.event.on_particletime(period=10 * tf.Universe.dt, invoke_method=_master_particle_event, ptype=Little)
# tf.event.on_particletime(period=10 * tf.Universe.dt, invoke_method=_master_particle_event, ptype=LeadingEdge)
