"""Set up tissue-forge events to handle dynamical changes to the system.

Creates a master invoke_method that is executed on each repeated event invocation. By default, it does nothing,
until execute_repeatedly() is called, creating a task list.
"""
from collections.abc import Callable
from typing import TypedDict

import tissue_forge as tf
from utils import tf_utils as tfu

class Task(TypedDict, total=False):
    invoke: Callable[..., None]     # required
    args: dict                      # not required, if invoke function takes no args

def execute_repeatedly(tasks: list[Task] = None) -> None:
    """Execute these tasks, in order, once per event invocation.
    
    Each time this is called, the old task list is replaced by the new one.
    To clear the task list so nothing is happening, omit the arg or pass an empty list.
    
    If called prior to calling initialize_master_event() (or providing your own event), no tasks
    are executed until master_event_method() gets invoked by an event.
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

_tasks: list[Task] = []

def master_event_method(evt: tf.event.Event) -> int:
    """This is intended to be used as the _invoke_method of a Tissue Forge timed event.
    
    By default, does nothing until it has been provided with a task list. Change its behavior on the fly
    and as often as desired, by providing a new task list, using the function execute_repeatedly().
    
    ToDo: This isn't quite general enough for sharing, yet. Needs to pass the evt object to the invoke methods
    of each task, so that they can .remove() when desired, and especially for ParticleTimeEvent so that they
    can extract Particle and ParticleType information.
    
    Note: evt object will be either TimeEvent or ParticleTimeEvent, depending on the type of event invocation that
    was set up. I type-hinted plain "Event" in order to represent that generality. Doesn't seem to matter;
    PyCharm doesn't complain, regardless of which type-hint I use or which type of event is sent, presumably
    because master_event_method() is never called directly from python, but only indirectly from TF itself.
    """
    for task in _tasks:
        invoke: Callable[..., None] = task["invoke"]
        args: dict = task["args"]
        try:
            invoke(**args)
        except Exception:
            # display the error and stack trace, which python fails to do
            tfu.exception_handler()
            # python won't exit, so at least cancel the event instead of calling a broken event repeatedly
            evt.remove()
            # That stops custom events, but it won't stop the script (and even calling sys.exit() won't work
            # from here), and it won't stop TF timestepping. To do that, set a signal so that the main script
            # (outside this invoke method) can exit for us. (Only works with tf.step(), not tf.show(); check
            # the signal in a loop between calls to tf.step(), and take appropriate action if detected.)
            global _exit_signal
            _exit_signal = True
            # TF docs say to do this on error, unclear if it has any effect
            return 1
    
    return 0

_exit_signal: bool = False
def event_exception_was_thrown() -> bool:
    return _exit_signal

def initialize_master_event() -> None:
    """Sets up to invoke events once per timestep.
    
    Convenience method to set up a master event. Alternatively, provide your own event, using either
    of the timed-event functions in Tissue Forge – on_time() or on_particletime() – with any desired
    customizing parameters. Use master_event_method as your invoke_method. master_event_method will
    be called repeatedly forever unless error.
    
    If called prior to calling execute_repeatedly(), events do nothing until they've been provided
    with a task list by calling that function.
    """
    # Note: dt * 0.9 in order to deal with a weirdness in the timer arithmetic. This
    # will actually result in invoke_method being called once each timestep. Using
    # exactly dt would result in the event being skipped about a third of the time!
    tf.event.on_time(period=tf.Universe.dt * 0.9, invoke_method=master_event_method)
