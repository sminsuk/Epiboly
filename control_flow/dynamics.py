"""Set up dynamical changes to the system.

Creates events that are run every simulation step. By default, they do nothing,
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

_tasks: list[Task] = []

def _master_event(evt: tf.event.TimeEvent) -> int:
    """This is intended to be run every simulation step. Runs repeatedly forever unless error."""
    global _tasks
    
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
            # TF docs say to do this on error, unclear if it has any effect
            return 1
    
    return 0

# This will happen as soon as it's imported (after the tf.init)
tf.event.on_time(period=tf.Universe.dt, invoke_method=_master_event)
