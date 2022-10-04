from collections.abc import Callable
from typing import Optional, TypedDict

from epiboly_init import *
import sharon_utils as su


class Task(TypedDict, total=False):
    invoke: Callable[[], None]
    wait: Callable[[], bool]
    verbose: bool


def execute_sequentially(queue: list[Task] = None):
    """Sequentially invoke a list of functions each paired with a wait condition
    
    queue: a list of tasks, each represented by a dictionary. Each task has a main function to be
        invoked (under the dictionary key "invoke"), and a second function representing a wait
        condition (under the dictionary key "wait"). For each task: 
            "invoke": function to call, that requires no arguments; return values will be ignored.
            "wait": function that takes no arguments and returns boolean; when it returns true,
                control will proceed to the next task.
            "verbose": boolean. If True, "." is printed each cycle as long as wait condition
                continues to return False. (For the moment, cycle period is fixed equal to
                Universe.dt.) Only meaningful for tasks that have a wait condition.
        
    "invoke" and "wait" are callbacks, hence are just function *names*. Where these function calls
    need arguments passed, lambdas can be provided, in order to do that. But progress reporting
    and exception handling is more readable and informative if the functions have names. So,
    preferably, write a short named function instead of a lambda.
    
    Within each task, each key is optional and can be given the value None, or simply omitted. Sequential
    execution continues until the last task has completed. A task is complete when the "invoke"
    function (if any) has executed and its associated "wait" function (if any) has returned True.

    Since this functionality is entirely dependent on simulation time, no "invoke" function is called until
    "Run" is clicked on the simulator. To make the first item in the queue execute immediately without
    waiting for "Run", invoke it separately, before invoking execute_sequentially(). Then enter
    "'invoke': None" in the first task dictionary in the queue (or simply omit "invoke"), and provide a "wait"
    function. The cycle will start with the first "wait". (Again, a named do-nothing function as a
    placeholder in that first item makes for more informative text feedback.)
    """
    if queue is None:
        queue = []

    # Probably doesn't matter, since list is likely to be short, but for efficient popping, reverse 
    # the list and pop from the end. Alternatively, could import deque and popleft()
    queue.reverse()

    invoke_func: Optional[Callable[[], None]] = None
    wait_condition: Optional[Callable[[], bool]] = None
    verbose: bool = True
    ready_for_next_invoke: bool = True

    def condition_tester(condition: Callable[[], bool]) -> bool:
        """Test the condition, wrapped in a try/except"""

        result = True
        try:
            result = condition()
        except Exception as e:
            su.exception_handler(e, condition.__name__)
        finally:
            # If exception occurs in the condition, returns True to prevent it from
            # being called (and reported) over and over. This means execution will proceed
            # to the next action in the queue.
            return result

    def get_next():
        nonlocal invoke_func, wait_condition, verbose

        current_task = None if not queue else queue.pop() or {}
        invoke_func = (
            None if not current_task
            else (
                (lambda: None) if "invoke" not in current_task
                else current_task["invoke"] or (lambda: None)
            )
        )
        wait_condition = (
            None if not current_task
            else (
                (lambda: True) if "wait" not in current_task
                else current_task["wait"] or (lambda: True)
            )
        )
        verbose = (
            None if not current_task
            else (
                True if "verbose" not in current_task
                else current_task["verbose"]
            )
        )

    def event_cycle(event):
        """Invoke each function then wait until its condition returns True; repeat until no functions left."""

        nonlocal ready_for_next_invoke

        if ready_for_next_invoke:
            ready_for_next_invoke = False
            get_next()
            if invoke_func:
                print(f"\n---Calling {invoke_func.__name__}()")
                try:
                    invoke_func()
                except Exception as e:
                    su.exception_handler(e, invoke_func.__name__)
            else:
                # All out of functions to invoke, so we're done.
                event.remove()
        elif condition_tester(wait_condition):
            ready_for_next_invoke = True
        elif verbose:
            print(".", end="", flush=True)

    tf.event.on_time(period=tf.Universe.dt, invoke_method=event_cycle)


def execute_and_wait_until(invoke_func=None,
                           wait_condition=lambda: True,
                           verbose=True):
    """Invoke the provided function, then wait until condition is true.
    
    invoke_func: function to call. (Optional. if omitted, this is just a wait, equivalent to wait_until().)
    wait_condition: function that returns boolean.
    verbose: boolean. If True, "." is printed each cycle while waiting.
        (For the moment, cycle period is fixed equal to Universe.dt.)
    """
    execute_sequentially(queue=[{"invoke": invoke_func, "wait": wait_condition, "verbose": verbose}])


def wait_until(wait_condition=lambda: True,
               verbose=True):
    """Wait until condition is true, and signal the user.
    
    wait_condition: function that returns boolean.
    verbose: boolean. If True, "." is printed each cycle while waiting.
        (For the moment, cycle period is fixed equal to Universe.dt.)
    
    This is equivalent to execute_and_wait_until() with the invoke_func omitted.
    """
    execute_and_wait_until(wait_condition=wait_condition,
                           verbose=verbose)
