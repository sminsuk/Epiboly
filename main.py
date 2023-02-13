"""Epiboly simulation"""
import os.path
import sys

import tissue_forge as tf
import config as cfg

from biology import bond_maintenance as bonds,\
    microtubules as mt
from control_flow import dynamics as dyn
import setup_and_equilibrate as setup
from utils import epiboly_utils as epu,\
    tf_utils as tfu,\
    plotting as plot,\
    video_export as vx

from control_flow.interactive import is_interactive, toggle_visibility
if is_interactive():
    # Importing this at the global level causes PyCharm to keep deleting my epiboly_init import,
    # (ToDo: see if this is still true, now that I got rid of "import *".)
    # I guess because it thinks it's redundant; something about the "import *".
    # Doing it here prevents that. And this is only needed in interactive sessions.
    # (Edit: um, none of that is relevant anymore, after refactoring; no more epiboly_init here.
    # But keep this comment for history until the whole interactivity mess gets resolved.)
    # (ToDo: also, turns out is_interactive() functionality is now provided in tf, I can probably lose mine.)
    from control_flow.interactive import *
    print("Interactive module imported!")

# Force debugger usage. When I have breakpoints set to try and catch an intermittent bug,
# I tend to forget to run it in the debugger and just hit "run" instead, and don't notice
# until it's too late. Uncomment the following to make the script refuse to run except in
# the debugger:
# if sys.gettrace() is None:
#     # The debugger is not running
#     exit_msg: str = """    ###################################################################
#     ###                                                             ###
#     ###   Oops! Please start over and use the debugger this time!   ###
#     ###                                                             ###
#     ###################################################################
#     """
#     sys.exit(exit_msg)
    
print(f"tissue-forge version = {tf.version.version}")
print(f"System: {tf.version.system_name} {tf.version.system_version}")
print(f"CUDA installed: {'Yes' if tf.has_cuda else 'No'}")
tfu.init_export()
vx.init_screenshots()
logFilePath: str = os.path.join(tfu.export_path(), "Epiboly.log")
tf.Logger.enableFileLogging(fileName=logFilePath, level=tf.Logger.ERROR)

dyn.initialize_master_event()
epu.reset_camera()

# Choose one:
# setup.initialize_embryo()       # the old one
# setup.new_initialize_embryo()   # the new one under development
setup.unified_initialize_embryo()   # Start all bond activites all at once

# Call now so I at least get the graph for equilibration if I later abort execution;
# then again at the end so I get the whole thing if the script completes.
plot.save_graph(end=False)

# toggle_visibility()
# toggle_visibility()
dyn.execute_repeatedly(tasks=[
        {"invoke": vx.save_screenshot_repeatedly},
        {"invoke": plot.show_graph},
        {"invoke": mt.apply_even_tangent_forces},
        {"invoke": bonds.maintain_bonds},
        ])

vx.set_screenshot_export_interval()
if cfg.windowed_mode:
    tf.show()
    dyn.execute_repeatedly(tasks=[
            {"invoke": vx.save_screenshot_repeatedly},
            {"invoke": plot.show_graph},
            {"invoke": mt.apply_even_tangent_forces},
            {"invoke": bonds.maintain_bonds,
             "args": {
                      #     ###For making/breaking algorithm:
                      # "k_neighbor_count": 1.0,
                      # "k_angle": 1.0,
                      # "k_edge_neighbor_count": 1.0,
                      # "k_edge_angle": 1.0,
                      # "k_particle_diffusion": 20,
                      #     ###For relaxation:
                      # "relaxation_saturation_factor": 2,
                      # "viscosity": 0.001
                      }
             },
            ])
    # mt.remove_tangent_forces()
    
    # ### Note, this block can go, but am just keeping it for the comment about irun(), to remind me,
    # ### in case I ever get to where I actually have interactive execution available and working.
    # ### the is_equilibrated() function was already eliminated, in fact that whole module was.
    # tf.step()
    # while not xt.is_equilibrated(epsilon=0.01):
    #     # print("stepping!")
    #     tf.step()
    #     tf.Simulator.redraw()  # will this work once irun() is working? Do irun() before the loop?
    
    # tf.step(50)
    #
    # toggle_radius()
    # toggle_radius()
    tf.show()
    
    vx.save_screenshot("Final")
else:
    def sim_finished() -> bool:
        # Choose one:
        # Truncated run with provided duration:
        # return tf.Universe.time > 310 + 40  # 310 for equilibration (fast), + more (slow; ~25 per hour on old Mac)
        # Full epiboly:
        return epu.leading_edge_max_phi() > cfg.stopping_condition_phi
    
    while True:
        if sim_finished():
            break
        if dyn.event_exception_was_thrown():
            # Could sys.exit() here (it works here, just not inside a TF event invoke method), but even better,
            # do any final graphing and movie making and THEN exit. The main thing is to get out of the loop.
            break
        tf.step()

# troubleshooting: call this one final time without having to be triggered by an edge transformation event.
bonds.test_ring_is_fucked_up()

plot.save_graph(end=True)
vx.make_movie()

# Only after making the movie, so that these stills won't be included
vx.final_result_screenshots()
