"""Epiboly simulation"""

import tissue_forge as tf
import config as cfg

from biology import bond_maintenance as bonds,\
    microtubules as mt
from control_flow import dynamics as dyn, \
    exec_tests as xt
import setup_and_equilibrate as setup
from utils import epiboly_utils as epu,\
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

print(f"tissue-forge version = {tf.version.version}")
print(f"CUDA installed: {'Yes' if tf.has_cuda else 'No'}")

epu.reset_camera()
setup.initialize_embryo()

# toggle_visibility()
# toggle_visibility()
dyn.execute_repeatedly(tasks=[
        {"invoke": vx.save_screenshot_repeatedly},
        {"invoke": mt.apply_even_tangent_forces,
         "args": {"magnitude": 5}
         },
        {"invoke": bonds.maintain_bonds},
        ])

vx.set_screenshot_export_interval()
if cfg.windowed_mode:
    tf.show()
    dyn.execute_repeatedly(tasks=[
            {"invoke": vx.save_screenshot_repeatedly},
            {"invoke": mt.apply_even_tangent_forces,
             "args": {"magnitude": 5}
             },
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
        return epu.leading_edge_max_phi() > cfg.stopping_condition_phi
    
    # failsafe maximum; remember this is steps, not exported images.
    # (Number of exported images = steps / screenshot_export_interval)
    # A good value for quick smoke-test runs where you want the sim to run to completion quickly, is 50-100
    max_steps: int = 13000  # enough to capture a full epiboly, determined by trial and error
    
    for step in range(max_steps + 1):
        if sim_finished():
            break
        tf.step()

vx.make_movie()
