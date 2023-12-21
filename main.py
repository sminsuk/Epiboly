"""Epiboly simulation"""
from datetime import datetime, timezone
import sys

import tissue_forge as tf
import epiboly_init
import config as cfg

import biology.bond_maintenance as bonds
import biology.cell_division as cd
import biology.microtubules as mt
import control_flow.events as events
from control_flow.interactive import toggle_visibility
import setup_and_equilibrate as setup
import utils.epiboly_utils as epu
import utils.plotting as plot
import utils.sim_state_export as state
import utils.tf_utils as tfu
import utils.video_export as vx

# from control_flow.interactive import is_interactive, toggle_visibility
# if is_interactive():
if tf.system.is_terminal_interactive() or tf.system.is_jupyter_notebook():
    # Importing this at the global level causes PyCharm to keep deleting my epiboly_init import,
    # (ToDo: see if this is still true, now that I got rid of "import *".)
    # I guess because it thinks it's redundant; something about the "import *".
    # Doing it here prevents that. And this is only needed in interactive sessions.
    # (Edit: um, none of that is relevant anymore, after refactoring; no more vars from epiboly_init here.
    # But keep this comment for history until the whole interactivity mess gets resolved.)
    # (ToDo: also, turns out is_interactive() functionality is now provided in tf, I can probably lose mine.)
    # (However, tf functionality is different: is_terminal_interactive() does not catch Jupyter, you have
    # to test for that separately.)
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

local: datetime = datetime.now(timezone.utc).astimezone()
print("Start:", local.strftime("%Y-%m-%d %I:%M:%S %p %Z"))
print(f"tissue-forge version = {tf.version.version}")
print(f"System: {tf.version.system_name} {tf.version.system_version}")
print(f"CUDA installed: {'Yes' if tf.has_cuda else 'No'}")

if cfg.initialization_directory_name:
    epiboly_init.init_from_import()
else:
    epiboly_init.init()

events.initialize_master_event()

# Uncomment to render for publication printing: 3d bonds (note: not performance-friendly), and no scene decorations
# ToDo: customize radius of rendered bonds. Needs patch release to expose feature to python
# tf.system.set_rendering_3d_bonds(True)
# tf.system.decorate_scene(False)

# Setup and equilibration – unless importing saved state from a prior run
if not cfg.initialization_directory_name:
    vx.init_camera_data()
    
    setup.initialize_embryo()
    cd.initialize_division_rate_tracking()
    mt.initialize_tangent_forces()
    
    # Call now so that state is exported after setup/equilibration but before any update events;
    # then again at the end so I get the final state if the script completes.
    state.export("Timestep true zero")

# toggle_visibility()
# toggle_visibility()
events.execute_repeatedly(tasks=[
        {"invoke": vx.save_screenshot_repeatedly},
        {"invoke": plot.show_graphs},
        {"invoke": mt.apply_even_tangent_forces},
        {"invoke": bonds.maintain_bonds},
        {"invoke": cd.cell_division},
        {"invoke": state.export_repeatedly},
        ])

vx.set_screenshot_export_interval()
if cfg.windowed_mode:
    tf.show()
    events.execute_repeatedly(tasks=[
            {"invoke": vx.save_screenshot_repeatedly},
            {"invoke": plot.show_graphs},
            {"invoke": mt.apply_even_tangent_forces},
            {"invoke": bonds.maintain_bonds,
             "args": {
                      #     ###For making/breaking algorithm:
                      # "k_neighbor_count": 1.0,
                      # "k_angle": 1.0,
                      # "k_edge_neighbor_count": 1.0,
                      # "k_edge_angle": 1.0
                      }
             },
            {"invoke": cd.cell_division},
            {"invoke": state.export_repeatedly},
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
    
    vx.save_screenshots("Final")
else:
    def sim_finished() -> bool:
        # Choose one:
        
        # For truncated run, uncomment this return statement and set desired duration:
        # setup.equilibration_time accounts for equilibration; many time units (hundreds) but fast executing (minutes)
        # plus a few more for the sim proper if desired (slow; ~30 per hour on my old Intel Mac)
        # return tf.Universe.time > setup.equilibration_time + 5
    
        if cfg.run_balanced_force_control:
            # No epiboly should occur, so can't use epiboly progress to decide when to stop.
            # Go same total time that epiboly would take (approximate).
            # Rather than trying to match the duration of the current config (which has a lot of permutations
            # after recent addition of new options), match the duration of the LONGEST-running
            # version of epiboly, which is the version with no cell division at all.
            return tf.Universe.time > setup.equilibration_time + 800
        else:
            # Full epiboly:
            return epu.leading_edge_mean_phi() > cfg.stopping_condition_phi
    
    def run_sim() -> None:
        while True:
            if sim_finished():
                break
            if events.event_exception_was_thrown():
                # Could sys.exit() here (it works here, just not inside a TF event invoke method), but even better,
                # do any final graphing and movie making and THEN exit. The main thing is to get out of the loop.
                break
            tf.step()

    def recoil_test(remodel_bonds: bool, duration: float) -> None:
        """Let the system equilibrate (no external force), with or without bond remodeling
        
        Without bond remodeling: reveals the instantaneous or short-time-scale elastic behavior.
        With bond remodeling: reveals the ability for further viscoelastic deformation (reversal) under tension
        
        Note that running with bond remodeling may not be so interesting when cell division is enabled,
        because all the space has been filled by additional particles, limiting how much shrinkage can occur.
        
        duration: should be in units of Universe.time
        """
        # ToDo?
        # Might want to do something with plotting, here, to capture plots at the moment we switch mode.
        # It's a bit tricky, though, because of accumulating data for time averaging, so I haven't tried to implement.
        # (And the simplest plots - one dot per time point like leading edge phi, and forces - don't really need
        # a plot here anyway. It's the others, with separate plots vs. phi for each time point, that we'd like
        # to capture. Of those, Aggregate Tension plot doesn't have time averaging, and it might be nice to capture
        # just that one.)
        
        # Turn off the external force
        mt.remove_tangent_forces()
        
        # Set up new task list. Include same observational monitoring as in the main sim, but omit external
        # force application, and cell division. Include/exclude bond breaking/making as indicated.
        tasks: list[events.Task] = [
                {"invoke": vx.save_screenshot_repeatedly},
                {"invoke": plot.show_graphs},
                ]
        if remodel_bonds:
            tasks.append({"invoke": bonds.maintain_bonds})
        tasks.append({"invoke": state.export_repeatedly})
        events.execute_repeatedly(tasks=tasks)
    
        recoil_start_time: float = tf.Universe.time
        while True:
            if tf.Universe.time > recoil_start_time + duration:
                break
            if events.event_exception_was_thrown():
                break
            tf.step()

    run_sim()
    
    # Two recoil tests; depending on config, can run them in isolation or sequentially.
    # Note that after this point, can't really restart sim after any crash or manual abort, unless I do some
    # more work, because this gives state to the formerly stateless main module, which I'd need to capture.
    # (With one exception, a commonly used workflow: it should work fine to abort in order to intentionally end the
    # simulation prematurely, restarting with a truncated sim_finished() criterion only in order to do the cleanup —
    # generate movie, get final screenshots. Just turn off the two recoil flags before restarting.)
    if cfg.test_recoil_without_bond_remodeling:
        recoil_test(remodel_bonds=False, duration=cfg.recoil_duration_without_remodeling)
    if cfg.test_recoil_with_bond_remodeling and not cfg.cell_division_enabled:
        # This one really only meaningful in the absence of cell division, so don't bother when cell division enabled.
        recoil_test(remodel_bonds=True, duration=cfg.recoil_duration_with_remodeling)

# troubleshooting: call this one final time without having to be triggered by an edge transformation event.
# bonds.test_ring_is_fubar()

# Do final plot before final state export so that the final graphed data point is included in the export
plot.show_graphs(end=True)

if events.event_exception_was_thrown():
    # We're not really finished, so we want to be able to recover.
    # (It's important not to delete what has already been exported, so we have to catch this case.
    # It's not super important to export the last bit, but we might as well, because we can.)
    
    # On second thought, maybe not. I'm getting lots of asserts with v. 0.1.0 that never happened before.
    # Looks like Angles are getting corrupted. Don't want to reimport those broken configurations, so
    # probably best to rewind back to a previously saved state. So, neither delete what was previously
    # exported, nor export anything new:
    pass
    # state.export("Exception")
else:
    state.export("Final")
    state.remove_unneeded_state_exports("state", keep_final=False)
    state.remove_unneeded_state_exports("extra", keep_final=True)

vx.make_movie()

# Only after making the movie, so that these stills won't be included
vx.final_result_screenshots()
