"""Save and retrieve state of the simulation

Protection against crash: retrieve and continue. Also for post-processing
"""
import json
import os

import tissue_forge as tf
import config as cfg

import biology.cell_division as cd
import utils.global_catalogs as gc
import utils.logging as logging
import utils.plotting as plot
import utils.tf_utils as tfu
import utils.video_export as vx

_state_export_path: str
_state_export_interval: int = 0
_previous_export_timestep: int = 0
_current_export_timestep: int = 0
_sim_state_subdirectory: str = "Sim_state"

def sim_state_subdirectory() -> str:
    """Return subdirectory to be used when saved state is reloaded"""
    return _sim_state_subdirectory
    
def init_export() -> None:
    """
    Set up subdirectory for all simulation state output

    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _state_export_path, _state_export_interval
    
    # Copy cfg property to module _protected; not caller-changeable at runtime. Ignore cfg henceforth and use this.
    _state_export_interval = cfg.sim_state_export_interval

    if not export_enabled():
        return
    
    _state_export_path = os.path.join(tfu.export_path(), _sim_state_subdirectory)
    os.makedirs(_state_export_path, exist_ok=True)

def export_enabled() -> bool:
    """Convenience function. Interpret _state_export_interval as flag for whether export is enabled"""
    return _state_export_interval != 0

def _export_additional_state(filename: str) -> None:
    """Export other info that this script maintains, not known to Tissue Forge
    
    WIP: May need to add additional state later, as needed.
    
    Modules that have even a little bit of state that would need to be explicitly saved in order to recover it:
    - plotting
    - video_export
    - sim_state_export (this one! the previous_ and current_ export timestep, if I don't want them to start over from 0)
    
    These and other modules also have state that can be reconstituted from scratch on reload
    """
    # Tell plot to save a graph. Not exactly needed for preserving results; export saves all the graph data, so after
    # reload, the whole graph will redraw. But this also results in the saved graph adding data points as you
    # go, so you don't have to wait until the sim finishes, to see ti.
    # (Note these have the same filename each time, so they're not accumulating, they're conveniently replacing
    # an existing graph with a newer better one with more data in it.)
    plot.save_graph()

    export_dict: dict = {"self": get_state(),
                         "video_export": vx.get_state(),
                         "cell_division": cd.get_state(),
                         "logging": logging.get_state(),
                         "plot": plot.get_state(),
                         }
    
    path: str = os.path.join(_state_export_path, filename)
    with open(path, mode="w") as fp:
        json.dump(export_dict, fp, indent=2)

def import_additional_state(import_path: str) -> None:
    import_dict: dict
    with open(import_path) as fp:
        import_dict = json.load(fp)
    
    set_state(import_dict["self"])
    plot.set_state(import_dict["plot"])
    logging.set_state(import_dict["logging"])
    cd.set_state(import_dict["cell_division"])
    vx.set_state(import_dict["video_export"])
    
def _export_state(filename: str) -> None:
    path: str = os.path.join(_state_export_path, filename)
    print(f"Saving complete simulation state to '{path}'")
    tf.io.toFile(path)
    
def remove_all_state_exports() -> None:
    """Exports are very large files and can be deleted if not needed"""
    entry: os.DirEntry
    with os.scandir(_state_export_path) as dir_entry_it:
        for entry in dir_entry_it:
            os.remove(entry.path)

def export(filename: str, show_timestep: bool = True) -> None:
    """
    Calling this method directly, is intended for one-off export operations *outside* of timestep events.
    Within repeated timestep events, use export_state_repeatedly(), which will generate unique filenames.

    Caller provides filename (no extension). Saves as json.
    Timestep will be appended to filename unless show_timestep = False (and filename is not blank).
    """
    if not export_enabled():
        return
    
    suffix: str = f"Timestep = {_current_export_timestep}"
    suffix += f"; Universe.time = {round(tf.Universe.time, 2)}"
    if not filename:
        filename = suffix
    elif show_timestep:
        filename += "; " + suffix
    
    # Before we export, make sure the state is clean. (Hopefully won't be needed after bugfix in future version.)
    gc.clean_state()

    # Remove all unneeded content of the directory before exporting more
    if not cfg.sim_state_export_keep:
        remove_all_state_exports()
    
    _export_state(filename + "_state.json")
    _export_additional_state(filename + "_extra.json")

def export_repeatedly() -> None:
    """For use inside timestep events. Keeps track of export interval, and names files accordingly."""
    global _previous_export_timestep, _current_export_timestep
    if not export_enabled():
        return
    
    # Note that this implementation means that the first time this function is ever called, the export
    # will always take place, and will be defined (and labeled) as Timestep 0. Even if the simulation has
    # been running before that, and Universe.time > 0.
    
    elapsed: int = _current_export_timestep - _previous_export_timestep
    if elapsed % _state_export_interval == 0:
        _previous_export_timestep = _current_export_timestep
        export("")  # just timestep as filename
    
    _current_export_timestep += 1

def get_state() -> dict:
    """We not only take care of exporting/importing extra (non-TF) state from other modules, but also from this one!

    We are keeping track of the timing of our exports, so this module is itself stateful, so that needs to
    be exported along with all the rest, in order to pick up the export timing where we left off.
    """
    return {"previous_step": _previous_export_timestep,
            "current_step": _current_export_timestep}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved.
    
    In this case, we increment _current because at the moment of export, it hadn't yet incremented (see
    export_repeatedly()), but now we've experienced an additional timestep.
    
    Compare vx, where the opposite is true. There, save_screenshot_repeatedly always completes, so at
    the moment of state export, its corresponding _current variable has already pre-incremented for the
    next timestep.
    """
    global _previous_export_timestep, _current_export_timestep
    _previous_export_timestep = d["previous_step"]
    _current_export_timestep = d["current_step"] + 1
