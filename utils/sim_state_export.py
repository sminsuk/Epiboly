"""Save and retrieve state of the simulation

Protection against crash: retrieve and continue. Also for post-processing
"""
import os

import tissue_forge as tf
import config as cfg
import utils.tf_utils as tfu

_state_export_path: str
_state_export_interval: int = 0
_previous_export_timestep: int = 0
_current_export_timestep: int = 0

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
    
    _state_export_path = os.path.join(tfu.export_path(), "Sim_state")
    os.makedirs(_state_export_path)

def export_enabled() -> bool:
    """Convenience function. Interpret _state_export_interval as flag for whether export is enabled"""
    return _state_export_interval != 0

def _export_state(filename: str) -> None:
    path: str = os.path.join(_state_export_path, filename)
    print(f"Saving complete simulation state to '{path}'")
    tf.io.toFile(path)

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
    
    _export_state(filename + ".json")

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
