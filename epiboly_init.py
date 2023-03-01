"""Create basic types and initialize the simulation"""
import os

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import utils.global_catalogs as gc
import utils.sim_state_export as state
import utils.tf_utils as tfu

class LittleType(tf.ParticleTypeSpec):
    mass = 15
    radius = 0.08
    dynamics = tf.Overdamped

class BigType(tf.ParticleTypeSpec):
    mass = 1000
    radius = 3
    dynamics = tf.Overdamped

# Same as LittleType, but they will have different potentials and maybe other properties.
# As a subclass of Little, still gets its own color, and binding the superclass to a
# potential does NOT result in this getting bound.
class LeadingEdgeType(LittleType):
    pass

########
# ParticleTypeSpecs can be defined before Tissue Forge is initialized, but ParticleType instances
# can't be instantiated until afterward.

_window_size: list[int] = [800, 600]  # [800, 600] is default; [1200, 900] is nice and big for presentations
_dim = [10., 10., 10.]

def init_from_import() -> None:
    tfu.init_export(directory_name=cfg.initialization_directory_name)
    saved_state_path: str = os.path.join(tfu.export_path(), state.sim_state_subdirectory())
    
    # Find the latest saved state: two files
    state_entries: list[os.DirEntry] = []
    extra_state_entries: list[os.DirEntry] = []
    entry: os.DirEntry
    with os.scandir(saved_state_path) as dir_entries_it:
        for entry in dir_entries_it:
            if entry.name.endswith("_state.json"):
                state_entries.append(entry)
            elif entry.name.endswith("_extra.json"):
                extra_state_entries.append(entry)
    latest_state_path: str = max(state_entries, key=lambda entry: entry.stat().st_mtime_ns).path
    latest_extra_state_path: str = max(extra_state_entries, key=lambda entry: entry.stat().st_mtime_ns).path
    
    tf.init(load_file=latest_state_path,
            dim=_dim,
            # cutoff=2,
            windowless=not cfg.windowed_mode,
            window_size=_window_size)
    
    g.Little = tf.ParticleType_FindFromName("LittleType")
    g.Big = tf.ParticleType_FindFromName("BigType")
    g.LeadingEdge = tf.ParticleType_FindFromName("LeadingEdgeType")
    
    gc.initialize_state()
    state.import_additional_state(latest_extra_state_path)
    
def init() -> None:
    tfu.init_export()
    
    # Cutoff = largest potential.max in the sim, so that all necessary potentials will be evaluated:
    tf.init(dim=_dim,
            # cutoff=2,
            windowless=not cfg.windowed_mode,
            window_size=_window_size)
    
    g.Little = LittleType.get()
    g.Big = BigType.get()
    g.LeadingEdge = LeadingEdgeType.get()
    
    g.Little.style.color = tfu.cornflower_blue
    g.LeadingEdge.style.color = tfu.gold
