"""Some basic types I want to have globally available in all modules

Everything that *should* be in the global namespace, should be here, and that should not be much.
Everything else should be local to a function, or in a module.
"""
import os

import tissue_forge as tf
import config as cfg
import utils.global_catalogs as gc
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

# Before trying to use any of the following globally, call one of the init methods, which instantiate them.
_Little: tf.ParticleType
_Big: tf.ParticleType
_LeadingEdge: tf.ParticleType
latest_extra_state_path: str

# Really wanted those to be truly global, but couldn't maintain it with imported state because of module circularity.
# Things getting too complicated. Next best thing, import these functions to use globally:
def Big() -> tf.ParticleType:
    return _Big

def Little() -> tf.ParticleType:
    return _Little

def LeadingEdge() -> tf.ParticleType:
    return _LeadingEdge

########
# In order to have my ParticleType instances also be globally available, I have to initialize
# Tissue Forge before instantiating them, so that has to be done here as well.

_window_size: list[int] = [800, 600]  # [800, 600] is default; [1200, 900] is nice and big for presentations
_dim = [10., 10., 10.]

def init_from_import(sim_state_subdirectory: str) -> None:
    global _Big, _Little, _LeadingEdge, latest_extra_state_path

    tfu.init_export(directory_name=cfg.initialization_directory_name)
    saved_state_path: str = os.path.join(tfu.export_path(), sim_state_subdirectory)
    
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
    latest_extra_state_path = max(extra_state_entries, key=lambda entry: entry.stat().st_mtime_ns).path
    
    tf.init(load_file=latest_state_path,
            dim=_dim,
            # cutoff=2,
            windowless=not cfg.windowed_mode,
            window_size=_window_size)
    
    _Little = tf.ParticleType_FindFromName("_Little")
    _Big = tf.ParticleType_FindFromName("_Big")
    _LeadingEdge = tf.ParticleType_FindFromName("_LeadingEdge")
    
    gc.initialize_state()
    
def init() -> None:
    global _Big, _Little, _LeadingEdge
    tfu.init_export()
    
    # Cutoff = largest potential.max in the sim, so that all necessary potentials will be evaluated:
    tf.init(dim=_dim,
            # cutoff=2,
            windowless=not cfg.windowed_mode,
            window_size=_window_size)
    
    _Little = LittleType.get()
    _Big = BigType.get()
    _LeadingEdge = LeadingEdgeType.get()
    
    _Little.style.color = tfu.cornflower_blue
    _LeadingEdge.style.color = tfu.gold
