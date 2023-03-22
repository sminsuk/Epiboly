"""Create basic types and initialize the simulation"""
import os

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import utils.global_catalogs as gc
import utils.logging as logging
import utils.sim_state_export as state
import utils.tf_utils as tfu
import utils.video_export as vx

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
    print(f"Restarting simulation \"{cfg.initialization_directory_name}\" from latest state export...")
    tfu.init_export(directory_name=cfg.initialization_directory_name)
    
    # These two inits only depend on the exported directory being found, by tfu.init_export(), so can do them up here.
    vx.init_screenshots()
    state.init_export()
    
    saved_state_path: str = os.path.join(tfu.export_path(), state.sim_state_subdirectory())
    screenshots_path: str = os.path.join(tfu.export_path(), vx.screenshots_subdirectory())
    
    # Find the latest saved state: two files
    state_entries: list[os.DirEntry] = []
    extra_state_entries: list[os.DirEntry] = []
    state_entry: os.DirEntry
    with os.scandir(saved_state_path) as state_entries_it:
        for state_entry in state_entries_it:
            if state_entry.name.endswith("_state.json"):
                state_entries.append(state_entry)
            elif state_entry.name.endswith("_extra.json"):
                extra_state_entries.append(state_entry)

    latest_state_entry: os.DirEntry = max(state_entries, key=lambda entry: entry.stat().st_mtime_ns)
    latest_extra_state_entry: os.DirEntry = max(extra_state_entries, key=lambda entry: entry.stat().st_mtime_ns)
    
    tf.init(load_file=latest_state_entry.path,
            dim=_dim,
            # cutoff=2,
            windowless=not cfg.windowed_mode,
            window_size=_window_size)
    
    g.Little = tf.ParticleType_FindFromName("LittleType")
    g.Big = tf.ParticleType_FindFromName("BigType")
    g.LeadingEdge = tf.ParticleType_FindFromName("LeadingEdgeType")
    
    gc.initialize_state()
    state.import_additional_state(latest_extra_state_entry.path)
    
    # This init depends on the extra state import having already happened, so
    # can't be done until down here, after state.import_additional_state()
    logging.init_logging()
    
    if vx.screenshot_export_enabled():
        # Delete screenshots that were created *after* that last state was saved, since we'll be regenerating
        # them and we don't want the old ones to end up in the movie.
        unneeded_image_paths: list[str]
        image_entry: os.DirEntry
        with os.scandir(screenshots_path) as image_entries_it:
            unneeded_image_paths = [image_entry.path for image_entry in image_entries_it
                                    if image_entry.stat().st_mtime_ns > latest_state_entry.stat().st_mtime_ns]
        num_images = len(unneeded_image_paths)
        if num_images > 0:
            print(f"Removing {num_images} unneeded images")
            for path in unneeded_image_paths:
                os.remove(path)

def init() -> None:
    tfu.init_export()
    vx.init_screenshots()
    state.init_export()
    logging.init_logging()
    
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
