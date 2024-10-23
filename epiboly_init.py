"""Create basic types and initialize the simulation"""
import os

import tissue_forge as tf
import epiboly_globals as g
import config as cfg
import utils.epiboly_utils as epu
import utils.global_catalogs as gc
import utils.tf_logging as logging
import utils.sim_state_export as state
import utils.tf_utils as tfu
import utils.video_export as vx

class LittleType(tf.ParticleTypeSpec):
    mass = 1.0
    radius = cfg.evl_particle_radius
    dynamics = tf.Overdamped

class BigType(tf.ParticleTypeSpec):
    mass = 70.0
    radius = 3.0
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
    tfu.init_export(sim_directory_name=cfg.initialization_directory_name)
    
    # These two inits only depend on the exported directory being found, by tfu.init_export(), so can do them up here.
    vx.init_screenshots()
    state.init_export()
    
    latest_state_entry: os.DirEntry
    latest_extra_state_entry: os.DirEntry
    latest_state_entry, latest_extra_state_entry = state.find_exported_state_files()
    
    tf.init(load_file=latest_state_entry.path,
            dim=_dim,
            dt=cfg.dt,
            # cutoff=2,
            windowless=not cfg.windowed_mode,
            window_size=_window_size,
            throw_exc=True)
    
    g.Little = tf.ParticleType_FindFromName("LittleType")
    g.Big = tf.ParticleType_FindFromName("BigType")
    g.LeadingEdge = tf.ParticleType_FindFromName("LeadingEdgeType")
    
    state.import_additional_state(latest_extra_state_entry.path)
    gc.initialize_state()
    
    # This init depends on the extra state import having already happened, so
    # can't be done until down here, after state.import_additional_state()
    logging.init_logging()
    
    screenshots_path: str = os.path.join(tfu.export_path(), vx.screenshots_subdirectory())
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
            dt=cfg.dt,
            # cutoff=2,
            windowless=not cfg.windowed_mode,
            window_size=_window_size,
            throw_exc=True)
    
    g.Little = LittleType.get()
    g.Big = BigType.get()
    g.LeadingEdge = LeadingEdgeType.get()
    
    g.Little.style.color = epu.evl_undivided_color
    g.LeadingEdge.style.color = epu.evl_margin_undivided_color
