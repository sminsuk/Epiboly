"""video_export.py - Export simulation screenshots and compile them into movies

All export functions are no-ops unless export has been enabled in config.py

StackOverflow: https://stackoverflow.com/a/62434934
Documentation of MoviePy: https://zulko.github.io/moviepy/index.html
"""
from dataclasses import dataclass
import math
import os

from moviepy.video.io import ImageSequenceClip

import tissue_forge as tf
import config as cfg
import control_flow.events as events
import epiboly_globals as g
import utils.epiboly_utils as epu
import utils.tf_utils as tfu

@dataclass
class CameraData:
    center: tf.fVector3
    rotation: tf.fQuaternion
    zoom: float

def final_result_screenshots() -> None:
    """If enabled, capture still images from multiple angles.

    Note: This doesn't work in windowed mode; all 4 images come out the same. I.e., the camera updates
    don't result in a display update. Not worth trying to get around.
    """
    if not cfg.windowed_mode and screenshot_export_enabled():
        tf.system.camera_view_front()
        tf.system.camera_zoom_to(-12)
        _save_screenshot("Front", show_timestep=False)
        
        tf.system.camera_view_left()
        tf.system.camera_zoom_to(-12)
        _save_screenshot("Left", show_timestep=False)
        
        tf.system.camera_view_back()
        tf.system.camera_zoom_to(-12)
        _save_screenshot("Back", show_timestep=False)
        
        tf.system.camera_view_right()
        tf.system.camera_zoom_to(-12)
        _save_screenshot("Right", show_timestep=False)

        tf.system.camera_view_top()
        tf.system.camera_zoom_to(-12)
        _save_screenshot("Top", show_timestep=False)

        tf.system.camera_view_bottom()
        tf.system.camera_zoom_to(-12)
        _save_screenshot("Bottom", show_timestep=False)

_rotation_started: bool = False
_rotation_finished: bool = False

def _test_when_to_start_rotation() -> None:
    """At the appropriate time, trigger rotation by changing the value of _rotation_started from False to True"""
    global _rotation_started
    
    if cfg.paint_pattern == cfg.PaintPattern.ORIGINAL_TIER and cfg.paint_tier > 1:
        # suppress camera rotation for this lineage tracing, so the labeled cells don't rotate out of view
        return
    
    # Position of leading edge at which we start rotating the camera. π/2 is good for testing (the equator,
    # which is reached by the leading edge early in the simulation). For full epiboly, wait until close to veg pole.
    rotation_start_position: float = math.pi * 0.75
    
    if epu.leading_edge_max_phi() > rotation_start_position:
        _rotation_started = True

def _test_when_to_finish_rotation() -> None:
    """At the appropriate time, halt rotation by changing the value of _rotation_finished from False to True.
    
    Rotate from Front down to Bottom. Specific behavior of the quaternion (return value of tf.system.camera_rotation())
    is a bit weird; I selected angle() as the relevant property, and dropping below pi as the relevant behavior,
    based on empirical testing, but it may be very specific to this particular rotation. Seems like this may be brittle.
    (In fact, a small refactor – no algorithmic change – necessitated an adjustment in the tolerance, inexplicably.)
    (And, now that I've added the quadruple video, with 4 screenshots per export, one from each side, "Front" is quite
    intentionally done last, otherwise this wouldn't work. This is called after all four screenshots are done,
    and the criterion is specific to the rotation on that side.)
    """
    global _rotation_finished
    
    # Camera angle starts at 1.5π at Front position (pointing at equator), drops to π at Bottom (pointing at vegetal
    # pole), and then continues to drop lower if you rotate past Bottom and up the other side. So we want to stop
    # when angle < π. However, it does not hit exactly. At very bottom, value is slightly greater than π;
    # stopping at π thus overshoots by one rotation increment. So add a small tolerance to stop in the right place.
    # Furthermore, the timing of when modified camera data becomes readable is different for windowed mode
    # (maybe because I'm not moving the camera around as much, or maybe due to internal TF issues); either way,
    # in windowed mode, the angle we read is for the previous position so we need to stop rotating one
    # timestep earlier, so just make the tolerance bigger.
    target_camera_angle: float
    if cfg.paint_pattern == cfg.PaintPattern.PATCH:
        # Quick-and-dirty attempt to stay centered on the patch. Tracking positions of labeled cells would
        # be better, but is harder. So instead, assuming the patch is up against the leading edge
        # (cfg.patch_margin_gap == 0), just stop rotation earlier, half way between the initial position
        # (side view) and the vegetal pole. This is approximate, so don't worry about tolerance.
        target_camera_angle = 1.25 * math.pi
    else:
        tolerance: float = 0.04 if cfg.windowed_mode else 0.01
        target_camera_angle = math.pi + tolerance
    
    quat: tf.fQuaternion = tf.system.camera_rotation()
    # print(f"Target  = {target_camera_angle}\ncurrent = {quat.angle()}")
    if quat.angle() < target_camera_angle:
        # print("Setting rotation finished!")
        _rotation_finished = True

def _automated_camera_rotate() -> None:
    """Camera control during windowless export"""
    # This function is mainly intended for windowless export, but also automates while simulator is displayed.
    # To suppress automation in the windowed simulator, uncomment the following if statement:
    # if cfg.windowed_mode:
    #     return
    
    if _rotation_started and not _rotation_finished:
        # Note that "up" is "down". Behavior of "rotate up" is that the camera moves down toward vegetal.
        # The camera angle indeed "rotates up" around it's OWN axis while its position rotates DOWN relative to the
        # universe at the same time – continuing to point toward the center of the universe the whole time.
        # The effect is that the particles rotate upward in the field of view, i.e. the camera rotates downward.
        tf.system.camera_rotate_up()
    
def init_screenshots() -> None:
    """
    Set up subdirectory for all screenshot and movie output
    
    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _image_path, _movie_path
    if not screenshot_export_enabled():
        return
    
    base_path: str = tfu.export_path()
    _image_path = os.path.join(base_path, _screenshots_subdirectory)
    _movie_path = os.path.join(base_path, _movie_subdirectory)
    os.makedirs(_image_path, exist_ok=True)
    os.makedirs(_movie_path, exist_ok=True)
    
def _export_screenshot(filename: str) -> None:
    """Note, in system.screenshot(), bgcolor (and decorate) args only work in windowless mode."""
    path: str = os.path.join(_image_path, filename)
    result: int = tf.system.screenshot(path, bgcolor=[0, 0, 0])
    if result != 0:
        print(tfu.bluecolor + f"Something went wrong with screenshot export, result code = {result}" + tfu.endcolor)

def _save_screenshot(filename: str, side: str = None, show_timestep: bool = True) -> None:
    """side = "Front", "Back", "Left", "Right", "Top", "Bottom".
    
    Caller provides filename (no extension). Saves as jpg. All other formats crash the app, segfault!
    Timestep and side will be appended to filename unless show_timestep = False (and filename is not blank).
    """
    if not screenshot_export_enabled():
        return
    
    suffix: str = f"Timestep = {_current_screenshot_timestep}"
    suffix += f"; Universe.time = {round(tf.Universe.time, 2)}"
    suffix += f"; mean phi = {round(epu.leading_edge_mean_phi(), 2)}"
    suffix += f"; {len(g.Little.items()) + len(g.LeadingEdge.items())} cells"
    suffix += f"; {len(tf.BondHandle.items())} bonds"
    if side:
        suffix += f" - {side}"
        
    if not filename:
        filename = suffix
    elif show_timestep:
        filename += "; " + suffix
    
    _export_screenshot(filename + ".jpg")
    
def save_screenshots(filename: str) -> None:
    """
    Calling this method directly, is intended for one-off export operations *outside* of timestep events. Thus it
    only works in windowless mode. In windowed mode, always access this through save_screenshot_repeatedly().
    (At least, doesn't work *before* the sim window opens, though seems to work after it closes.)
    """
    if not _rotation_started:
        _test_when_to_start_rotation()
        
    precision: int = 4 if cfg.run_balanced_force_control else 2
    print(f"Saving screenshots for Timestep = {_current_screenshot_timestep};"
          f" Universe.time = {round(tf.Universe.time, 2)};"
          f" mean phi = {round(epu.leading_edge_mean_phi(), precision)}")

    if cfg.windowed_mode:
        _automated_camera_rotate()
        _save_screenshot(filename, "Front")
    else:
        # First, top view. Location never changes, so no load/save of data, no rotating
        tf.system.camera_view_top()
        tf.system.camera_zoom_to(-12)
        _save_screenshot(filename, "Top")
        
        # Then the four sides, keeping track of changing camera position
        for side in ["Left", "Right", "Back", "Front"]:
            _load_camera_data(side)
            _automated_camera_rotate()
            _save_screenshot(filename, side)
            _save_camera_data(side)
    
    if _rotation_started and not _rotation_finished:
        _test_when_to_finish_rotation()

def save_screenshot_repeatedly() -> None:
    """For use inside timestep events. Keeps track of export interval, and names files accordingly.
    
    Works best in windowless mode.
    Can also use in windowed mode but only while the simulator window is running, not in tf.step().
    """
    global _previous_screenshot_timestep, _current_screenshot_timestep
    if not screenshot_export_enabled():
        return
    
    # Note that this implementation means that the first time this function is ever called, the screenshot
    # will always be saved, and will be defined (and labeled) as Timestep 0. Even if the simulation has
    # been running before that, and Universe.time > 0.
    # But also note that that first image is really simulation timestep 1, EVEN IF the simulation is
    # starting fresh. Because by the time the event gets called in Timestep 0, the physics has
    # already been integrated, and particles have moved. If need a screenshot before that, can get it
    # (in windowless mode only) by calling a one-off save_screenshots() before calling tf.step().
    
    elapsed: int = _current_screenshot_timestep - _previous_screenshot_timestep
    if elapsed % _screenshot_export_interval == 0:
        _previous_screenshot_timestep = _current_screenshot_timestep
        save_screenshots("")
    
    _current_screenshot_timestep += 1

_screenshots_subdirectory: str = "Screenshots"
_movie_subdirectory: str = "Movies"
_image_path: str
_movie_path: str
_previous_screenshot_timestep: int = 0
_current_screenshot_timestep: int = 0

# Will only ever be modified if running this module as __main__ from another process; see bottom of file
_retain_screenshots_after_movie: bool = cfg.retain_screenshots_after_movie

# module's copy can be adjusted dynamically
_screenshot_export_interval: int = cfg.screenshots_timesteps_per_export

def screenshots_subdirectory() -> str:
    """Return subdirectory where screenshots are stored"""
    return _screenshots_subdirectory

def set_screenshot_export_interval(interval: int = None) -> None:
    """Set module current value of screenshot interval
    
    (Interval may need to change during the simulation.)
    Call with no arg to reset to the cfg value (e.g. after changing module value temporarily)
    
    Enabling/disabling screenshots happens at launch and can't be changed thereafter. Thus:
    if disabled (cfg interval == 0), then caller may not enable it.
    if enabled (cfg interval > 0), then caller may not change module current value to 0.
    
    But, remember that once enabled, export can still be turned on and off on the fly by changing whether
    "save_screenshot_repeatedly" is included in the task list for repeated events.
    """
    global _screenshot_export_interval
    if screenshot_export_enabled():
        if interval is None:
            interval = cfg.screenshots_timesteps_per_export
        if interval > 0:
            _screenshot_export_interval = interval
            print(f"Screenshot export interval set to {interval} timesteps")
        else:
            print(tfu.bluecolor + "Warning: screenshot export cannot be disabled after initialization" + tfu.endcolor)

def screenshot_export_enabled() -> bool:
    """Convenience function. Interpret cfg.screenshots_timesteps_per_export as flag for whether export is enabled"""
    return cfg.screenshots_timesteps_per_export != 0

# A dict of four CameraData objects, with keys "Front", "Back", "Left", "Right".
# (For windowless only. In windowed mode, this data cannot be captured.)
_camera_data: dict[str, CameraData] = {}

def init_camera_data() -> None:
    if cfg.windowed_mode:
        tf.system.camera_view_front()
        tf.system.camera_zoom_to(-12)
    else:
        # Need to render after each of these movements in order to save valid values
        tf.system.camera_view_left()
        tf.system.camera_zoom_to(-12)
        _export_and_delete_a_junk_screenshot()
        _save_camera_data("Left")
        
        tf.system.camera_view_right()
        tf.system.camera_zoom_to(-12)
        _export_and_delete_a_junk_screenshot()
        _save_camera_data("Right")
        
        tf.system.camera_view_back()
        tf.system.camera_zoom_to(-12)
        _export_and_delete_a_junk_screenshot()
        _save_camera_data("Back")
        
        tf.system.camera_view_front()
        tf.system.camera_zoom_to(-12)
        _export_and_delete_a_junk_screenshot()
        _save_camera_data("Front")
    
def _save_camera_data(side: str) -> None:
    """side = "Front", "Back", "Left", "Right"."""
    center: tf.fVector3 = tf.system.camera_center()
    rotation: tf.fQuaternion = tf.system.camera_rotation()
    zoom: float = tf.system.camera_zoom()

    _camera_data[side] = CameraData(center, rotation, zoom)
    
def _load_camera_data(side: str) -> None:
    """Do this just before rendering, to make sure everything stays in sync"""
    data: CameraData = _camera_data[side]
    tf.system.camera_move_to(data.center, data.rotation, data.zoom)

def get_state() -> dict:
    """generate state to be saved to disk
    
    In composite runs, pick up the screenshot timing where we left off, and restore camera position
    
    Other aspects of state should automatically reconstitute themselves pretty well:
    _image_path, _rotation_started
    """
    camera_json: dict = {}
    if not cfg.windowed_mode:
        # In windowed mode, camera data cannot be captured. (save_camera_data() fails.) So there's no
        # way to save it to disk and recover that state later. This means, if you try to save and restore
        # state late in the sim, after rotation has begun, all camera info will be lost.
        for side in ["Left", "Right", "Back", "Front"]:
            data: CameraData = _camera_data[side]
            center: tf.fVector3 = data.center
            zoom: float = data.zoom
            rotation: tf.fQuaternion = data.rotation
            angle: float = rotation.angle()
            axis: tf.fVector3 = rotation.axis()
            camera_json[side] = {"camera_center": center.as_list(),
                                 "camera_zoom": zoom,
                                 "camera_angle": angle,
                                 "camera_axis": axis.as_list(),
                                 }

    return {"previous_step": _previous_screenshot_timestep,
            "current_step": _current_screenshot_timestep,
            "rotation_finished": _rotation_finished,
            "camera_data": camera_json,
            }

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved.

    In this case, we do not need to increment _current, because save_screenshot_repeatedly pre-increments it to
    represent the *next* timestep. (Compare the comparable function in sim_state_export, where the opposite is true.)
    """
    global _previous_screenshot_timestep, _current_screenshot_timestep, _rotation_finished
    _previous_screenshot_timestep = d["previous_step"]
    _current_screenshot_timestep = d["current_step"]
    _rotation_finished = d["rotation_finished"]
    
    if cfg.windowed_mode:
        # No camera state was saved, so recover as best we can.
        # I suppose I could calculate the position, if I were to capture and save the screenshot_timestep value
        # when rotation started, then get the number of screenshots (hence rotations) that would have occurred
        # since then, and from that get the new camera position. But, nah.
        if _rotation_finished:
            tf.system.camera_view_bottom()
            tf.system.camera_zoom_to(-12)
        else:
            tf.system.camera_view_front()
            tf.system.camera_zoom_to(-12)
        return
    
    camera_json: dict = d["camera_data"]
    for side, side_value in camera_json.items():
        camera_center: tf.fVector3 = tf.fVector3(side_value["camera_center"])
        camera_zoom: float = side_value["camera_zoom"]
        camera_angle: float = side_value["camera_angle"]
        camera_axis: tf.fVector3 = tf.fVector3(side_value["camera_axis"])
        camera_rotation: tf.fQuaternion = tf.fQuaternion.rotation(camera_angle, camera_axis)
        _camera_data[side] = CameraData(camera_center, camera_rotation, camera_zoom)
    _load_camera_data("Front")
    
    # _load_camera_data() doesn't actually change the camera state; it just says what it WILL move to,
    # on the subsequent render. So we need to render right now, in order get things in sync, so that
    # when the camera state is examined (see _test_when_to_finish_rotation()), it will be correct.
    _export_and_delete_a_junk_screenshot()

def _export_and_delete_a_junk_screenshot() -> None:
    """The only way I know to force a render. Necessary to make recent camera changes readable."""
    if cfg.windowed_mode:
        return
    
    path: str = os.path.join(_image_path, "junk.jpg")
    tf.system.screenshot(path)
    os.remove(path)

def make_movie(filename: str = None) -> None:
    if not screenshot_export_enabled():
        return

    sides: list[str] = ["Front"] if cfg.windowed_mode else ["Left", "Right", "Back", "Front", "Top"]
    with os.scandir(_image_path) as dir_entries_it:
        dir_entries_chron: list[os.DirEntry] = sorted(dir_entries_it, key=lambda entry: entry.stat().st_mtime_ns)
    for side in sides:
        image_filepaths = [entry.path
                           for entry in dir_entries_chron
                           if entry.name.endswith(f"{side}.jpg")]
        print(f"Assembling movie \"{side}\" from {len(image_filepaths)} images")
        
        # Normally, assemble a movie. But if we are doing Patch lineage tracing, don't bother with any but Front & Top
        if cfg.paint_pattern != cfg.PaintPattern.PATCH or side in ["Front", "Top"]:
            clip = ImageSequenceClip.ImageSequenceClip(image_filepaths, fps=24)
            
            # Save the clip to the Screenshots subdirectory using tfu.export_directory() also as the movie name
            # (Or, if filename was provided, then __name__ == "__main__", see below.)
            if filename is None:
                filename = tfu.export_directory()
            clip.write_videofile(os.path.join(_movie_path, filename + f" {side}.mp4"))
        
        # Discard all the exported image files.
        # (But not if there was an exception, because I may still need them.)
        # (And not if simulation still running in a different process, because definitely still need them.)
        # (And if we are doing lineage tracing with a patch, retain the initial and final image for Front and
        # Top sides. Note that the final image will be taken from an oblique camera angle so will not be the
        # same as the image captured later by final_result_screenshots(), which will be from the side.)
        # (And if doing elastic stretch (no bond remodeling), retain the initial image for all 4 side views.)
        if cfg.paint_pattern == cfg.PaintPattern.PATCH and side in ["Front", "Top"]:
            image_filepaths.pop()
            image_filepaths.pop(0)
        elif not cfg.bond_remodeling_enabled and side != "Top":
            image_filepaths.pop(0)
        if not events.event_exception_was_thrown() and not _retain_screenshots_after_movie:
            print(f"\nRemoving {len(image_filepaths)} images...")
            for path in image_filepaths:
                os.remove(path)

    # Notes on codec and file type:
    # .mp4 (defaults to codec "libx264"): looks okay; quality not as good as the jpg files it's made from;
    #   "quality tunable using bitrate argument". (See below.)
    # .mp4, codec = mpeg4
    # .mov, codec = mpeg4
    #   These two are identical as far as I can tell. (Same codec, different wrappers.) Precisely the same file size,
    #   and the same quality. Quality is TERRIBLE for some reason, starting out looking same as the default, but
    #   getting worse as the simulation proceeds. File size is a bit smaller. Unclear whether these are also "tunable".
    # .avi (2 different codecs): much bigger files, promises "perfect quality", but unfortunately I can't open them.
    
    # Notes on bitrate argument: Barely documented at all, except that another arg called audio_bitrate is a string
    # with a number followed by "k", so I tried that for video bitrate, too.
    # But experiment shows:
    #   "50k" gives a tiny file (42 KB instead of 3 MB) with an utterly degraded image;
    #   "3000k" gives a file just slightly smaller than the default (1.8 MB instead of 2.1 MB) and with no
    #       noticeable quality difference.
    #   "10000k" and "30000k" both give much larger files (8 and 24 MB respectively, vs. 3), with again no discernable
    #       difference in quality. So, what's the point? Leave bitrate argument out, accept the default ("None").
    #       (Actually, this *does* improve the quality of the mpeg4 codecs, getting them comparable to the default one.
    #       But still, only at the expense of having a much larger file, so why bother? Though I wonder... how high
    #       can you go, and how much better can quality get? Can you approach back to the quality of the origin jpgs?)
    #       (Also: it's all relative. For a full-length epiboly video, file size of 10000K mpeg4 was 63 MB vs 16 for
    #       the default. Bigger, but this is still tiny compared to what you get from the MacOS video capture.)
    
def make_movie_in_post(directory_name: str, retain_screenshots: bool) -> None:
    global _image_path, _movie_path, _retain_screenshots_after_movie
    
    _retain_screenshots_after_movie = retain_screenshots
    base_path: str = tfu.export_path(directory_name)
    _image_path = os.path.join(base_path, _screenshots_subdirectory)
    _movie_path = os.path.join(base_path, _movie_subdirectory)
    make_movie(directory_name)

if __name__ == "__main__":
    # Be sure to supply the directory name before running this.
    # This is the parent directory with the datetime in the name, not the "Screenshots" subdirectory.
    #
    # Typically, simulation is still running in another process (otherwise, better to run main instead of this),
    # so we don't want to delete the screenshots. But if simulation is finished, set retain_screenshots = False
    make_movie_in_post(directory_name="Directory name goes here", retain_screenshots=True)
