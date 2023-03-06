"""Export simulation screenshots and compile them into movies

All export functions are no-ops unless export has been enabled in config.py

StackOverflow: https://stackoverflow.com/a/62434934
Documentation of MoviePy: https://zulko.github.io/moviepy/index.html
"""
import math
import os

import moviepy.video.io.ImageSequenceClip as movieclip

import tissue_forge as tf
import config as cfg
import control_flow.events as events
from utils import epiboly_utils as epu
from utils import tf_utils as tfu

def final_result_screenshots() -> None:
    """If enabled, capture still images from multiple angles.

    Note: This doesn't work in windowed mode; all 4 images come out the same. I.e., the camera updates
    don't result in a display update. Not worth trying to get around.
    """
    if not cfg.windowed_mode and screenshot_export_enabled():
        tf.system.camera_view_front()
        tf.system.camera_zoom_to(-12)
        save_screenshot("Front", show_timestep=False)
        
        tf.system.camera_view_left()
        tf.system.camera_zoom_to(-12)
        save_screenshot("Left", show_timestep=False)
        
        tf.system.camera_view_back()
        tf.system.camera_zoom_to(-12)
        save_screenshot("Back", show_timestep=False)
        
        tf.system.camera_view_right()
        tf.system.camera_zoom_to(-12)
        save_screenshot("Right", show_timestep=False)

        tf.system.camera_view_top()
        tf.system.camera_zoom_to(-12)
        save_screenshot("Top", show_timestep=False)

        tf.system.camera_view_bottom()
        tf.system.camera_zoom_to(-12)
        save_screenshot("Bottom", show_timestep=False)

_rotation_started: bool = False
_rotation_finished: bool = False

def _test_when_to_start_rotation() -> None:
    """At the appropriate time, trigger rotation by changing the value of _rotation_started from False to True"""
    global _rotation_started
    
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
    """
    global _rotation_finished
    
    # Camera angle starts at 1.5π at Front position (pointing at equator), drops to π at Bottom (pointing at vegetal
    # pole), and then continues to drop lower if you rotate past Bottom and up the other side. So we want to stop
    # when angle < π. However, it does not hit exactly. At very bottom, value is slightly greater than π;
    # stopping at π thus overshoots by one rotation increment. So add a small tolerance to stop in the right place.
    target_camera_angle: float = math.pi + 0.04
    
    quat: tf.fQuaternion = tf.system.camera_rotation()
    if quat.angle() < target_camera_angle:
        _rotation_finished = True

def _automated_camera_rotate() -> None:
    """Camera control during windowless export"""
    if cfg.windowed_mode:
        # This function is mainly intended for windowless export; not needed while simulator is displayed.
        # But, to observe the behavior of this function in real time, comment out this if statement
        return
    
    if not _rotation_started:
        _test_when_to_start_rotation()
    
    if _rotation_started and not _rotation_finished:
        # Note that "up" is "down". Behavior of "rotate up" is that the camera moves down toward vegetal.
        # The camera angle indeed "rotates up" around it's OWN axis while its position rotates DOWN relative to the
        # universe at the same time – continuing to point toward the center of the universe the whole time.
        # The effect is that the particles rotate upward in the field of view, i.e. the camera rotates downward.
        tf.system.camera_rotate_up()
        _test_when_to_finish_rotation()
    
def init_screenshots() -> None:
    """
    Set up subdirectory for all screenshot and movie output
    
    tfu.init_export() should have been run before running this, to create the parent directories.
    """
    global _image_path
    if not screenshot_export_enabled():
        return
    
    _image_path = os.path.join(tfu.export_path(), "Screenshots")
    os.makedirs(_image_path, exist_ok=True)
    
def _export_screenshot(filename: str) -> None:
    """Note, in system.screenshot(), bgcolor (and decorate) args only work in windowless mode."""
    path: str = os.path.join(_image_path, filename)
    print(f"Saving screenshot to '{path}'")
    result: int = tf.system.screenshot(path, bgcolor=[0, 0, 0])
    if result != 0:
        print(tfu.bluecolor + f"Something went wrong with screenshot export, result code = {result}" + tfu.endcolor)

def save_screenshot(filename: str, show_timestep: bool = True) -> None:
    """
    Calling this method directly, is intended for one-off export operations *outside* of timestep events. Thus it
    only works in windowless mode. In windowed mode, always access this through save_screenshot_repeatedly().
    
    Caller provides filename (no extension). Saves as jpg. All other formats crash the app, segfault!
    Timestep will be appended to filename unless show_timestep = False (and filename is not blank).
    """
    if not screenshot_export_enabled():
        return
    
    suffix: str = f"Timestep = {_current_screenshot_timestep}"
    suffix += f"; Universe.time = {round(tf.Universe.time, 2)}"
    suffix += f"; phi = {round(epu.leading_edge_mean_phi(), 2)}"
    suffix += f"; veloc.z = {round(epu.leading_edge_velocity_z(), 4)}"
    if not filename:
        filename = suffix
    elif show_timestep:
        filename += "; " + suffix
    
    _export_screenshot(filename + ".jpg")

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
    # (in windowless mode only) by calling a one-off save_screenshot() before calling tf.step().
    
    elapsed: int = _current_screenshot_timestep - _previous_screenshot_timestep
    if elapsed % _screenshot_export_interval == 0:
        _previous_screenshot_timestep = _current_screenshot_timestep
        save_screenshot("")  # just timestep as filename
        _automated_camera_rotate()
    
    _current_screenshot_timestep += 1

_image_path: str
_previous_screenshot_timestep: int = 0
_current_screenshot_timestep: int = 0

# module's copy can be adjusted dynamically
_screenshot_export_interval: int = cfg.screenshot_export_interval

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
            interval = cfg.screenshot_export_interval
        if interval > 0:
            _screenshot_export_interval = interval
            print(f"Screenshot export interval set to {interval} timesteps")
        else:
            print(tfu.bluecolor + "Warning: screenshot export cannot be disabled after initialization" + tfu.endcolor)

def screenshot_export_enabled() -> bool:
    """Convenience function. Interpret cfg.screenshot_export_interval as flag for whether export is enabled"""
    return cfg.screenshot_export_interval != 0

def get_state() -> dict:
    """generate state to be saved to disk
    
    For now, in composite runs, just try to pick up the screenshot timing where we left off.

    Could add: camera position and angle, in case camera no longer at the default position.
    
    Other aspects of state should automatically reconstitute themselves pretty well:
    _image_path, _rotation_started, _rotation_finished
    """
    return {"previous_step": _previous_screenshot_timestep,
            "current_step": _current_screenshot_timestep}

def set_state(d: dict) -> None:
    """Reconstitute state from what was saved.

    In this case, we do not need to increment _current, because save_screenshot_repeatedly pre-increments it to
    represent the *next* timestep. (Compare the comparable function in sim_state_export, where the opposite is true.)
    
    ToDo: Maybe automatically delete, or hide away, all the screenshots that were saved *after* the final
     state export? (Because screenshots happen much more frequently than state exports.) So that those won't
     get included in the movie. They will all be regenerated during the resumed simulation. For now, I'll
     need to remove them manually before the movie gets compiled.
    """
    global _previous_screenshot_timestep, _current_screenshot_timestep
    _previous_screenshot_timestep = d["previous_step"]
    _current_screenshot_timestep = d["current_step"]

def make_movie(filename: str = None) -> None:
    if not screenshot_export_enabled():
        return

    with os.scandir(_image_path) as dir_entries_it:
        dir_entries_chron: list[os.DirEntry] = sorted(dir_entries_it, key=lambda entry: entry.stat().st_mtime_ns)
    image_filepaths = [entry.path
                       for entry in dir_entries_chron
                       if entry.name.endswith(".jpg")]
    print(f"Assembling movie from {len(image_filepaths)} images")
    clip = movieclip.ImageSequenceClip(image_filepaths, fps=24)
    
    # Save the clip using tfu.export_directory() also as the movie name, and save it to the Screenshots subdirectory
    # (Or, if filename was provided, then __name__ == "__main__", see below.)
    if filename is None:
        filename = tfu.export_directory()
    clip.write_videofile(os.path.join(_image_path, filename + ".mp4"))
    
    # Discard all the exported image files except the final one.
    # (But not if there was an exception, because I may still need them.)
    if not events.event_exception_was_thrown():
        image_filepaths.pop()   # remove final item
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
    
def make_movie_in_post(directory_name: str) -> None:
    global _image_path
    _image_path = os.path.join(tfu.export_path(directory_name), "Screenshots")
    make_movie(directory_name)

if __name__ == "__main__":
    # Be sure to supply the directory name before running this.
    # This is the parent directory with the datetime in the name, not the "Screenshots" subdirectory
    make_movie_in_post(directory_name="Directory name goes here")
