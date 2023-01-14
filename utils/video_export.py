"""Export simulation screenshots and compile them into movies

StackOverflow: https://stackoverflow.com/a/62434934
Documentation of MoviePy: https://zulko.github.io/moviepy/index.html
"""
from datetime import datetime, timezone
import math
import os
import os.path as path

import moviepy.video.io.ImageSequenceClip as movieclip

import tissue_forge as tf
from epiboly_init import LeadingEdge, windowless
from utils import epiboly_utils as epu
from utils import tf_utils as tfu

_rotation_started: bool = False
_rotation_finished: bool = False

def _test_when_to_start_rotation() -> None:
    """At the appropriate time, trigger rotation by changing the value of _rotation_started from False to True"""
    global _rotation_started
    
    # Position of leading edge at which we start rotating the camera. π/2 is good for testing (the equator,
    # which is reached by the leading edge early in the simulation). For full epiboly, wait until close to veg pole.
    rotation_start_position: float = math.pi * 0.8
    
    arbitrary_edge_particle: tf.ParticleHandle = LeadingEdge.items()[0]
    leading_edge_progress: float = epu.embryo_phi(arbitrary_edge_particle)
    if leading_edge_progress > rotation_start_position:
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
    if not windowless:
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
    
def _timestring() -> str:
    # timezone-aware local time in local timezone:
    local: datetime = datetime.now(timezone.utc).astimezone()
    
    # 12-hr clock, with am/pm and timezone, and no colons, e.g. '2023-01-07 12-50-35 AM PST'
    # (i.e., suitable for directory or file name)
    return local.strftime("%Y-%m-%d %I-%M-%S %p %Z")

def _init_screenshots() -> None:
    """
    Set up directories for all image output: root directory for all output from the current script, ever;
    and a subdirectory for all output of the CURRENT RUN of the current script.

    Maybe ToDo: output a text file to that directory? With lots of metadata. DateTime, params, etc.?
    ToDo: in fact, maybe output the console, to capture any errors, etc.? Important for automated runs.
    """
    global _image_dir, _image_path
    if not screenshot_export_enabled():
        return
    
    # one directory for all TF output from this script, ever:
    image_root = path.expanduser("~/TissueForge_image_export/")
    
    # subdirectory with unique name for all output of the current run:
    _image_dir = _timestring()
    
    # full path to that directory
    _image_path = os.path.join(image_root, _image_dir)
    
    # Creates the parent directory if it doesn't yet exist; and the subdirectory UNLESS it already exists:
    os.makedirs(_image_path)
    
def _export_screenshot(filename: str) -> None:
    path: str = os.path.join(_image_path, filename)
    print(f"Saving file to '{path}'")
    result: int = tf.system.screenshot(path, bgcolor=[0, 0, 0])
    if result != 0:
        print(tfu.bluecolor + f"Something went wrong with screenshot export, result code = {result}" + tfu.endcolor)

def save_screenshot(filename: str, show_timestep: bool = True) -> None:
    """
    Didn't work: jpg gives blank image (black even if I set a different bgcolor)
        (**Until** I launch the simulator; then it works after that, even after dismissing the simulator!
        Could have sworn I had already tried that.)
    Tried all the other formats, they crash the app, segfault!

    For use outside of timestep events. A single one-off export. Caller provides filename (no extension).
    Timestep will be appended to filename unless show_timestep = False (and filename is not blank).
    """
    if not screenshot_export_enabled():
        return
    
    suffix: str = f"Timestep = {_current_screenshot_timestep}"
    suffix += f"; Universe.time = {round(tf.Universe.time, 2)}"
    if not filename:
        filename = suffix
    elif show_timestep:
        filename += "; " + suffix
    
    _export_screenshot(filename + ".jpg")

def save_screenshot_repeatedly() -> None:
    """For use inside timestep events. Keeps track of export interval, and names files accordingly"""
    global _previous_screenshot_timestep, _current_screenshot_timestep
    if not screenshot_export_enabled():
        return
    
    # Note that this implementation means that the first time this function is ever called, the screenshot
    # will always be saved, and will be defined (and labeled) as Timestep 0. Even if the simulation has
    # been running before that, and Universe.time > 0.
    # But also note that this image is really (I think) simulation timestep 1, EVEN IF the simulation is
    # starting fresh. Because (I think) by the time the event gets called in Timestep 0, the physics has
    # already been integrated, and particles have moved. Currently unable to get a screenshot before that,
    # until this works without running tf.show().
    
    elapsed: int = _current_screenshot_timestep - _previous_screenshot_timestep
    if elapsed % _screenshot_export_interval == 0:
        _previous_screenshot_timestep = _current_screenshot_timestep
        save_screenshot("")  # just timestep as filename
        _automated_camera_rotate()
    
    _current_screenshot_timestep += 1

_image_dir: str
_image_path: str
_previous_screenshot_timestep: int = 0
_current_screenshot_timestep: int = 0

# Caller settable parameter (through the setter only)
# Screenshot export: Use 0 to mean no export;
# Anything greater than 0, counts timesteps between exports.
_screenshot_export_interval: int = 10

def set_screenshot_export_interval(interval: int = 10) -> None:
    """Safely set screenshot interval
    
    (Interval may need to change during the simulation.)
    Call with no arg to reset to the default value (e.g. after changing it temporarily)
    
    Enabling/disabling screenshots happens at launch and can't be changed thereafter. Thus:
    if disabled (stored interval == 0), then caller may not change it.
    if enabled (stored interval > 0), then caller may not change it to 0.
    
    But, remember that once enabled, the task list for repeated events can always be changed to start/stop calling
    the "save_screenshot" functions.
    """
    if screenshot_export_enabled():
        if interval > 0:
            _screenshot_export_interval = interval
        else:
            print(tfu.bluecolor + "Warning: screenshot export cannot be disabled after initialization" + tfu.endcolor)
    else:
        if interval != 0:
            print(tfu.bluecolor + "Warning: screenshot export cannot be enabled after initialization" + tfu.endcolor)

def screenshot_export_enabled() -> bool:
    """Convenience function. Interpret screenshot_export_interval as flag for whether export is enabled"""
    return _screenshot_export_interval != 0

def make_movie() -> None:
    if not screenshot_export_enabled():
        return

    with os.scandir(_image_path) as dir_entries_it:
        dir_entries_chron: list[os.DirEntry] = sorted(dir_entries_it, key=lambda entry: entry.stat().st_mtime_ns)
    image_filenames = [entry.path
                       for entry in dir_entries_chron
                       if entry.name.endswith(".jpg")]
    print(f"Assembling movie from {len(image_filenames)} images")
    clip = movieclip.ImageSequenceClip(image_filenames, fps=24)

    # Save the movie clip using the directory name also as the movie name, and save it to that same directory
    clip.write_videofile(os.path.join(_image_path, _image_dir + ".mp4"))

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
    
def make_movie_in_post(directory_name: str) -> None:
    global _image_dir, _image_path
    image_root = path.expanduser("~/TissueForge_image_export/")
    _image_dir = directory_name
    _image_path = os.path.join(image_root, _image_dir)
    make_movie()

if __name__ == "__main__":
    # Be sure to supply the directory name before running this
    make_movie_in_post(directory_name="Directory name goes here")
else:
    _init_screenshots()

