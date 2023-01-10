"""Export simulation screenshots"""
from datetime import datetime, timezone
import os
import os.path as path

import tissue_forge as tf
from utils import tf_utils as tfu

def timestring() -> str:
    # timezone-aware local time in local timezone:
    local: datetime = datetime.now(timezone.utc).astimezone()
    
    # 12-hr clock, with am/pm and timezone, and no colons, e.g. '2023-01-07 12-50-35 AM PST'
    # (i.e., suitable for directory or file name)
    return local.strftime("%Y-%m-%d %I-%M-%S %p %Z")

def init_screenshots() -> None:
    """
    Set up directories for all image output: root directory for all output from the current script, ever;
    and a subdirectory for all output of the CURRENT RUN of the current script.

    Maybe ToDo: output a text file to that directory? With lots of metadata. DateTime, params, etc.?
    """
    global _image_dir
    if not screenshot_export_enabled():
        return
    
    # one directory for all TF output from this script, ever:
    image_root = path.expanduser("~/TissueForge_image_export/")
    
    # subdirectory with unique name for all output of the current run:
    _image_dir = image_root + timestring()
    
    # Creates the parent directory if it doesn't yet exist; and the subdirectory UNLESS it already exists:
    os.makedirs(_image_dir)

def _export_screenshot(filename: str) -> None:
    path: str = f"{_image_dir}/{filename}"
    print(f"Saving file to '{path}'")
    result: int = tf.system.screenshot(path, decorate=False, bgcolor=[0, 0, 0])
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
    
    _current_screenshot_timestep += 1

_image_dir: str
_previous_screenshot_timestep: int = 0
_current_screenshot_timestep: int = 0

# Screenshot export: Use 0 to mean, display rendering only, no export;
# Anything greater than 0, export only, and no display rendering
# (That was the intent, for a workaround that unfortunately did not work around.
# So for now, can export but still have to run everything in the simulator.)
_screenshot_export_interval: int = 100

def screenshot_export_enabled() -> bool:
    """Read-only version of _screenshot_export_interval to be used by callers as a flag"""
    return _screenshot_export_interval != 0

init_screenshots()
