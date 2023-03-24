"""Logging

    in case sim is ended and restarted from exported data, prevent new log file from overwriting
    the old one. Keep both as sequentially numbered files.
    
    (Note, originally logging.py, but this caused moviepy to crash under certain circumstances
    (apparently only when running video_export as __main__), because moviepy has its own logging.py,
    but it loaded mine instead of its own! Renaming this module fixed it.)
"""
import os

import tissue_forge as tf
import utils.tf_utils as tfu

_logfile_num: int = 1

def init_logging() -> None:
    logfile_path: str = os.path.join(tfu.export_path(), f"Epiboly_{_logfile_num}.log")
    tf.Logger.enableFileLogging(fileName=logfile_path, level=tf.Logger.ERROR)

def get_state() -> dict:
    """In composite runs, produce a separate numbered log file for each segment of the run."""
    return {"logfile_num": _logfile_num}

def set_state(d: dict) -> None:
    """Reconstitute state of module from what was saved.

    Increment _logfile_num with each run to generate a new filename, hence separate file
    """
    global _logfile_num
    _logfile_num = d["logfile_num"] + 1

