# enums.py
# This module defines the VideoMode enumeration, which represents the different video modes available.

from enum import Enum

class VideoMode(Enum):
    LIVE = "live"
    RECORDING = "recording"