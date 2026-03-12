# session.py
# This module defines the Session class, which is responsible for managing the session data.

from enums import VideoMode

class Session:
    
    def __init__(self):
        self.video_mode = None
        self.video_path = None

    def set_video_mode(self, mode):
        self.video_mode = mode
        
    def get_video_mode(self):
        return self.video_mode
    
    def set_video_path(self, path):
        self.video_path = path

    def get_video_path(self):
        return self.video_path