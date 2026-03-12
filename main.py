# main.py
# This is the main entry point for the Attentionator application. It initializes the session, launches the GUI, and manages the flow of data between the different components of the application.

import session
import gui
import threading
import os
from tkinter import messagebox
from enums import VideoMode

def main():
    current_session = session.Session()
    launch_window = gui.LaunchWindow()
    current_session.set_video_mode(launch_window.get_selected_video_mode())
    video_mode_information_window = gui.VideoModeInformationWindow(mode=current_session.get_video_mode())
    video_path = video_mode_information_window.get_video_path()
    current_session.set_video_path(video_path)
    analysis_window = gui.VideoRecordingAnalysisWindow()

    if not os.path.isfile(video_path):
            messagebox.showerror("Error", f"The video file '{video_path}' does not exist.")

if __name__ == "__main__":
    main()