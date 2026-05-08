# main.py
# This is the main entry point for the Attentionator application. It initializes the session, launches the GUI, and manages the flow of data between the different components of the application.

import threading
import os
import session
import gui
import live_analysis
from tkinter import messagebox
from enums import VideoMode
import calibration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Run calibration if no model exists yet
    model_exists = os.path.exists(os.path.join(BASE_DIR, "model", "random_forest.joblib"))
    if not model_exists:
        calibration.run()
    # Display the launch window and get the selected video mode
    current_session = session.Session()
    launch_window = gui.LaunchWindow()
    current_session.set_video_mode(launch_window.get_selected_video_mode())

    # Display the video mode information window and get the video path if necessary
    video_mode_information_window = gui.VideoModeInformationWindow(mode=current_session.get_video_mode())
    current_session.set_video_path(video_mode_information_window.get_video_path())

    # Start the appropriate analysis based on the selected video mode
    if current_session.get_video_mode() == VideoMode.LIVE:
        live_analysis.main()
    elif current_session.get_video_mode() == VideoMode.RECORDING:
        video_path = current_session.get_video_path()
        if not os.path.isfile(video_path):
            messagebox.showerror("Error", f"The video file '{video_path}' does not exist.")
            return

if __name__ == "__main__":
    main()