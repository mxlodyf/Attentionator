# gui.py
# This module defines the graphical user interface (GUI) for the Attentionator application, which provides users with an interactive way to select video modes (live webcam or video recording) and view analysis results.

import threading
import sys
import cv2
import numpy as np
from pathlib import Path
from enums import VideoMode
from tkinter import Tk, Canvas, Entry, Button, PhotoImage, Label, messagebox
from PIL import Image, ImageTk

INTRODUCTORY_WINDOW_WIDTH = 500 
INTRODUCTORY_WINDOW_CONTENT_WIDTH = 440
LAUNCH_WINDOW_HEIGHT = 420
VIDEO_MODE_INFORMATION_WINDOW_HEIGHT = 500

BACKGROUND_COLOUR = "#F3F3F3"

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets"
TEXT_PATH = OUTPUT_PATH / "text"

def relative_to_project(path: str) -> Path:
    return OUTPUT_PATH / Path(path)

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def relative_to_text(path: str) -> Path:
    return TEXT_PATH / Path(path)

class BaseWindow:

    def __init__(self):
        self.window = Tk()
        self.window.geometry("500x420")
        self.window.configure(bg = BACKGROUND_COLOUR)
        self.canvas = Canvas(
            self.window,
            bg = BACKGROUND_COLOUR,
            height = LAUNCH_WINDOW_HEIGHT,
            width = INTRODUCTORY_WINDOW_WIDTH,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )
        self.window.resizable(False, False)

# LaunchWindow
# Responsible for displaying the initial launch window where users can select between live webcam mode and video recording mode.
class LaunchWindow(BaseWindow):

    def __init__(self):
        super().__init__()
        self.selected_video_mode = None
        self.display()

    def display(self):
        self.canvas.place(x = 0, y = 0)

        self.canvas.create_text(
            30.0,
            20.0,
            anchor="nw",
            text=load_text("application_introduction.txt"),
            fill="#000000",
            font=("ArialMT", 12 * -1),
            width=INTRODUCTORY_WINDOW_CONTENT_WIDTH
        )
        
        live_webcam_button_image = PhotoImage(file=relative_to_assets("live_webcam_button.png"))

        live_webcam_button = Button(
            image=live_webcam_button_image,
            borderwidth=0,
            highlightthickness=0,
            command=self.handle_live_button_click,
            relief="flat",
            bg="#FFFFFF",      
            activebackground="#FFFFFF"
        )

        live_webcam_button.place(
            x=136.0,
            y=310.0,
            width=229.0,
            height=39.0
        )

        video_recording_button_image = PhotoImage(file=relative_to_assets("video_recording_button.png"))

        video_recording_button = Button(
            image=video_recording_button_image,
            borderwidth=0,
            highlightthickness=0,
            command=self.handle_recording_button_click,
            relief="flat",
            bg="#FFFFFF",      
            activebackground="#FFFFFF"
        )

        video_recording_button.place(
            x=136.0,
            y=360.0,
            width=229.0,
            height=39.0
        )
        
        self.window.resizable(False, False)
        self.window.mainloop()

    def handle_live_button_click(self):
        self.set_selected_video_mode(VideoMode.LIVE)
        self.window.destroy()

    def handle_recording_button_click(self):
        self.set_selected_video_mode(VideoMode.RECORDING)
        self.window.destroy()
    
    def set_selected_video_mode(self, mode):
        self.selected_video_mode = mode

    def get_selected_video_mode(self):
        return self.selected_video_mode
    
    def show_selection_error(self):
        messagebox.showerror("Error", "Please select an exercise from the drop down menu.")

# VideoModeInformationWindow
# Responsible for displaying information about the selected video mode and allowing users to input a video path if they selected video recording mode. It also contains a launch button that users can click to proceed to the analysis phase after they have read the information and inputted a video path if necessary.
class VideoModeInformationWindow(BaseWindow):

    def __init__(self, mode=VideoMode.LIVE):
        super().__init__()
        self.video_mode = mode
        self.video_path = None
        self.video_path_input_field = None
        self.display(mode)

    def display(self, mode):
        self.canvas.place(x = 0, y = 0)
        launch_button_image = PhotoImage(file=relative_to_assets("launch_button.png"))

        # Display Video Recording Mode information and input field.
        if mode == VideoMode.RECORDING:

            self.canvas.create_text(
                30.0,
                20.0,
                anchor="nw",
                text=load_text("video_recording_mode_information.txt"),
                fill="#000000",
                font=("ArialMT", 12 * -1),
                width=INTRODUCTORY_WINDOW_CONTENT_WIDTH
            )

            self.canvas.create_text(
                30.0,
                295.0,
                anchor="nw",
                text="Video Path",
                fill="#000000",
                font=("ArialMT", 14 * -1)
            )
            
            self.video_path_input_field = Entry(
                bd=0,
                bg="#E6E6E6",
                fg="#000716",
                highlightthickness=0
            )

            self.video_path_input_field.place(
                x=30.0,
                y=315.0,
                width=INTRODUCTORY_WINDOW_CONTENT_WIDTH,
                height=30.0
            )

        # Display Live Video Mode information.
        elif mode == VideoMode.LIVE:

            self.canvas.create_text(
                30.0,
                20.0,
                anchor="nw",
                text=load_text("live_webcam_mode_information.txt"),
                fill="#000000",
                font=("ArialMT", 12 * -1),
                width=INTRODUCTORY_WINDOW_CONTENT_WIDTH
            )

        # Place launch button for both modes.
        self.launch_button = Button(
            image=launch_button_image,
            borderwidth=0,
            highlightthickness=0,
            command=self.handle_launch_button_click,
            relief="flat",
            bg="#FFFFFF",      
            activebackground="#FFFFFF"
        )

        self.launch_button.place(
            x=136.0,
            y=360.0,
            width=229.0,
            height=39.0
        )

        self.window.resizable(False, False)
        self.window.mainloop()
    
    def set_video_mode(self, mode):
        self.video_mode = mode
    
    def get_video_mode(self):
        return self.video_mode
    
    def set_video_path(self):
        if self.video_path_input_field is not None:
            self.video_path = self.video_path_input_field.get()

    def get_video_path(self):
        return self.video_path
    
    def handle_launch_button_click(self):
        self.set_video_path()
        self.window.destroy()

# Helper function to load text from a file in the text directory.
def load_text(filename: str) -> str:
    with open(relative_to_text(filename), "r") as f:
        return f.read()