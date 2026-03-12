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

class VideoModeInformationWindow(BaseWindow):

    def __init__(self, mode=VideoMode.LIVE):
        super().__init__()
        self.video_mode = mode
        self.video_path = None
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
    
class VideoRecordingAnalysisWindow(BaseWindow):

    def __init__(self, mode=VideoMode.LIVE, video_path=None):
        super().__init__()
        self.window.geometry("1000x500")
        self.canvas.config(height=500, width=1000)
        self.images = {
            "cross": relative_to_assets("cross.png"),
            "exclamation_mark": relative_to_assets("exclamation_mark.png"),
            "check_mark": relative_to_assets("check_mark.png")
        }
        self.text = {}
        # Label to display the original video.
        self.original_video_label = Label(self.window)
        self.original_video_label.place(x=0, y=0)
        # Label to display the annotated video.
        self.processed_video_label = Label(self.window)
        self.processed_video_label.place(x=(959 - 651), y=0)

        self.analysis = None

    def display(self, analysis):
        self.analysis = analysis
        self.canvas.place(x = 0, y = 0)
        
        background_image = PhotoImage(file=relative_to_assets("grey_rectangle.png"))
        self.canvas.create_image(
            651 + (308 / 2),  
            170,
            image=background_image,
            anchor="center"
        )
    
        # Ensure the image is not garbage collected.
        self.canvas.background_image = background_image

        # Create the custom button icons.
        replay_icon = PhotoImage(file=relative_to_assets("replay_button.png"))
        change_icon = PhotoImage(file=relative_to_assets("change_button.png"))
        exit_icon = PhotoImage(file=relative_to_assets("exit_button.png"))

        # Replay button.
        replay_label = Label(self.window, image=replay_icon, bg="#F3F3F3")
        replay_label.place(x=650, y=420, width=100, height=50)
        replay_label.bind("<Button-1>", lambda e: self.replay_video())

        # Change button.
        change_label = Label(self.window, image=change_icon, bg="#F3F3F3")
        chane_label.place(x=760, y=420, width=100, height=50)
        change_label.bind("<Button-1>", lambda e: self.change_video())

        # Exit button.
        exit_label = Label(self.window, image=exit_icon, bg="#F3F3F3")
        exit_label.place(x=870, y=420, width=100, height=50)
        exit_label.bind("<Button-1>", lambda e: self.exit_app())

        # Ensure the images are not garbage collected.
        replay_label.image = replay_icon
        change_label.image = change_icon
        exit_label.image = exit_icon

        self.window.resizable(False, False)
        self.window.mainloop()
                
    # Replay the video if the 'Replay' button is clicked.
    def replay_video(self):
        if not self.analysis.is_playing:  # Check if a video is already playing.
            self.analysis.is_playing = True
            threading.Thread(target=self.analysis.process_video, args=(self,), daemon=True).start()
            self.display(self.analysis)

    def exit_app(self):
        sys.exit(0)


    def change_video(self):
        self.window.destroy()
        import main
        main.main()

    def update_video_frame(self, frame, label):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        max_height = 500
        max_width = 959 - 651
        scale = max_height / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        # Create a new image with a white background (change the color here)
        canvas_image = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255  # 255 for white background
        # Crop the frame evenly from left and right if it exceeds 'max_width'.
        if new_width > max_width:
            excess_width = new_width - max_width
            # Calculate cropping boundaries.
            start_x = excess_width // 2
            end_x = start_x + max_width
            frame_cropped = frame_resized[:, start_x:end_x]
        else:
            frame_cropped = frame_resized
        image_pil = Image.fromarray(frame_cropped)
        image_tk = ImageTk.PhotoImage(image=image_pil)
        if label == 'original':
            self.original_video_label.configure(image=image_tk)
            # Reference to avoid garbage collection.
            self.original_video_label.image = image_tk 
        elif label == 'processed':
            self.processed_video_label.configure(image=image_tk)
            self.processed_video_label.image = image_tk
        self.window.update_idletasks()

def load_text(filename: str) -> str:
    with open(relative_to_text(filename), "r") as f:
        return f.read()