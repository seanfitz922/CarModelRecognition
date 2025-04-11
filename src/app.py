import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import tkinter.filedialog as fd
from PIL import Image, ImageTk
from predict import predict_image
from utils import load_model




class MyFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # Configure grid layout so the drop area expands.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create a label that serves as the drag-and-drop area and image display.
        self.drop_label = ctk.CTkLabel(self, text="Drag & Drop Image Here", anchor="center")
        self.drop_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Register the label as a drop target and bind the drop event.
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.drop)

        # Keep a reference to the displayed image.
        self.display_image = None

        self.model, self.device = load_model()

    def drop(self, event):
        # event.data may contain multiple file paths; here we process the first one.
        files = self.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            self.process_image(file_path)

    def process_image(self, file_path):
        """Process the image, create a thumbnail and update the drop label."""
        try:
            print(f"File dropped: {file_path}")

            img = Image.open(file_path)
            # print(img)

            predicted_model = predict_image(img, self.model, self.device)
            print(predicted_model)

            img.thumbnail((400, 400))  # Adjust size as needed.
            self.display_image = ImageTk.PhotoImage(img)
            # Update the label to display the image (and remove the text).
            self.drop_label.configure(image=self.display_image, text="")
        except Exception as e:
            print(f"Failed to process image: {e}")


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.current_mode = "dark"
        ctk.set_default_color_theme("blue")
        self.title("Image Processor GUI")
        self.geometry("800x600")

        # Define background colors for dark and light modes.
        self.dark_bg = "#2B2B2B"  # Choose a suitable dark background.
        self.light_bg = "#FFFFFF"  # Or any light background color you prefer.

        # Set the initial background for the root window.
        self.configure(bg=self.dark_bg)

        # Create a container frame that fills the window and uses the same background.
        self.container = ctk.CTkFrame(self, fg_color=self.dark_bg)
        self.container.pack(fill="both", expand=True)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(1, weight=1)

        # Header frame (inside the container) for the buttons.
        header_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        # Configure header frame with two columns:
        # Column 0 expands for the file input button; column 1 stays minimal for the appearance toggle.
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)

        # Input File Button on the left (column 0).
        self.input_file = ctk.CTkButton(header_frame, text="Select Image", command=self.open_file)
        self.input_file.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Load images for appearance toggling.
        image_size = (99, 34)
        pil_light_mode_image = Image.open("C:/Users/seanf/Desktop/School/Pattern Recognition/CarModelRecognition/src/light_mode.png")
        pil_dark_mode_image = Image.open("C:/Users/seanf/Desktop/School/Pattern Recognition/CarModelRecognition/src/dark_mode.png")
        self.light_mode_img = ctk.CTkImage(pil_light_mode_image, size=image_size)
        self.dark_mode_img = ctk.CTkImage(pil_dark_mode_image, size=image_size)
        # When in dark mode, display the dark mode toggle button with the light mode image
        # indicating that clicking it will switch to light mode.
        button_image = self.light_mode_img

        # Appearance Toggle Button on the right (column 1).
        self.appearance_button = ctk.CTkButton(
            header_frame,
            text='',
            image=button_image,
            fg_color="transparent",
            hover = False,
            border_width=0,
            command=self.toggle_appearance_mode
        )
        self.appearance_button.grid(row=0, column=1, padx=10, pady=10, sticky="e")

        # MyFrame is the main area (inside the container).
        self.my_frame = MyFrame(master=self.container)
        self.my_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

    def toggle_appearance_mode(self):
        if self.current_mode == "dark":
            # Switch from dark to light.
            self.current_mode = "light"
            ctk.set_appearance_mode("light")
            # Update the backgrounds to match light mode.
            self.configure(bg=self.light_bg)
            self.container.configure(fg_color=self.light_bg)
            # In light mode, show the dark mode image on the button.
            self.appearance_button.configure(image=self.dark_mode_img)
        else:
            # Switch from light to dark.
            self.current_mode = "dark"
            ctk.set_appearance_mode("dark")
            # Update the backgrounds to match dark mode.
            self.configure(bg=self.dark_bg)
            self.container.configure(fg_color=self.dark_bg)
            # In dark mode, show the light mode image on the button.
            self.appearance_button.configure(image=self.light_mode_img)


    def open_file(self):
        """Opens a file dialog for image selection and sends the image to MyFrame."""
        file_path = fd.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")]
        )
        if file_path:
            self.my_frame.process_image(file_path)


if __name__ == "__main__":
    app = App()
    app.mainloop()
