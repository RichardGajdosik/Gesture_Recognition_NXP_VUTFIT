import gi
import os
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf

class CameraApp(Gtk.Window):
   def __init__(self):
       super().__init__(title="Camera Viewer")
       self.set_default_size(1280, 720)
       self.set_border_width(10)
       # Main Layout
       vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
       self.add(vbox)
       # Camera Window
       camera_view = Gtk.Frame()
       camera_view.set_size_request(640, 480) 
       vbox.pack_start(camera_view, True, True, 0)
       # Box for small windows
       thumbnails_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
       thumbnails_box.set_homogeneous(True) # Same space for every image
       vbox.pack_start(thumbnails_box, True, True, 0)
       # Static images
       print(os.listdir("gestures"))
       for image_path in sorted(os.listdir("gestures")):
            print(image_path)
            #image = Gtk.Image.new_from_file("gestures/" + image_path)
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                filename="gestures/" + image_path, 
                width=100, 
                height=100, 
                preserve_aspect_ratio=True)

            image = Gtk.Image.new_from_pixbuf(pixbuf)
            frame = Gtk.Frame()
            frame.add(image)
            thumbnails_box.pack_start(frame, False, False, 0)
       # Dynamic windows
       dynamic_image = Gtk.Image.new_from_icon_name("image-missing", Gtk.IconSize.DIALOG)
       dynamic_frame = Gtk.Frame()
       dynamic_frame.add(dynamic_image)
       thumbnails_box.pack_start(dynamic_frame, False, False, 0)
       # Button for running the inference
       button = Gtk.Button(label="Start Inference")
       button.connect("clicked", self.on_inference_clicked)
       vbox.pack_start(button, False, True, 0)
       # logo of company and QR code
       logo_qr_box = Gtk.Box()
       logo_image = Gtk.Image.new_from_icon_name("nxp", Gtk.IconSize.DIALOG)
       qr_image = Gtk.Image.new_from_icon_name("image-missing", Gtk.IconSize.DIALOG)
       logo_qr_box.pack_start(logo_image, False, False, 0)
       logo_qr_box.pack_start(qr_image, False, False, 0)
       vbox.pack_end(logo_qr_box, False, False, 0)
   def on_inference_clicked(self, widget):
       print("Inference started")
# Creation
app = CameraApp()
app.connect("destroy", Gtk.main_quit)
app.show_all()
Gtk.main()
