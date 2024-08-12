import gi
import os
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, GdkPixbuf, Gst

class CameraApp(Gtk.Window):
    def __init__(self):
        super().__init__(title="Camera Viewer")
        self.set_default_size(1280, 720)
        self.set_border_width(10)
        Gst.init(None)  # Initialize GStreamer

        # Main Layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)

        # Camera Window for Video
        self.camera_view = Gtk.DrawingArea()
        self.camera_view.set_size_request(640, 480)
        vbox.pack_start(self.camera_view, True, True, 0)

        # Set up the GStreamer Pipeline
        self.setup_gstreamer_pipeline()

        # Thumbnails box
        thumbnails_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        thumbnails_box.set_homogeneous(True)
        vbox.pack_start(thumbnails_box, True, True, 0)

        # Display static images from directory
        print(os.listdir("src/recognized_gestures"))
        for image_path in sorted(os.listdir("src/recognized_gestures")):
            print(image_path)
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                filename="src/recognized_gestures/" + image_path, 
                width=100, 
                height=100, 
                preserve_aspect_ratio=True)
            image = Gtk.Image.new_from_pixbuf(pixbuf)
            frame = Gtk.Frame()
            frame.add(image)
            thumbnails_box.pack_start(frame, False, False, 0)

        # Additional UI components like button and logo as before
        button = Gtk.Button(label="Start Inference")
        button.connect("clicked", self.on_inference_clicked)
        vbox.pack_start(button, False, True, 0)

        logo_qr_box = Gtk.Box()
        logo_image = Gtk.Image.new_from_icon_name("nxp", Gtk.IconSize.DIALOG)
        qr_image = Gtk.Image.new_from_icon_name("image-missing", Gtk.IconSize.DIALOG)
        logo_qr_box.pack_start(logo_image, False, False, 0)
        logo_qr_box.pack_start(qr_image, False, False, 0)
        vbox.pack_end(logo_qr_box, False, False, 0)

    def setup_gstreamer_pipeline(self):
        # Setup GStreamer pipeline to play video in loop
        self.player = Gst.ElementFactory.make("playbin", "player")
        video_uri = 'file:///path/to/your/video.mp4'  # Update path to your video file
        self.player.set_property('uri', video_uri)
        self.player.set_property('video-sink', None)  # Play video directly to the window

        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self.on_eos)  # Loop the video on end of stream
        bus.connect("message::error", self.on_error)

        self.player.set_state(Gst.State.PLAYING)

    def on_eos(self, bus, msg):
        # Loop the video
        self.player.set_state(Gst.State.READY)
        self.player.set_state(Gst.State.PLAYING)

    def on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print("Error:", msg.src.get_name(), err.message, debug)

    def on_inference_clicked(self, widget):
        print("Inference started")

# Creation and main loop
app = CameraApp()
app.connect("destroy", Gtk.main_quit)
app.show_all()
Gtk.main()

