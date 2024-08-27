import gi

# Ensure that the correct versions of the libraries are used
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gst

class VideoPlayer(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Video Stream")
        self.set_default_size(800, 450)

        # Initialize GStreamer
        Gst.init(None)

        self.create_ui()

        # Create GStreamer pipeline
        self.pipeline = Gst.Pipeline.new("camera-pipeline")

        # Create the camera source element for macOS
        self.source = Gst.ElementFactory.make("avfvideosrc", "source")
        if not self.source:
            print("Failed to create avfvideosrc. Ensure your camera is connected and GStreamer is installed.")
            return

        # Create the video converter element
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
        
        # Create a sink element that integrates with GTK
        self.gtksink = Gst.ElementFactory.make("gtksink", "gtksink")
        if not self.gtksink:
            print("Failed to create gtksink.")
            return

        # Add elements to the pipeline
        self.pipeline.add(self.source)
        self.pipeline.add(self.videoconvert)
        self.pipeline.add(self.gtksink)

        # Link the elements in the pipeline
        self.source.link(self.videoconvert)
        self.videoconvert.link(self.gtksink)

        # Embed video in the UI
        self.video_area.add(self.gtksink.props.widget)

        # Connect to the bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self.on_eos)
        bus.connect("message::error", self.on_error)

        # State control flag
        self.inference_active = False

    def create_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(vbox)

        # Create a Box for video output
        self.video_area = Gtk.Box()
        vbox.pack_start(self.video_area, True, True, 0)

        # Create a Button for Inference
        self.button_inference = Gtk.Button(label="Start Inference")
        self.button_inference.connect("clicked", self.on_inference_clicked)
        vbox.pack_start(self.button_inference, False, True, 0)

    def on_eos(self, bus, msg):
        # End of stream, restart the video
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"Error: {err}, {debug}")

    def on_inference_clicked(self, button):
        self.inference_active = not self.inference_active
        if self.inference_active:
            self.button_inference.set_label("Stop Inference")
            print("Inference started")
        else:
            self.button_inference.set_label("Start Inference")
            print("Inference stopped")

    def run(self):
        # Start the video
        self.pipeline.set_state(Gst.State.PLAYING)
        self.connect("destroy", Gtk.main_quit)
        self.show_all()
        Gtk.main()

# Run the application
if __name__ == "__main__":
    player = VideoPlayer()
    player.run()