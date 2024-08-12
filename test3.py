import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gtk, Gst, GstVideo

class VideoPlayer(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Video Stream")
        self.set_default_size(800, 450)

        # Initialize GStreamer
        Gst.init(None)

        self.create_ui()

        # Create GStreamer pipeline
        self.pipeline = Gst.ElementFactory.make("playbin", "player")
        video_uri = 'file:///home/default/Gesture_Recognition_NXP_VUTFIT/src/video_test.mp4'
        self.pipeline.set_property("uri", video_uri)

        # Set up a Gtk video sink
        self.gtksink = Gst.ElementFactory.make("gtksink", None)
        if not self.gtksink:
            print("Failed to create gtksink. Trying glimagesink.")
            self.gtksink = Gst.ElementFactory.make("glimagesink", None)

        self.pipeline.set_property("video-sink", self.gtksink)

        # Embed video in the UI
        self.video_area.add(self.gtksink.props.widget)

        # Connect to the bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self.on_eos)
        bus.connect("message::error", self.on_error)

    def create_ui(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(vbox)

        # Create a DrawingArea for video output
        self.video_area = Gtk.Box()
        vbox.pack_start(self.video_area, True, True, 0)

        # Create a Button for Play/Pause
        self.button = Gtk.Button(label="Pause")
        self.button.connect("clicked", self.toggle_playback)
        vbox.pack_start(self.button, False, True, 0)

    def on_eos(self, bus, msg):
        # End of stream, restart the video
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"Error: {err}, {debug}")

    def toggle_playback(self, button):
        if self.button.get_label() == "Pause":
            self.pipeline.set_state(Gst.State.PAUSED)
            self.button.set_label("Play")
        else:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.button.set_label("Pause")

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
