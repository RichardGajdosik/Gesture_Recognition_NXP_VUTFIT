import gi
import os
import numpy as np
import cv2  # Import OpenCV for image processing

gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gtk, GdkPixbuf, Gst, GstVideo

class VideoPlayer(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Video Stream")
        self.set_default_size(800, 450)

        # Initialize GStreamer
        Gst.init(None)

        self.create_ui()

        # Create GStreamer pipeline
        self.pipeline = Gst.Pipeline()
        self.source = Gst.ElementFactory.make("filesrc", "source")
        self.source.set_property("location", "/home/default/Gesture_Recognition_NXP_VUTFIT/src/video_test.mp4")
        self.decodebin = Gst.ElementFactory.make("decodebin", "decoder")

        self.source.link(self.decodebin)

        self.decodebin.connect("pad-added", self.on_pad_added)

        self.appsink = Gst.ElementFactory.make("appsink", "sink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self.on_new_sample, self.appsink)

        self.pipeline.add(self.source)
        self.pipeline.add(self.appsink)

        # Connect to the bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self.on_eos)
        bus.connect("message::error", self.on_error)

    def on_pad_added(self, element, pad):
        caps = pad.get_current_caps()
        structure_name = caps.to_string()
        if structure_name.startswith("video/"):
            pad.link(self.appsink.get_static_pad("sink"))

    def on_new_sample(self, appsink, data):
        sample = appsink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                frame = np.ndarray(
                    shape=(map_info.size // (1920 * 3), 1920, 3),
                    dtype=np.uint8,
                    buffer=map_info.data)
                # Process frame with your inference engine here
                # For example, converting to grayscale using OpenCV
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print("Processing frame...")

            buffer.unmap(map_info)
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR

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
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"Error: {err}, {debug}")

    def on_inference_clicked(self, button):
        print("Inference started")

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
