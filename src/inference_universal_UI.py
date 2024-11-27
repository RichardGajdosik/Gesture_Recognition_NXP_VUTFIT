import numpy as np
import time
import os
import subprocess
import datetime
import cv2
import platform
import psutil
import threading
import queue
from collections import deque
import argparse
import gi
import cairo

gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, GdkPixbuf, GLib, Gdk, Gio, Pango


class VideoPlayer(Gtk.Window):
    def __init__(self, target='CPU', model_path='models/model_float32epoch20_mobilnetv2_100_per_gesture.tflite'):
        self.inference_enabled = False
        super(VideoPlayer, self).__init__()
        self.set_default_size(1280, 800)  # Increased size to accommodate new panels
        self.target = target
        self.model_path = model_path

        self.load_model()
        self.init_ui()
        self.init_video_capture()
        self.show_all()
        

    def init_ui(self):
        # Create a Header Bar
        header_bar = Gtk.HeaderBar()
        header_bar.set_show_close_button(True)
        self.set_titlebar(header_bar)

        # Add NXP Logo to Header Bar
        logo_path = "../readme_images/2560px-NXP-Logo.svg.png"
        if os.path.exists(logo_path):
            logo_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                logo_path, width=225, height=75, preserve_aspect_ratio=True)
            logo_image = Gtk.Image.new_from_pixbuf(logo_pixbuf)
            header_bar.pack_start(logo_image)

        # Start/Stop Inference Button in Header Bar
        self.button_inference = Gtk.Button(label="Start Inference")
        self.button_inference.connect("clicked", self.on_inference_clicked)
        self.button_inference.get_style_context().add_class("inference-button")
        header_bar.pack_end(self.button_inference)

        # Main Layout Grid
        self.main_grid = Gtk.Grid()
        self.main_grid.set_column_spacing(10)
        self.main_grid.set_row_spacing(10)
        self.main_grid.set_margin_top(10)
        self.main_grid.set_margin_bottom(10)
        self.main_grid.set_margin_left(10)
        self.main_grid.set_margin_right(10)
        self.main_grid.set_hexpand(True)
        self.main_grid.set_vexpand(True)

        # Left Panel for Hardware Information
        self.hardware_info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.hardware_info_box.set_halign(Gtk.Align.FILL)
        self.hardware_info_box.set_valign(Gtk.Align.START)
        self.hardware_info_box.get_style_context().add_class("info-box")
        self.hardware_info_box.set_hexpand(False)
        self.hardware_info_box.set_vexpand(True)
        self.populate_hardware_info()
        self.main_grid.attach(self.hardware_info_box, 0, 0, 1, 2)

        # Center Grid for Video and Gestures
        self.center_grid = Gtk.Grid()
        self.center_grid.set_column_spacing(10)
        self.center_grid.set_row_spacing(10)
        self.center_grid.set_hexpand(True)
        self.center_grid.set_vexpand(True)
        self.center_grid.set_halign(Gtk.Align.CENTER)
        self.center_grid.set_valign(Gtk.Align.CENTER)
        self.main_grid.attach(self.center_grid, 1, 0, 1, 2)

        # Right Panel for Log Information
        self.log_info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.log_info_box.set_halign(Gtk.Align.FILL)
        self.log_info_box.set_valign(Gtk.Align.START)
        self.log_info_box.get_style_context().add_class("info-box")
        self.log_info_box.set_hexpand(False)
        self.log_info_box.set_vexpand(True)
        self.populate_log_info()
        self.main_grid.attach(self.log_info_box, 2, 0, 1, 2)

        # Drawing Area for Video
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(960, 720)
        self.drawing_area.connect("draw", self.on_draw)
        self.drawing_area.set_halign(Gtk.Align.CENTER)
        self.drawing_area.set_valign(Gtk.Align.CENTER)
        self.drawing_area.set_hexpand(True)
        self.drawing_area.set_vexpand(True)
        self.drawing_area.get_style_context().add_class("nxp-border")
        self.center_grid.attach(self.drawing_area, 0, 0, 2, 1)

        # Gesture Images Box
        self.gestures_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=15)
        self.gestures_box.set_halign(Gtk.Align.CENTER)
        self.gestures_box.set_valign(Gtk.Align.CENTER)
        self.gestures_box.set_hexpand(True)
        self.gestures_box.set_vexpand(False)
        self.center_grid.attach(self.gestures_box, 0, 1, 1, 1)

        self.gesture_images_widgets = {}
        for class_name in self.class_names:
            if class_name in self.gesture_images:
                image_widget = Gtk.Image.new_from_pixbuf(self.gesture_images[class_name])
                frame = Gtk.Frame()
                frame.set_shadow_type(Gtk.ShadowType.NONE)
                frame.set_size_request(150, 150)
                image_widget.set_size_request(150, 150)
                frame.add(image_widget)
                frame.get_style_context().add_class("gesture-frame")
                self.gestures_box.pack_start(frame, expand=True, fill=True, padding=0)
                self.gesture_images_widgets[class_name] = image_widget

        # Dynamic Gesture Image without Label
        self.dynamic_gesture_frame = Gtk.Frame()
        self.dynamic_gesture_frame.set_label_align(0.5, 0.5)
        self.dynamic_gesture_frame.set_shadow_type(Gtk.ShadowType.NONE)
        self.dynamic_gesture_image = Gtk.Image()
        self.dynamic_gesture_image.set_size_request(130, 130)
        self.dynamic_gesture_frame.set_size_request(150, 150)
        self.dynamic_gesture_frame.add(self.dynamic_gesture_image)
        self.dynamic_gesture_frame.set_hexpand(True)
        self.dynamic_gesture_frame.set_vexpand(False)
        self.dynamic_gesture_frame.set_halign(Gtk.Align.CENTER)
        self.dynamic_gesture_frame.get_style_context().add_class("nxp-border")
        self.center_grid.attach(self.dynamic_gesture_frame, 1, 1, 1, 1)

        # Add the main grid to the window
        self.add(self.main_grid)

        # Apply CSS Styling
        self.apply_css()

    def apply_css(self):
        # NXP Official Colors
        nxp_orange = "#FF8200"
        nxp_green = "#8DC63F"
        nxp_blue = "#00ADEF"
        nxp_gray = "#4D4D4F"
        nxp_light_gray = "#A7A8AA"

        css = f"""
        window {{
            background-color: {nxp_blue};
        }}
        headerbar {{
            background-color: {nxp_gray};
        }}
        button {{
            background-color: {nxp_orange};
            color: {nxp_green};
            border-radius: 5px;
            font-weight: bold;
        }}
        .inference-button {{
            background-color: #FFD700;  /* Gold color */
            color: #000000;             /* Black text */
        }}
        frame {{
            background-color: {nxp_orange};
            color: #FFFFFF;
            border-radius: 5px;
        }}
        .gesture-frame {{
            border: 1px solid {nxp_orange};
        }}
        label {{
            color: #FFFFFF;
            font-size: 14px;
        }}
        .nxp-border {{
            border: 6px solid {nxp_green};
        }}
        .info-box {{
            background-color: {nxp_gray};
            padding: 10px;
            border-radius: 5px;
        }}
        """

        style_provider = Gtk.CssProvider()
        style_provider.load_from_data(css.encode())

        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            style_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def populate_hardware_info(self):
        # Create labels for hardware information
        self.hardware_labels = {}
        labels_info = [
            ('Platform Type', self.platform_type),
            ('Platform Model', self.platform),
            ('CPU Model', platform.processor()),
            ('CPU Cores', str(psutil.cpu_count(logical=True))),
            ('Total Memory', f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB"),
            ('Available Memory', f"{round(psutil.virtual_memory().available / (1024**3), 2)} GB"),
            ('Target Device', self.target)
        ]
        for label_text, value in labels_info:
            label = Gtk.Label()
            label.set_markup(f"<b>{label_text}:</b> {value}")
            label.set_xalign(0)
            self.hardware_info_box.pack_start(label, False, False, 0)
            self.hardware_labels[label_text] = label

    def populate_log_info(self):
        # Create labels for log information
        self.log_labels = {}
        labels_info = [
            ('Current Time', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ('Inference Status', 'Stopped'),
            ('Average Inference Time', 'N/A'),
            ('Average FPS', 'N/A'),
            ('Last Detected Gesture', 'None'),
            ('Model Accuracy', 'N/A'),
            ('Memory Usage', 'N/A')
        ]
        for label_text, value in labels_info:
            label = Gtk.Label()
            label.set_markup(f"<b>{label_text}:</b> {value}")
            label.set_xalign(0)
            self.log_info_box.pack_start(label, False, False, 0)
            self.log_labels[label_text] = label

    def update_log_info(self):
        # Update dynamic log information
        self.log_labels['Current Time'].set_markup(
            f"<b>Current Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_labels['Inference Status'].set_markup(
            f"<b>Inference Status:</b> {'Running' if self.inference_enabled else 'Stopped'}")

        if self.inference_times:
            avg_inference_time = (sum(self.inference_times) / len(self.inference_times)) * 1000
            self.log_labels['Average Inference Time'].set_markup(
                f"<b>Average Inference Time:</b> {avg_inference_time:.2f} ms")

        if self.frame_times:
            avg_fps = len(self.frame_times) / sum(self.frame_times)
            self.log_labels['Average FPS'].set_markup(
                f"<b>Average FPS:</b> {avg_fps:.2f}")

        self.log_labels['Last Detected Gesture'].set_markup(
            f"<b>Last Detected Gesture:</b> {self.predicted_class_name}")

        if self.model_accuracies:
            avg_accuracy = sum(self.model_accuracies) / len(self.model_accuracies)
            self.log_labels['Model Accuracy'].set_markup(
                f"<b>Model Accuracy:</b> {avg_accuracy:.2f}%")

        current_memory = self.get_memory_usage()
        self.log_labels['Memory Usage'].set_markup(
            f"<b>Memory Usage:</b> {current_memory} MB")

    def init_video_capture(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Cannot open video capture device.")
            exit()

        # Create a queue to communicate with the worker thread
        self.frame_queue = queue.Queue()

        # Start the worker thread for frame capture and processing
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self.frame_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        # Schedule the UI update method
        GLib.timeout_add(30, self.update_ui)  # Refresh UI every 30 ms

    def load_model(self):
        # Load the tflite model and libraries
        delegate = "None"
        if os.path.exists("/usr/lib/libvx_delegate.so"):
            self.platform = "i.MX8MP"
            self.platform_type = "ARM"
            delegate = "/usr/lib/libvx_delegate.so"
        elif os.path.exists("/usr/lib/libethosu_delegate.so"):
            self.platform = "i.MX93"
            self.platform_type = "ARM"
            delegate = "/usr/lib/libethosu_delegate.so"
        else:
            self.platform = "PC"
            self.platform_type = "x86"

        print("----------------------------------------------------------------------------------------------------------------------------------------")
        print(f"Running inference with these values: Platform type: {self.platform_type}, platform: {self.platform}, target: {self.target}\n delegate: {delegate}, model path: {self.model_path}")
        print("----------------------------------------------------------------------------------------------------------------------------------------")

        if self.platform_type == 'ARM':
            from tflite_runtime.interpreter import Interpreter
            if self.target == 'NPU':
                from tflite_runtime.interpreter import load_delegate
                ext_delegate = [load_delegate(delegate)]
                self.interpreter = Interpreter(model_path=self.model_path, experimental_delegates=ext_delegate)
            elif self.target == 'CPU':
                self.interpreter = Interpreter(model_path=self.model_path)
            else:
                print('Error: Passed wrong target type. Please set target type to \'CPU\' or \'NPU\'')
                exit()
        elif self.platform_type == 'x86' and self.target == 'NPU':
            print('Error: Passed wrong target in combination with platform type. Cannot target NPU when on x86 platform type')
            exit()
        elif self.platform_type == 'x86':
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        else:
            print('Error: Passed wrong platform type. Please set platform type to arm or x86')
            exit()

        self.used_memory_before_inference = self.get_memory_usage()

        # Pre-inference setup
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
        self.input_dtype = self.input_details[0]['dtype']

        # Initialize variables for inference
        self.previous_avg_gesture = None
        self.draw_times = []
        self.frame_times = []
        self.cycle_times = []
        self.preprocess_times = []
        self.inference_times = []
        self.memory_usages = []
        self.model_accuracies = []
        self.last_predictions = deque(maxlen=20)
        self.avg_gesture = "Init"
        self.class_names = ['dislike', 'fist', 'like', 'peace', 'stop']
        self.predicted_class_name = "None"
        self.gesture_images = self.load_recognized_gestures(self.class_names)

    def get_memory_usage(self):
        # Get current process memory usage in MB
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return int(mem_info.rss / 1024 / 1024)

    def load_recognized_gestures(self, class_names):
        images = {}
        for name in class_names:
            image_path = f'recognized_gestures/{name}.jpg'
            if os.path.exists(image_path):
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(image_path, width=140, height=140, preserve_aspect_ratio=True)
                images[name] = pixbuf
        return images

    def preprocess_frame(self, frame):
        try:
            start_preprocess_time = time.time()

            # Input shape should be (height, width), for example (224, 224)
            target_height, target_width = self.input_shape[:2]

            # Determine the minimum dimension of the frame and crop it to a square
            min_dim = min(frame.shape[:2])
            start_x = (frame.shape[1] - min_dim) // 2
            start_y = (frame.shape[0] - min_dim) // 2
            cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

            # Resize the cropped frame to the target size expected by the model
            resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

            # Normalize the resized frame (assuming the model expects values between 0 and 1)
            normalized_frame = resized_frame / 255.0

            # Store preprocess time
            preprocess_time = time.time() - start_preprocess_time
            self.preprocess_times.append(preprocess_time)

            # Flip the frame for display
            display_frame = cv2.flip(frame, 1)

            # Draw rectangle on display_frame to indicate the area used for processing
            frame_height, frame_width = frame.shape[:2]
            rectangle_start_x = frame_width - (start_x + min_dim)
            rectangle_end_x = frame_width - start_x

            # Draw rectangle on display_frame
            cv2.rectangle(display_frame, (rectangle_start_x, start_y), (rectangle_end_x, start_y + min_dim), (0, 255, 0), 2)

            # Expand dimensions to match the model's input shape and convert to the expected dtype
            return np.expand_dims(normalized_frame, axis=0).astype(self.input_dtype), display_frame
        except Exception as e:
            print(f"Exception in preprocess_frame: {e}")
            return None, frame  # Return the original frame for display

    def update_ui(self):
        try:
            if not self.frame_queue.empty():
                self.current_frame = self.frame_queue.get_nowait()
                print("Frame retrieved from queue")
                # Update the drawing area
                self.drawing_area.queue_draw()
                # Update the log information
                self.update_log_info()
        except Exception as e:
            print(f"Exception in update_ui: {e}")
        return True  # Continue calling this method
    
    def frame_worker(self):
        print("Worker thread started")
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if ret:
                print("Frame captured")
                if self.inference_enabled:
                    processed_frame, display_frame = self.preprocess_frame(frame)
                    self.run_inference(processed_frame, display_frame)
                else:
                    display_frame = cv2.flip(frame, 1)
                    self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                # Put the current frame in the queue
                if not self.frame_queue.full():
                    self.frame_queue.put(self.current_frame)
                    print("Frame put into queue")
            else:
                print("Warning: Failed to read frame from camera.")

    def run_inference(self, processed_frame, display_frame):
        try:
            start_cycle_time = time.time()
            start_inference_time = time.time()

            # Run object detection
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_frame)
            self.interpreter.invoke()

            # Store inference time
            inference_time = time.time() - start_inference_time
            self.inference_times.append(inference_time)

            # Get memory usage
            memory_usage = self.get_memory_usage()
            self.memory_usages.append(memory_usage)

            # Get predicted gestures and their percentages of certainty
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Get the index and the name of the gesture with highest percentage of certainty
            predicted_class_index = np.argmax(predictions[0])
            self.predicted_class_name = self.class_names[predicted_class_index]

            # Populate our queue
            self.last_predictions.append(self.predicted_class_name)

            # Normalize float numbers to percentages
            probabilities = predictions[0] * 100

            # New avg found
            if len(self.last_predictions) == self.last_predictions.maxlen:
                self.avg_gesture = max(set(self.last_predictions), key=self.last_predictions.count)
                self.model_accuracies.append(probabilities[predicted_class_index])

                # Check if the average gesture has changed
                if self.avg_gesture != self.previous_avg_gesture:
                    self.previous_avg_gesture = self.avg_gesture

                    # Update GUI elements
                    # Update dynamic gesture image
                    if self.avg_gesture in self.gesture_images:
                        GLib.idle_add(self.dynamic_gesture_image.set_from_pixbuf, self.gesture_images[self.avg_gesture])
                    else:
                        # If gesture image is not available, clear the dynamic gesture image
                        GLib.idle_add(self.dynamic_gesture_image.clear)

            cycle_time = time.time() - start_cycle_time
            # Store cycle time
            self.cycle_times.append(cycle_time)

            # Convert display_frame to RGB
            self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Exception in run_inference: {e}")

    def on_draw(self, widget, cr):
        draw_start_time = time.time()
        if hasattr(self, 'current_frame'):
            # Get the size of the drawing area
            allocation = self.drawing_area.get_allocation()
            area_width = allocation.width
            area_height = allocation.height

            # Get the dimensions of the current frame
            frame_height, frame_width, _ = self.current_frame.shape

            # Calculate scaling factors to maintain aspect ratio
            scale = min(area_width / frame_width, area_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            # Resize the frame to fit the drawing area
            resized_frame = cv2.resize(self.current_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert the resized frame to a pixbuf
            pixbuf = self.gdk_pixbuf_from_frame(resized_frame)

            # Calculate position to center the image
            x = (area_width - new_width) // 2
            y = (area_height - new_height) // 2

            # Draw the pixbuf at the calculated position
            Gdk.cairo_set_source_pixbuf(cr, pixbuf, x, y)
            cr.paint()

        draw_time = time.time() - draw_start_time
        self.draw_times.append(draw_time)

        return False

    def gdk_pixbuf_from_frame(self, frame):
        h, w, c = frame.shape
        return GdkPixbuf.Pixbuf.new_from_data(
            frame.tobytes(),  # Pixel data
            GdkPixbuf.Colorspace.RGB,
            False,            # No alpha
            8,                # Bits per channel
            w, h,             # Width, height
            w * c             # Rowstride (number of bytes per row)
        )

    def on_inference_clicked(self, button):
        self.inference_enabled = not self.inference_enabled
        if self.inference_enabled:
            button.set_label("Stop Inference")
        else:
            button.set_label("Start Inference")
            # Optionally, log performance metrics
            self.log_performance_metrics()

    def log_performance_metrics(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"logs/{self.platform}/{self.target}_{current_time}.txt"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)

        # Calculate averages
        average_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0
        average_preprocess_time = sum(self.preprocess_times) / len(self.preprocess_times) if self.preprocess_times else 0
        average_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        average_memory_usage = sum(self.memory_usages) / len(self.memory_usages) if self.memory_usages else 0
        average_accuracy = sum(self.model_accuracies) / len(self.model_accuracies) if self.model_accuracies else 0
        average_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        average_draw_time = sum(self.draw_times) / len(self.draw_times) if self.draw_times else 0

        # Write to log
        with open(log_filename, 'w') as log_file:
            log_entry = (
                f"Time: {current_time}\n"
                f"{self.platform} inferencing on: {self.target}\n"
                f"The name of the model: {self.model_path}\n"
                f"Avg. Cycle Time per frame: {average_frame_time * 1000:.2f} ms\n"
                f"Avg. Preprocess Frames Time: {average_preprocess_time * 1000:.2f} ms\n"
                f"Avg. Inference Time: {average_inference_time * 1000:.2f} ms\n"
                f"Avg. Draw Time: {average_draw_time * 1000:.2f} ms\n"
                f"Avg. Memory Usage: {average_memory_usage} MB\n"
                f"Avg. Accuracy: {average_accuracy:.2f}%\n"
            )
            log_file.write(log_entry)

    def on_close(self, *args):
        self.stop_event.set()
        self.worker_thread.join()
        self.capture.release()
        cv2.destroyAllWindows()
        Gtk.main_quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', help='Either we want to inference on CPU or NPU',
                        default='CPU')
    parser.add_argument('--model', help='Name of the .tflite file with path to it from the script',
                        default='models/model_float32epoch20_mobilnetv2_100_per_gesture.tflite')
    args = parser.parse_args()

    win = VideoPlayer(target=args.target, model_path=args.model)
    win.connect("destroy", win.on_close)
    Gtk.main()