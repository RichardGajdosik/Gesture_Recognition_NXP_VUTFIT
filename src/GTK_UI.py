import numpy as np
import time
import os
import subprocess
import datetime
import cv2
from collections import deque
import argparse
import gi
import cairo

gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, GdkPixbuf, GLib, Gdk

class VideoPlayer(Gtk.Window):
    def __init__(self, target='CPU', model_path='models/model_float32epoch20_mobilnetv2_100_per_gesture.tflite'):
        super(VideoPlayer, self).__init__(title="GTK Video Stream")
        self.set_default_size(1024, 768)
        self.target = target
        self.model_path = model_path

        self.load_model()
        self.init_ui()
        self.init_video_capture()
        self.show_all()
        self.inference_enabled = False

    def init_ui(self):
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(self.vbox)

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect("draw", self.on_draw)
        self.vbox.pack_start(self.drawing_area, expand=True, fill=True, padding=0)

        self.gestures_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.vbox.pack_start(self.gestures_box, expand=False, fill=True, padding=0)

        self.gesture_images_widgets = {}
        for class_name in self.class_names:
            if class_name in self.gesture_images:
                image_widget = Gtk.Image.new_from_pixbuf(self.gesture_images[class_name])
                self.gestures_box.pack_start(image_widget, expand=False, fill=True, padding=5)
                self.gesture_images_widgets[class_name] = image_widget

        self.dynamic_gesture_image = Gtk.Image()
        self.gestures_box.pack_start(self.dynamic_gesture_image, expand=False, fill=True, padding=5)

        self.button_inference = Gtk.Button(label="Start Inference")
        self.button_inference.connect("clicked", self.on_inference_clicked)
        self.vbox.pack_start(self.button_inference, expand=False, fill=True, padding=0)

    def init_video_capture(self):
        self.capture = cv2.VideoCapture(0)
        #self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 224)  # Set the width
        #self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)  # Set the height
        GLib.timeout_add(60, self.update_frame)  # Timeout in milliseconds to refresh frame

    def load_model(self):
        # Load the tflite model and libraries
        delegate = "None"
        if os.path.exists("/usr/lib/libvx_delegate.so"):
            self.platform = "i.MX8MP"
            platformType = "ARM"
            delegate = "/usr/lib/libvx_delegate.so"
        elif os.path.exists("/usr/lib/libethosu_delegate.so"):
            self.platform = "i.MX93"
            platformType = "ARM"
            delegate = "/usr/lib/libethosu_delegate.so"
        else:
            self.platform = "PC"
            platformType = "x86"

        print("----------------------------------------------------------------------------------------------------------------------------------------")
        print(f"Running inference with these values: Platform type: {platformType}, platform: {self.platform}, target: {self.target}\n delegate: {delegate}, model path: {self.model_path}")
        print("----------------------------------------------------------------------------------------------------------------------------------------")

        if platformType == 'ARM':
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
        elif platformType == 'x86' and self.target == 'NPU':
            print('Error: Passed wrong target in combination with platform type. Cannot target NPU when on x86 platform type')
            exit()
        elif platformType == 'x86':
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
        self.cycle_times = []
        self.preprocess_times = []
        self.inference_times = []
        self.memory_usages = []
        self.model_accuracies = []
        self.last_predictions = deque(maxlen=20)
        self.avg_gesture = "Init"
        self.class_names = ['dislike', 'fist', 'like', 'peace', 'stop']
        self.predicted_class_name = ""
        self.gesture_images = self.load_recognized_gestures(self.class_names)

    def get_memory_usage(self):
        # Execute 'free' command and capture its output
        try:
            mem_usage = subprocess.check_output(['free', '-m']).decode('utf-8').split('\n')
            # Take the used memory from output
            used_memory = mem_usage[1].split()[2]

            return int(used_memory)
        except:
            return -1

    def load_recognized_gestures(self, class_names):
        images = {}
        for name in class_names:
            image_path = f'recognized_gestures/{name}.jpg'
            if os.path.exists(image_path):
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(image_path, width=150, height=150, preserve_aspect_ratio=True)
                images[name] = pixbuf
        return images

    def preprocess_frame(self, frame):
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
        
        # Expand dimensions to match the model's input shape and convert to the expected dtype
        return np.expand_dims(normalized_frame, axis=0).astype(self.input_dtype), cv2.flip(resized_frame, 1)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Convert the frame from camera to YxZ pixels required by the model (e.g. 224x224), leave display_frame for display and normalize the processed frame   
            processed_frame, display_frame = self.preprocess_frame(frame)
            if self.inference_enabled:
                self.run_inference(processed_frame, display_frame)
            else:
                self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.drawing_area.queue_draw()
        return True  # Return True to continue callback

    def run_inference(self, processed_frame, display_frame):
        start_cycle_time = time.time()
        start_preprocess_time = time.time()

        preprocess_time = time.time() - start_preprocess_time

        # Store preprocess time
        self.preprocess_times.append(preprocess_time)
        start_inference_time = time.time()

        # Run object detection
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_frame)
        self.interpreter.invoke()
        inference_time = time.time() - start_inference_time

        # Store inference time
        self.inference_times.append(inference_time)

        # Get memory usage using 'free' command
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
                    self.dynamic_gesture_image.set_from_pixbuf(self.gesture_images[self.avg_gesture])
                    #GLib.idle_add(self.dynamic_gesture_image.set_from_pixbuf, self.gesture_images[self.predicted_class_name])
                else:
                    # If gesture image is not available, clear the dynamic gesture image
                    self.dynamic_gesture_image.clear()
                    #GLib.idle_add(self.dynamic_gesture_image.clear)

        cycle_time = time.time() - start_cycle_time
        # Store cycle time
        self.cycle_times.append(cycle_time)

        # Draw the predicted class name on the display_frame
        cv2.putText(display_frame, f"Predicted: {self.predicted_class_name}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        # Convert display_frame to RGB
        self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    
    def on_draw(self, widget, cr):
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
        return False
    
    def gdk_pixbuf_from_frame(self, frame):
        h, w, c = frame.shape
        return GdkPixbuf.Pixbuf.new_from_data(
            frame.flatten(),  # Pixel data
            GdkPixbuf.Colorspace.RGB,
            False,            # No alpha
            8,                # Bits per channel
            w, h,             # Width, height
            w*c               # Rowstride (number of bytes per row)
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
        log_filename = f"logs/{self.platform}/{self.target}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        
        # Calculate averages
        average_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0
        average_preprocess_time = sum(self.preprocess_times) / len(self.preprocess_times) if self.preprocess_times else 0
        average_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        average_memory_usage = sum(self.memory_usages) / len(self.memory_usages) if self.memory_usages else 0
        average_accuracy = sum(self.model_accuracies) / len(self.model_accuracies) if self.model_accuracies else 0
        
        # Write to log
        with open(log_filename, 'w') as log_file:
            log_entry = (
                f"Time: {current_time}\n"
                f"The name of the model: {self.model_path}\n"
                f"Avg. Cycle Time: {average_cycle_time:.8f}s\n"
                f"Avg. Preprocess Time: {average_preprocess_time:.8f}s\n"
                f"Avg. Inference Time: {average_inference_time:.8f}s\n"
                f"Avg. Memory Usage: {average_memory_usage} MB\n"
                f"Avg. Accuracy: {average_accuracy:.2f}%\n"
            )
            log_file.write(log_entry)

    def on_close(self, *args):
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