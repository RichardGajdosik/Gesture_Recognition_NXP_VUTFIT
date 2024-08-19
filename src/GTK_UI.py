import numpy as np
import time
import os
import subprocess
import datetime
import cv2
from collections import deque
import argparse

# Global variables
class_names = ['dislike', 'fist', 'like', 'peace', 'stop']
cycle_times = []
preprocess_times = []
inference_times = []
memory_usages = []
model_accuracies = []

def get_memory_usage():
    # Execute 'free' command and capture its output
    try:
        mem_usage = subprocess.check_output(['free', '-m']).decode('utf-8').split('\n')
        # Take the used memory from output
        used_memory = mem_usage[1].split()[2]

        return int(used_memory)
    except:
        return -1

def log_performance_metrics():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/{platform}/{target}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    # Calculate averages
    average_cycle_time = sum(cycle_times) / len(cycle_times)
    average_preprocess_time = sum(preprocess_times) / len(preprocess_times)
    average_inference_time = sum(inference_times) / len(inference_times)
    average_memory_usage = sum(memory_usages) / len(memory_usages)
    average_accuracy = sum(model_accuracies) / len(model_accuracies)
    
    # Write to log
    with open(log_filename, 'w') as log_file:
        log_entry = (
            f"Time: {current_time}\n"
            f"The name of the model: {model}\n"
            f"Avg. Cycle Time: {average_cycle_time:.8f}s\n"
            f"Avg. Preprocess Time: {average_preprocess_time:.8f}s\n"
            f"Avg. Inference Time: {average_inference_time:.8f}s\n"
            f"Avg. Memory Usage: {average_memory_usage} MB\n"
            f"Avg. Accuracy: {average_accuracy:.2f}%\n"
        )
        log_file.write(log_entry)

# Our model takes only 224x224 pixels, so we preprocess input from camera to that size
def preprocess_frame(frame, input_shape, input_dtype):
    # Input shape should be (height, width), for example (448, 448)
    target_height, target_width = input_shape[:2]
    
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
    return np.expand_dims(normalized_frame, axis=0).astype(input_dtype), resized_frame

def create_gstreamer_pipeline():
    return (
        'v412src device=/dev/video0 ! '
        'videoconvert ! '
        'autovideosink'
    )

# Loads images of gestures that are supported in the model and returns them
def load_recognized_gestures(class_names):
    images = {}
    for name in class_names:
        image_path = f'recognized_gestures/{name}.jpg'
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image, (150, 150))
            images[name] = resized_image
    return images

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', help='Either we want to inference on CPU or NPU',
                    default='CPU')
parser.add_argument('--model', help='Name of the .tflite file with path to it from the script',
                    default='models/model_uint8epoch20_edl2_100_per_gesture.tflite')
parser.add_argument('--input_type', help='Type of input: webcam, videofile',
                    default='videofile')
parser.add_argument('--filename', help='Name of video or image input file',
                    default='video_test.mp4')
args = parser.parse_args()

# Load args
target = args.target
model = args.model
inputType = args.input_type
inputFile = args.filename

delegate = "None"
# Check type of platform (i.MX 8M Plus vs i.MX 93)
if os.path.exists("/usr/lib/libvx_delegate.so"):
    platform = "i.MX8MP"
    platformType = "ARM"
    delegate = "/usr/lib/libvx_delegate.so"
    #VIV_VX_ENABLE_CACHE_GRAPH_BINARY='1'
    #VIV_VX_CACHE_BINARY_GRAPH_DIR="/opt/gopoint-apps/downloads"
elif os.path.exists("/usr/lib/libethosu_delegate.so"):
    platform = "i.MX93"
    platformType = "ARM"
    delegate = "/usr/lib/libethosu_delegate.so"
else:
    platform = "PC"
    platformType = "x86"

print("----------------------------------------------------------------------------------------------------------------------------------------")
print(f"Running inference with these values: Platform type: {platformType}, platform: {platform}, target: {target}\n Input type: {inputType}, delegate: {delegate}, model path: {model}")
print("----------------------------------------------------------------------------------------------------------------------------------------")

# Load the tflite model and libraries
if platformType == 'ARM':
    from tflite_runtime.interpreter import Interpreter
    if target == 'NPU':
        from tflite_runtime.interpreter import load_delegate
        ext_delegate = [load_delegate(delegate)]
        interpreter = Interpreter(model_path=model, experimental_delegates=ext_delegate)
    elif target == 'CPU':
        interpreter = Interpreter(model_path=model)
    else:
        print('Error: Passed wrong target type. Please set target type to \'CPU\' or \'NPU\'')
        exit()
elif platformType == 'x86' and target == 'NPU':
    print('Error: Passed wrong target in combination with platform type. Cannot target NPU when on x86 platform type')
    exit()
elif platformType == 'x86':
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=model)
else: 
    print('Error: Passed wrong platform type. Please set platform type to arm or x86')
    exit()

used_memory_before_inference = get_memory_usage()

# Pre-inference setup
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]
input_dtype = input_details[0]['dtype']

# Video capture setup
cap = object()
if inputType == 'videofile':
    print(f"Opening video file {inputFile}")
    cap = cv2.VideoCapture(inputFile)
elif inputType == 'webcam':
    #cap = cv2.VideoCapture(create_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0) # video interface slot for webcam(see in dir /dev) as parameter, 1 for macOS
    if not cap.isOpened():
        print('Error: Unable to access camera')
else:
    print('Error: Passed wrong input type. Please set input type to webcam or videofile')

if not cap.isOpened():
    print("Error: Could not open video or webcam.")
    exit(1)

# Create inital time stamp for frame rate calculation, time measuring and a way to store last prediction
# Get framerate of our input 
frame_rate = cap.get(cv2.CAP_PROP_FPS)
current_frame = 0

# Create a queue from which we will be able to get an average of last 20 predictions
last_predictions = deque(maxlen=20)
avg_gesture = "Init"

# Get supported gestures in image form to be able to display them to the user
gesture_images = load_recognized_gestures(class_names)

# Empty data creation
#empty_input = np.zeros(input_shape, dtype=input_dtype)
#interpreter.set_tensor(input_details[0]['index'], empty_input)

while(cap.isOpened()):
    # Grab frame from the video stream
    ret, frame = cap.read()
    if ret==True:
        start_cycle_time = time.time()
        start_preprocess_time = time.time()

        # Convert the frame from camera to 224x224 pixels
        processed_frame, display_frame = preprocess_frame(frame, input_shape, input_dtype)
        preprocess_time = time.time() - start_preprocess_time

        # Store preprocess time
        preprocess_times.append(preprocess_time)
        start_inference_time = time.time()

        # Run object detection
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()
        inference_time = time.time() - start_inference_time

        # Store inference time
        inference_times.append(inference_time)

        # Get memory usage using 'free' command
        memory_usage = get_memory_usage()
        memory_usages.append(memory_usage)

        # Get predicted gestures and their percentages of certainty
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get the index and the name of the gesture with highest percentage of certainty
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]

        # Populate our queue
        last_predictions.append(predicted_class_name)

        # Normalize float numbers to percentages
        probabilities = predictions[0] * 100

        # New avg found
        if len(last_predictions) == last_predictions.maxlen:
            avg_gesture = max(set(last_predictions), key=last_predictions.count)
            avg_predicted_gesture_image = gesture_images[predicted_class_name]
            model_accuracies.append(probabilities[predicted_class_index])
            cv2.rectangle(avg_predicted_gesture_image, (10, 145), (90, 120), (0, 0, 0), -1)
            cv2.putText(avg_predicted_gesture_image, f"{probabilities[predicted_class_index]:.2f}%", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            if inputType != 'videofile':
                cv2.imshow('Predicted Gesture Avg', avg_predicted_gesture_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        current_time = current_frame / frame_rate
        current_frame += 1
        
        # Only text version when inferencing from a videofile - used for debugging on system with only CLI support
        if inputType == 'videofile':
                os.system('clear')
                print("{:>5} {:>15} {:>25}".format("Čas", "Priemerný odhad", "Predikované gesto"))
                print("{:>5.2f} {:>15} {:>25}".format(current_time, avg_gesture, predicted_class_name))
        else:
            display_img = np.zeros((150, 150 * len(class_names), 3), dtype=np.uint8)
            for i, name in enumerate(class_names):
                if gesture_images.get(name) is not None:
                    display_img[:, i*150:(i+1)*150] = gesture_images[name]
                    cv2.rectangle(display_img, (i*150+10, 145), (i*150+90, 120), (0, 0, 0), -1)
                    cv2.putText(display_img, f"{probabilities[i]:.2f}%", (i*150+10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Dostupné gestá', display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print("{:>5} {:>15} {:>25}".format("Čas", "Priemerný odhad", "Predikované gesto"))
            print("{:>5.2f} {:>15} {:>25}".format(current_time, avg_gesture, predicted_class_name))
        
            model_resized_display_frame = cv2.resize(display_frame, (224*3, 224*3), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Model', model_resized_display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print('Error: No frames to inference or to show')
        break
    cycle_time = time.time() - start_cycle_time
    # Store cycle time
    cycle_times.append(cycle_time)
log_performance_metrics()