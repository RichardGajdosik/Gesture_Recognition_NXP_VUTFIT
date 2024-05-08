import os
import subprocess
import time
import datetime
import argparse
import cv2
import numpy as np
from collections import deque
from collections import Counter

cycle_times = []
preprocess_times = []
inference_times = []
memory_usages = []
model_accuracies = []

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', help='Name of video or image input file',
                    default='video_device.mp4')
parser.add_argument('--graph', help='Name of the .tflite file, if different than qmc_uint8epoch60_edl2_all.tflite',
                    default='qmc_uint8epoch60_edl2_all.tflite')
parser.add_argument('--input_type', help='Type of input: webcam, videofile',
                    default='webcam')
parser.add_argument('--platform', help='Type of platform: arm, x64',
                    default='arm')
args = parser.parse_args()

# Load args
inputFile = args.filename
GRAPH_NAME = args.graph
inputType = args.input_type
platformType = args.platform

# Get path to current working directory
CWD_PATH = os.getcwd()
# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,GRAPH_NAME)
# We need to import different tensorflow libraries based on what platform we are using
if platformType == 'arm':
    # If you are on i.MX device, we need to use different library
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
    ext_delegate = [load_delegate("/usr/lib/libethosu_delegate.so")]
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=ext_delegate)
elif platformType == 'x64':
    # IF you are not on i.MX device
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=PATH_TO_CKPT)
else: 
    print('Error: Passed wrong platform type. Please set platform type to arm or x64')
class_names = ['dislike', 'fist', 'like', 'peace', 'stop', 'no_gesture']

######## MAIN LOGIC #########

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
    log_filename = f"logs/{GRAPH_NAME}_{current_time}.log"
    
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
            f"Avg. Cycle Time: {average_cycle_time:.8f}s\n"
            f"Avg. Preprocess Time: {average_preprocess_time:.8f}s\n"
            f"Avg. Inference Time: {average_inference_time:.8f}s\n"
            f"Avg. Memory Usage: {average_memory_usage} MB\n"
            f"Avg. Accuracy: {average_accuracy:.2f}%\n"
        )
        log_file.write(log_entry)

# Our model takes only 448x448 pixels, so we preprocess input from camera to that size
def preprocess_frame(frame):
    min_dim = min(frame.shape[:2])
    start_x = (frame.shape[1] - min_dim) // 2
    start_y = (frame.shape[0] - min_dim) // 2
    cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized_frame = cv2.resize(cropped_frame, (448, 448))
    return np.expand_dims(resized_frame, axis=0).astype(np.uint8), resized_frame

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


def main():
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    cap = object()
    if inputType == 'videofile':
        cap = cv2.VideoCapture(inputFile)
    elif inputType == 'webcam':
        cap = cv2.VideoCapture(0) # video interface slot for webcam(see in dir /dev) as parameter
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
    while(cap.isOpened()):

        # Grab frame from the video stream
        ret, frame = cap.read()
        
        if ret==True:
            start_cycle_time = time.time()
            start_preprocess_time = time.time()
            # Convert the frame from camera to 448x448 pixels
            processed_frame, display_frame = preprocess_frame(frame)
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

            # Get predicted gesture
            classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
            counts = Counter(classes)
            gesture = counts.most_common(1)

            # Get the index and the name of the gesture with highest percentage of certainty
            predicted_class_index = int(gesture[0][0])
            predicted_class_name = class_names[predicted_class_index]

            # Get percentages of certainty
            probabilities = interpreter.get_tensor(output_details[0]['index'])[0]# Confidence of detected objects

            # Normalize float numbers to percentages
            probabilities = probabilities * 100
            probability = probabilities[0]
            model_accuracies.append(probability)

            #print('Classes:' + str(classes))
            #print('Gesture:' + str(predicted_class_index) + ". name: " + str(predicted_class_name))
            #print('Scores' + str(probabilities))

            # Populate our queue
            last_predictions.append(predicted_class_name)

            # New avg found
            if len(last_predictions) == last_predictions.maxlen:
                avg_gesture = max(set(last_predictions), key=last_predictions.count)
                avg_predicted_gesture_image = gesture_images[predicted_class_name]
                if inputType != 'videofile':
                        cv2.imshow('Predicted Gesture Avg', avg_predicted_gesture_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            current_time = current_frame / frame_rate
            current_frame += 1

            # Only text version when inferencing from a videofile - used for debugging on system with only CLI support
            if inputType == 'videofile':
                    #os.system('clear')
                    print("{:>5} {:>25} {:>25} {:>15}".format("Čas", "Priemerný odhad", "Predikované gesto", "Odhad"))
                    print("{:>5.2f} {:>15} {:>25} {:>25}".format(current_time, avg_gesture, predicted_class_name, f"{probability:.2f}%"))
            else:
                display_img = np.zeros((150, 150 * len(class_names), 3), dtype=np.uint8)
                for i, name in enumerate(class_names):
                    if gesture_images.get(name) is not None:
                        display_img[:, i*150:(i+1)*150] = gesture_images[name]
                
                cv2.imshow('Dostupné gestá', display_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print("{:>5} {:>25} {:>25} {:>15}".format("Čas", "Priemerný odhad", "Predikované gesto", "Odhad"))
                print("{:>5.2f} {:>15} {:>25} {:>25}".format(current_time, avg_gesture, predicted_class_name, f"{probability:.2f}%"))
            
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                cv2.imshow('Model', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print('Error: No frames to inference or to show')
            break
        cycle_time = time.time() - start_cycle_time
        # Store cycle time
        cycle_times.append(cycle_time)
    log_performance_metrics()
if __name__ == '__main__':
    main()
