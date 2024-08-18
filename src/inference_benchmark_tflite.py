import numpy as np
import time
import os
import subprocess
from collections import deque
import argparse

def get_memory_usage():
    mem_usage = subprocess.check_output(['free', '-m']).decode('utf-8').split('\n')
    used_memory = mem_usage[1].split()[2]
    return int(used_memory)

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', help='Either we want to inference on CPU or NPU',
                    default='CPU')
parser.add_argument('--model', help='Name of the .tflite file with path to it from the script',
                    default='models/yolov5nu_converted.tflite')
args = parser.parse_args()

# Load args
target = args.target
model = args.model

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
print(f"Benchmarking with these values: Platform type: {platformType}, platform: {platform}, target: {target} \nDelegate: {delegate}, model path: {model}")
print("----------------------------------------------------------------------------------------------------------------------------------------")

# Load the tflite model
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

# Empty data creation
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
empty_input = np.zeros(input_shape, dtype=input_dtype)

interpreter.set_tensor(input_details[0]['index'], empty_input)

times = deque(maxlen=100)
memory = deque(maxlen=100)

start_time = time.time()
for i in range(100):
    # Inference
    interpreter.invoke()

    memory.append(get_memory_usage())
    times.append(time.time() - start_time)
    start_time = time.time()

used_memory_after_inference = get_memory_usage()

Memory_used_for_inference = round(np.mean(memory)) - used_memory_before_inference

print((f"The name of the model: {model}"))
print(f"Average inference time: {round(np.mean(times) * 1000)} milliseconds")
print(f"Used memory before inference: {used_memory_before_inference} MB")
print(f"Used memory during inference: {np.mean(memory)} MB")
print(f"Memory used for inference: {Memory_used_for_inference} MB")

log_path = f"logs/{platform}/{target}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

with open(log_path, "w") as file:
    file.write(f"The name of the model: {model}\n")
    file.write(f"Average inference time: {round(np.mean(times) * 1000)} milliseconds\n")
    file.write(f"Memory used for inference: {Memory_used_for_inference} MB")
    