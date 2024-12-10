#
# Copyright 2024 NXP
#
import tensorflow as tf
import os

# Get path to script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# Load Keras model from path
model_path = os.path.join(script_dir, '..', '..', 'final_model')

# Incompatible with the 2.16 version of TensorFlow, ahh!
model = tf.keras.models.load_model(model_path)

# Initialize the TFLite converter to convert Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model without quantization
tflite_model = converter.convert()
with open('model_float32epoch20_mobilnetv2_100_per_gesture.tflite', 'wb') as f:
    f.write(tflite_model)

# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Generate the quantized model
# tflite_quant_model = converter.convert()

# with open('model_quantized.tflite', 'wb') as f:
#     f.write(tflite_quant_model)
