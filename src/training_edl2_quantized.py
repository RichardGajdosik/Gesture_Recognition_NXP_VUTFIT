#import numpy as np
import os
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite2')

train_data = object_detector.DataLoader.from_pascal_voc(images_dir="/home/default/nxp/dataset/dataset_small_10_percent_combined_in_one_dir/dataset", annotations_dir="/home/default/nxp/dataset/dataset_small_10_percent_combined_in_one_dir/annotations_small_10_percent_VOC",
        label_map={ 1:"dislike", 2:"fist", 3:"like", 4:"peace", 5:"stop", 6:"no_gesture" }, num_shards=10)

validation_data = object_detector.DataLoader.from_pascal_voc(images_dir="/home/default/nxp/dataset/dataset_small_10_percent_combined_in_one_dir/dataset", annotations_dir="/home/default/nxp/dataset/dataset_small_10_percent_combined_in_one_dir/annotations_small_10_percent_VOC",
        label_map={ 1:"dislike", 2:"fist", 3:"like", 4:"peace", 5:"stop", 6:"no_gesture" }, num_shards=10)

model = object_detector.create(train_data, model_spec=spec, batch_size=8, epochs=60, train_whole_model=True, validation_data=validation_data)
config = QuantizationConfig.for_int8(validation_data, inference_input_type=tf.uint8, inference_output_type=tf.float32, supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8])
model.export(export_dir='.', tflite_filename="qmc_uint8epoch60_edl2_all.tflite", quantization_config=config)