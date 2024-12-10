#
# Copyright 2024 NXP
#
import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, BackupAndRestore

# Package for quantization
# from tensorflow_model_optimization as tfmot

# Get path to script
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# Dataset directories
train_data_dir = os.path.join(script_dir, '..', 'dataset', 'mobilnetv2', 'small_dataset_100_images_perG')
validation_data_dir = os.path.join(script_dir, '..', 'dataset', 'mobilnetv2', 'small_dataset_20_images_perG_testing')

# Load the pre-trained MobileNetV2 model without the top classification layer
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in model.layers:
    layer.trainable = False

# Adding custom layers
x = model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(5, activation='softmax')(x)

# Define the final model
model_final = models.Model(inputs=model.input, outputs=predictions)

# Apply quantization to the final model
# quantize_model = tfmot.quantization.keras.quantize_model
# model_final = quantize_model(model_final)

# Compile the model with SGD optimizer
optimizer = SGD(learning_rate=0.001, momentum=0.9)
model_final.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model_final.summary()

# Define image generators with additional augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, # We perhaps don't want to flip due to the nasture of human hand
    rotation_range=30,
)
# TODO
validation_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size and image size
batch_size = 32
img_height, img_width = 224, 224

# Prepare data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Callbacks

# TensorFlow 2.16
# checkpointer = ModelCheckpoint('model_checkpoint_one.keras', verbose=1, save_best_only=True)

# TensorFlow 2.15
checkpointer = ModelCheckpoint('model_checkpoint_one.h5', verbose=1, save_best_only=True)

early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
backupnrestore = BackupAndRestore(backup_dir='checkpoint')
callbacks = [checkpointer, early_stopper, backupnrestore]

# Start training
model_final.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks
)

# Save the model
model_path = os.path.join(script_dir, '..', 'final_model')

# TensorFlow 2.16
# tf.saved_model.save(model_final, model_path)

# TensorFlow 2.15
model_final.save(model_path)
