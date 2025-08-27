import tensorflow as tf

# List all physical devices of type GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        gpu_details = tf.config.experimental.get_device_details(gpu)
        gpu_name = gpu_details.get('device_name', 'Unknown GPU')
        print(f"Found a GPU with ID: {gpu}, Name: {gpu_name}")
else:
    print("Failed to detect a GPU.")



import os
import math
import numpy as np
import tensorflow as tf
import time
import glob

from keras import backend as K
from keras.layers import Input
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from networks.tinyucloudnet import TinyUCloudNet
from networks.cloudsegnet import CloudSegNet
from networks.unet import build_unet

import random
import argparse

parser = argparse.ArgumentParser(description="Test script for running models")
parser.add_argument(
    "--model_name",
    type=str,
    default="TinyUCloudNet",   # 👈 default value
    help="Name of the model to use (default: TinyUCloudNet)"
)

args = parser.parse_args()

model_name = args.model_name
print(f"Running test with model: {model_name}")

# Set seeds for reproducibility
seed = 42
# os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def set_seeds(seed=seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

# Activate Tensorflow deterministic behavior
def set_global_determinism(seed=seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=seed)

# -----------------             ---------------------------------------------------------------
# Set hyperparameters and paths
# --------------------------------------------------------------------------------
img_rows = 304
img_cols = 304
channels = 3
num_classes = 1  # Binary segmentation (cloud vs. non-cloud)
epochs = 100
batch_size = 16 

dataset_dir = "data"



train_image_dir = os.path.join(dataset_dir, "images")
train_mask_dir = os.path.join(dataset_dir, "GTmaps")


import glob

# Define the allowed image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

# Get sorted lists of image files in each directory
images = sorted([img for img in glob.glob(train_image_dir + "/*.*") if img.lower().endswith(tuple(image_extensions))])
masks = sorted([mask for mask in glob.glob(train_mask_dir + "/*.*") if mask.lower().endswith(tuple(image_extensions))])

print(len(images))
print(len(masks))
# Ensure that the number of images matches the number of masks
assert len(images) == len(masks), "The number of images and masks must be the same."

# Randomly split images and masks into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=seed
)

# Print the counts for verification
print(f"Number of training images: {len(train_images)}")
print(f"Number of training masks: {len(train_masks)}")
print(f"Number of validation images: {len(val_images)}")
print(f"Number of validation masks: {len(val_masks)}")



train_steps_per_epoch = int(len(train_images) / batch_size)
val_steps_per_epoch = int(len(val_images) / batch_size)
train_steps_per_epoch, val_steps_per_epoch


from data_generator import data_generator

train_generator = data_generator(
    train_images,
    train_masks,
    batch_size,
    size=(img_rows, img_cols)
    
)

validation_generator = data_generator(
    val_images,
    val_masks,
    batch_size,
    size=(img_rows, img_cols))


    # --------------------------------------------------------------------------------
# Build and compile the CloudSegNet model
# --------------------------------------------------------------------------------

input_shape = (
    (img_rows, img_cols, 3) 
    if K.image_data_format() == 'channels_last' 
    else (3, img_rows, img_cols)
)
input_tensor = Input(shape=input_shape)


# Select model architecture
model_name = 'TinyUCloudNet'
if model_name == 'unet':
    model = build_unet(input_tensor) 
elif model_name == 'cloudsegnet':
    model = CloudSegNet(input_tensor)  
elif model_name == 'TinyUCloudNet':
    model = TinyUCloudNet(input_tensor)

model_file = f'{model_name}.keras'


import os
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Custom callback to log learning rate
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Access the learning rate
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        logs['learning_rate'] = lr  # Add learning rate to logs
        print(f"\nEpoch {epoch + 1}: Learning rate is {lr:.6f}")

# Model checkpoint configuration
model_file = f'{model_name}.keras'
log_dir = "logs"

# Load checkpoint if found
if os.path.exists(model_file):
    print(f"Loading weights from checkpoint: {model_file}")
    model.load_weights(model_file)
else:
    print("No checkpoint found. Starting training from scratch.")

# Define model checkpoint
cp = ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Early stopping configuration
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True
)

# Learning rate schedule
lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)

# Compile the model
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

# Callbacks
csvlogger = CSVLogger(os.path.join(log_dir, "training.log"), separator=',', append=True)
lr_logger = LearningRateLogger()  # Instantiate the learning rate logger

start = time.time()
# Train the model
model_history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[cp, csvlogger, early_stopping, lr_logger]
)
end = time.time()


print('Training Time (s)', end-start)