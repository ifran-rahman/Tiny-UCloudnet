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


import os
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.models import load_model
from tqdm import tqdm

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

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
model_file = f'{model_name}.keras'

checkpoint_path = model_file
threshold = 0.5
# model.load_weights(checkpoint_path)
# --------------------------------------------------------------------------------
# Load the best model after training
# --------------------------------------------------------------------------------
print("Loading the best model...")
model = load_model(checkpoint_path)

# --------------------------------------------------------------------------------
# Generate predictions and measure latency
# --------------------------------------------------------------------------------
start_time = time.time()
val_predictions = model.predict(validation_generator, steps=val_steps_per_epoch)
end_time = time.time()

inference_time = end_time - start_time
num_val_samples = len(val_images)
average_latency = inference_time / num_val_samples

print(f"Number of validation predictions: {len(val_predictions)}")
print(f"Inference time for all validation samples: {inference_time:.2f} seconds")
print(f"Average latency per image: {average_latency:.4f} seconds")

# --------------------------------------------------------------------------------
# Binarize predictions
# --------------------------------------------------------------------------------
val_predictions_binary = (val_predictions > threshold).astype(np.uint8)
print(f"Number of validation predictions (binary): {len(val_predictions_binary)}")

# --------------------------------------------------------------------------------
# Evaluate model on validation set
# --------------------------------------------------------------------------------
loss, accuracy = model.evaluate(validation_generator, steps=val_steps_per_epoch)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# --------------------------------------------------------------------------------
# Calculate additional metrics
# --------------------------------------------------------------------------------
precision_scores, recall_scores, f1_scores, accuracy_scores, dice_scores = [], [], [], [], []

# Reset the generator if necessary
validation_generator = data_generator(
    val_images,
    val_masks,
    1,
    size=(img_rows, img_cols)
)

# Process each batch
for _ in tqdm(range(val_steps_per_epoch)):
    imgs, masks = next(validation_generator)
    predictions = model.predict(imgs)

    predictions_flat = (predictions > threshold).astype(int).flatten()
    masks_flat = (masks > threshold).astype(int).flatten()

    precision_scores.append(precision_score(masks_flat, predictions_flat, zero_division=0))
    recall_scores.append(recall_score(masks_flat, predictions_flat, zero_division=0))
    f1_scores.append(f1_score(masks_flat, predictions_flat, zero_division=0))
    accuracy_scores.append(accuracy_score(masks_flat, predictions_flat))

    intersection = np.sum(masks_flat * predictions_flat)
    dice_score =  (2. * intersection) / (np.sum(masks_flat) + np.sum(predictions_flat) + 1e-7)
    dice_scores.append(dice_score)

# Calculate average metrics
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_accuracy = np.mean(accuracy_scores)
error_rate = 1 - avg_accuracy
dice_score = np.mean(dice_scores)

print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Error Rate: {error_rate:.4f}")
print(f"Average Dice Score: {dice_score:.4f}")

# --------------------------------------------------------------------------------
# Measure the model size in MB
# --------------------------------------------------------------------------------
model_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
print(f"Model size: {model_size:.2f} MB")
