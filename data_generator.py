import numpy as np
import cv2 as cv

def data_generator(imglist, maplist, batchsize, size=(32, 32)):
    """
    Simplified data generator for binary classification problems without augmentations.
    
    Args:
        imglist (list): List of image file paths.
        maplist (list): List of corresponding mask file paths.
        batchsize (int): Number of images per batch.
        size (tuple): Target size for resizing (height, width).
        
    Yields:
        img_batch (np.array): Batch of preprocessed images.
        mask_batch (np.array): Batch of corresponding masks.
    """
    assert len(imglist) == len(maplist), "Mismatch between image and mask counts!"
    
    h, w = size
    while True:
        # Initialize batch arrays
        img_batch = np.zeros((batchsize, h, w, 3), dtype=np.float32)  # For RGB images
        mask_batch = np.zeros((batchsize, h, w, 1), dtype=np.float32)  # For binary masks
        
        # Randomly sample indices for the batch
        indices = np.random.choice(len(imglist), batchsize, replace=False)
        
        for i, idx in enumerate(indices):
            # Load image and mask
            img = cv.imread(imglist[idx], cv.IMREAD_COLOR)  # Load as RGB
            mask = cv.imread(maplist[idx], cv.IMREAD_GRAYSCALE)  # Load as grayscale
            
            # Resize to target size
            img = cv.resize(img, (w, h), interpolation=cv.INTER_CUBIC)
            mask = cv.resize(mask, (w, h), interpolation=cv.INTER_NEAREST)
            
            # Normalize image and scale mask
            img_batch[i] = img / 255.0  # Normalize to [0, 1]
            mask_batch[i] = np.expand_dims(mask / 255.0, axis=-1)  # Binary masks scaled to [0, 1]
        
        yield img_batch, mask_batch
